#!/usr/bin/env python3
"""
Weakness Profile Extractor for MMLU+DOVE Dataset

Implements the EvalTree algorithm for extracting nodes with significantly low accuracy
using confidence intervals and the two-pass extraction method.
"""

import json
import sys
from typing import Dict, Any, List, Tuple, Optional
import statsmodels.api as sm
from scipy import stats
import numpy as np

class WeaknessProfileExtractor:
    """
    Extracts weakness profiles from MMLU trees with model performance data.
    
    Based on the EvalTree algorithm:
    1. Calculate confidence intervals for each node
    2. Test nodes for significantly low accuracy 
    3. Extract nodes that pass tests but have children that don't
    """
    
    def __init__(self, alpha: float = 0.05, threshold: float = 0.5, 
                 min_size_parent: int = 20, min_size_child: int = 5,
                 direction: str = "lower"):
        """
        Initialize the extractor with hyperparameters.
        
        Args:
            alpha: Confidence level for statistical tests (default 0.05)
            threshold: Accuracy threshold for weakness detection (default 0.5)
            min_size_parent: Minimum size for parent nodes to be considered (σ1)
            min_size_child: Minimum size for child nodes to be tested (σ2)  
            direction: "lower" for weaknesses, "higher" for strengths
        """
        self.alpha = alpha
        self.threshold = threshold
        self.min_size_parent = min_size_parent
        self.min_size_child = min_size_child
        self.direction = direction
        
    def calculate_confidence_intervals(self, tree: Dict[Any, Any], 
                                     model_name: str = "Llama-3.1-8B-Instruct") -> Dict[Any, Any]:
        """
        Calculate confidence intervals for accuracy at each node in the tree.
        
        Args:
            tree: The MMLU tree structure
            model_name: Name of the model to extract accuracy for
            
        Returns:
            Tree with confidence interval information added
        """
        
        def calculate_node(node: Dict[Any, Any]) -> Dict[Any, Any]:
            """Calculate statistics for a single node."""
            result = {k: v for k, v in node.items() if k != 'subtrees'}
            
            if isinstance(node.get('subtrees'), (int, type(None))) and 'ranking' in node:
                # Leaf node - extract accuracy for the specified model
                ranking = node['ranking']
                model_accuracy = None
                
                for model_data in ranking:
                    if model_data[0] == model_name:
                        model_accuracy = model_data[1]
                        break
                
                if model_accuracy is not None:
                    result.update({
                        'size': 1,
                        'sum_correct': int(model_accuracy > 0.5),  # Binary correct/incorrect
                        'accuracy': model_accuracy,
                        'model_name': model_name
                    })
                else:
                    # Model not found in ranking
                    result.update({
                        'size': 0,
                        'sum_correct': 0,
                        'accuracy': 0.0,
                        'model_name': model_name
                    })
                    
                result['subtrees'] = node.get('subtrees', None)
                
            elif isinstance(node.get('subtrees'), list):
                # Internal node with list of subtrees
                result['subtrees'] = []
                total_size = 0
                total_correct = 0
                
                for subtree in node['subtrees']:
                    subtree_result = calculate_node(subtree)
                    result['subtrees'].append(subtree_result)
                    total_size += subtree_result.get('size', 0)
                    total_correct += subtree_result.get('sum_correct', 0)
                
                result.update({
                    'size': total_size,
                    'sum_correct': total_correct,
                    'accuracy': total_correct / total_size if total_size > 0 else 0.0,
                    'model_name': model_name
                })
            else:
                # Shouldn't happen in our MMLU structure, but handle gracefully
                result.update({
                    'size': 0,
                    'sum_correct': 0,
                    'accuracy': 0.0,
                    'model_name': model_name
                })
                result['subtrees'] = node.get('subtrees')
            
            # Calculate confidence interval if we have enough data
            if result['size'] >= 5:
                try:
                    lower_bound, upper_bound = sm.stats.proportion_confint(
                        result['sum_correct'], result['size'], 
                        alpha=self.alpha, method='beta'
                    )
                    result['confidence_interval'] = {
                        str(self.alpha): (lower_bound, upper_bound)
                    }
                except:
                    result['confidence_interval'] = None
            else:
                result['confidence_interval'] = None
                
            return result
        
        return calculate_node(tree)
    
    def test_node_significance(self, node: Dict[Any, Any]) -> bool:
        """
        Test if a node has significantly low/high accuracy compared to threshold.
        
        Args:
            node: Node with confidence interval information
            
        Returns:
            True if the node passes the significance test
        """
        if node.get('confidence_interval') is None:
            return False
            
        ci_key = str(self.alpha)
        if ci_key not in node['confidence_interval']:
            return False
            
        lower_bound, upper_bound = node['confidence_interval'][ci_key]
        
        if self.direction == "lower":
            # For weaknesses: upper bound must be below threshold
            return upper_bound < self.threshold
        elif self.direction == "higher":
            # For strengths: lower bound must be above threshold  
            return lower_bound > self.threshold
        else:
            raise ValueError(f"Invalid direction: {self.direction}")
    
    def extract_weakness_nodes(self, tree_with_ci: Dict[Any, Any]) -> List[Dict[Any, Any]]:
        """
        Extract nodes that represent weaknesses using the two-pass algorithm.
        
        Args:
            tree_with_ci: Tree with confidence intervals calculated
            
        Returns:
            List of extracted weakness nodes
        """
        
        # First pass: Mark all nodes that pass the significance test
        def first_pass(node: Dict[Any, Any]) -> None:
            """Mark nodes that pass the binomial/confidence interval test."""
            node['passes_test'] = self.test_node_significance(node)
            
            if isinstance(node.get('subtrees'), list):
                for subtree in node['subtrees']:
                    first_pass(subtree)
        
        first_pass(tree_with_ci)
        
        # Second pass: Extract nodes according to the algorithm
        extracted_nodes = []
        
        def second_pass(node: Dict[Any, Any]) -> None:
            """Extract nodes that pass test but have children that don't."""
            
            # Check if this node qualifies for extraction
            if (node.get('size', 0) >= self.min_size_parent and 
                node.get('passes_test', False)):
                
                # Check if all significant children fail the test
                all_children_fail = True
                
                if isinstance(node.get('subtrees'), list):
                    for child in node['subtrees']:
                        if (child.get('size', 0) >= self.min_size_child and 
                            child.get('passes_test', False)):
                            all_children_fail = False
                            break
                
                if all_children_fail:
                    # Extract this node and mark it
                    node['extracted'] = True
                    extracted_nodes.append({
                        'capability': node.get('capability', 'Unknown capability'),
                        'size': node.get('size', 0),
                        'accuracy': node.get('accuracy', 0.0),
                        'confidence_interval': node.get('confidence_interval'),
                        'dove_scores': self._collect_dove_scores(node),
                        'subjects': self._collect_subjects(node),
                        'node_data': node
                    })
                    return  # Skip subtree to avoid overlap
            
            # Continue recursion if node not extracted
            node['extracted'] = False
            if isinstance(node.get('subtrees'), list):
                for child in node['subtrees']:
                    second_pass(child)
        
        second_pass(tree_with_ci)
        return extracted_nodes
    
    def _collect_dove_scores(self, node: Dict[Any, Any]) -> List[float]:
        """Collect all DOVE scores from leaf nodes under this node."""
        dove_scores = []
        
        def collect_from_node(n):
            if isinstance(n.get('subtrees'), (int, type(None))) and 'dove_score' in n:
                dove_scores.append(n['dove_score'])
            elif isinstance(n.get('subtrees'), list):
                for child in n['subtrees']:
                    collect_from_node(child)
        
        collect_from_node(node)
        return dove_scores
    
    def _collect_subjects(self, node: Dict[Any, Any]) -> List[str]:
        """Collect all unique subjects from leaf nodes under this node."""
        subjects = set()
        
        def collect_from_node(n):
            if isinstance(n.get('subtrees'), (int, type(None))) and 'subject' in n:
                subjects.add(n['subject'])
            elif isinstance(n.get('subtrees'), list):
                for child in n['subtrees']:
                    collect_from_node(child)
        
        collect_from_node(node)
        return list(subjects)
    
    def extract_profile(self, tree: Dict[Any, Any], 
                       model_name: str = "Llama-3.1-8B-Instruct") -> Dict[str, Any]:
        """
        Complete weakness profile extraction pipeline.
        
        Args:
            tree: MMLU tree structure
            model_name: Model to analyze
            
        Returns:
            Dictionary containing weakness profile results
        """
        print(f"Extracting weakness profile for {model_name}...")
        print(f"Parameters: alpha={self.alpha}, threshold={self.threshold}, direction={self.direction}")
        print(f"Min sizes: parent={self.min_size_parent}, child={self.min_size_child}")
        
        # Step 1: Calculate confidence intervals
        print("Step 1: Calculating confidence intervals...")
        tree_with_ci = self.calculate_confidence_intervals(tree, model_name)
        
        # Step 2: Extract weakness nodes
        print("Step 2: Extracting weakness nodes...")
        weakness_nodes = self.extract_weakness_nodes(tree_with_ci)
        
        # Step 3: Analyze results
        print(f"Found {len(weakness_nodes)} weakness nodes")
        
        total_questions = sum(node['size'] for node in weakness_nodes)
        
        if weakness_nodes:
            avg_accuracy = np.mean([node['accuracy'] for node in weakness_nodes])
            avg_dove_score = np.mean([
                np.mean(node['dove_scores']) for node in weakness_nodes 
                if node['dove_scores']
            ])
        else:
            avg_accuracy = 0.0
            avg_dove_score = 0.0
        
        return {
            'model_name': model_name,
            'extraction_params': {
                'alpha': self.alpha,
                'threshold': self.threshold,
                'direction': self.direction,
                'min_size_parent': self.min_size_parent,
                'min_size_child': self.min_size_child
            },
            'weakness_nodes': weakness_nodes,
            'summary': {
                'num_weaknesses': len(weakness_nodes),
                'total_questions_affected': total_questions,
                'avg_accuracy_in_weaknesses': avg_accuracy,
                'avg_dove_score_in_weaknesses': avg_dove_score
            },
            'tree_with_analysis': tree_with_ci
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract weakness profiles from MMLU+DOVE tree")
    parser.add_argument("--input", required=True, help="Path to MMLU combined tree JSON")
    parser.add_argument("--output", required=True, help="Path to save weakness profile")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct", help="Model name to analyze")
    parser.add_argument("--alpha", type=float, default=0.05, help="Confidence level")
    parser.add_argument("--threshold", type=float, default=0.5, help="Accuracy threshold")
    parser.add_argument("--direction", default="lower", choices=["lower", "higher"], 
                       help="Direction for weakness/strength detection")
    parser.add_argument("--min-size-parent", type=int, default=20, help="Minimum parent node size")
    parser.add_argument("--min-size-child", type=int, default=5, help="Minimum child node size")
    
    args = parser.parse_args()
    
    # Load the tree
    print(f"Loading tree from {args.input}...")
    with open(args.input, 'r') as f:
        tree = json.load(f)
    
    # Create extractor and run analysis
    extractor = WeaknessProfileExtractor(
        alpha=args.alpha,
        threshold=args.threshold,
        min_size_parent=args.min_size_parent,
        min_size_child=args.min_size_child,
        direction=args.direction
    )
    
    profile = extractor.extract_profile(tree, args.model)
    
    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=2)
    
    # Print summary
    summary = profile['summary']
    print("\n=== WEAKNESS PROFILE SUMMARY ===")
    print(f"Model: {args.model}")
    print(f"Number of weakness nodes: {summary['num_weaknesses']}")
    print(f"Total questions affected: {summary['total_questions_affected']}")
    print(f"Average accuracy in weaknesses: {summary['avg_accuracy_in_weaknesses']:.3f}")
    print(f"Average DOVE score in weaknesses: {summary['avg_dove_score_in_weaknesses']:.3f}")
    
    if profile['weakness_nodes']:
        print(f"\nTop 5 weakness capabilities:")
        sorted_weaknesses = sorted(profile['weakness_nodes'], 
                                 key=lambda x: x['size'], reverse=True)
        for i, weakness in enumerate(sorted_weaknesses[:5]):
            print(f"{i+1}. {weakness['capability']} (size: {weakness['size']}, "
                  f"accuracy: {weakness['accuracy']:.3f})")


if __name__ == "__main__":
    main() 