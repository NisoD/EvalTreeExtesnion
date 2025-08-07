#!/usr/bin/env python3
"""
Run Original EvalTree Weakness Profiling with DOVE Robustness Scores

This script attempts to use the original EvalTree weakness profiling pipeline
with our DOVE robustness scores instead of traditional accuracy.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

# Add EvalTree to Python path
sys.path.append('.')

def load_data():
    """Load MMLU tree structure and DOVE scores"""
    print("Loading MMLU tree structure and DOVE scores...")
    
    # Load MMLU tree structure
    with open("data/MMLU.json", "r") as f:
        tree_data = json.load(f)
    
    # Load DOVE scores
    with open("data/MMLU_DOVE.json", "r") as f:
        dove_scores = json.load(f)
    
    print(f"Loaded tree with {tree_data.get('size', 'unknown')} total questions")
    print(f"Loaded {len(dove_scores)} DOVE robustness scores")
    
    return tree_data, dove_scores


def collect_question_indices(node):
    """Recursively collect all question indices from a subtree"""
    indices = []
    
    if isinstance(node, dict):
        if "subtrees" in node:
            subtrees = node["subtrees"]
            
            if isinstance(subtrees, int):
                indices.append(subtrees)
            elif isinstance(subtrees, list):
                for subtree in subtrees:
                    indices.extend(collect_question_indices(subtree))
            elif isinstance(subtrees, dict):
                for subtree in subtrees.values():
                    indices.extend(collect_question_indices(subtree))
    
    return indices


def calculate_confidence_intervals(tree, dove_scores, alpha=0.05):
    """Calculate confidence intervals for each node using DOVE scores"""
    
    def calculate_node(node):
        if isinstance(node, int):
            # Leaf node - single question
            if str(node) in dove_scores:
                score = dove_scores[str(node)]
                # For single question, confidence interval is just the score
                return {
                    "size": 1,
                    "sum_metrics": score,
                    "confidence_interval": {str(alpha): [score, score]},
                    "subtrees": node
                }
            else:
                return {
                    "size": 0,
                    "sum_metrics": 0,
                    "confidence_interval": None,
                    "subtrees": node
                }
        
        # Internal node
        node_results = {
            "size": 0,
            "sum_metrics": 0,
            "confidence_interval": None,
            "subtrees": None
        }
        
        if isinstance(node["subtrees"], list):
            node_results["subtrees"] = []
            for subtree in node["subtrees"]:
                subtree_results = calculate_node(subtree)
                node_results["subtrees"].append(subtree_results)
                node_results["size"] += subtree_results["size"]
                node_results["sum_metrics"] += subtree_results["sum_metrics"]
                
        elif isinstance(node["subtrees"], dict):
            node_results["subtrees"] = {}
            for cluster, subtree in node["subtrees"].items():
                subtree_results = calculate_node(subtree)
                node_results["subtrees"][cluster] = subtree_results
                node_results["size"] += subtree_results["size"]
                node_results["sum_metrics"] += subtree_results["sum_metrics"]
        else:
            # Single subtree
            subtree_results = calculate_node(node["subtrees"])
            node_results["subtrees"] = subtree_results
            node_results["size"] = subtree_results["size"]
            node_results["sum_metrics"] = subtree_results["sum_metrics"]
        
        # Calculate confidence interval if we have data
        if node_results["size"] > 0:
            # Get all DOVE scores for this subtree
            question_indices = collect_question_indices(node)
            scores = []
            for idx in question_indices:
                if str(idx) in dove_scores:
                    scores.append(dove_scores[str(idx)])
            
            if len(scores) >= 3:  # Need minimum sample size
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Simple confidence interval using normal approximation
                try:
                    n = len(scores)
                    z = 1.96  # for 95% confidence
                    margin_error = z * std_score / np.sqrt(n)
                    
                    lower_bound = max(0.0, mean_score - margin_error)
                    upper_bound = min(1.0, mean_score + margin_error)
                    
                    node_results["confidence_interval"] = {
                        str(alpha): [float(lower_bound), float(upper_bound)]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not calculate confidence interval for node with {len(scores)} scores: {e}")
                    node_results["confidence_interval"] = None
        
        return node_results
    
    return calculate_node(tree)


def extract_weakness_profile(tree_results, alpha=0.05, threshold=0.5, direction="lower"):
    """Extract weakness profile using the original EvalTree algorithm"""
    
    # Import the original extraction function
    try:
        from EvalTree.WeaknessProfile.extract_subtrees import extract_subtrees
        
        # Run extraction
        extract_subtrees(tree_results, alpha, threshold, direction)
        
        # Collect extracted weaknesses
        weaknesses = []
        
        def collect_weaknesses(node, capability=""):
            if node.get("extracted", False):
                weaknesses.append(capability)
            
            if isinstance(node.get("subtrees"), list):
                for i, subtree in enumerate(node["subtrees"]):
                    collect_weaknesses(subtree, f"{capability}_subtree_{i}")
            elif isinstance(node.get("subtrees"), dict):
                for cluster, subtree in node["subtrees"].items():
                    collect_weaknesses(subtree, f"{capability}_{cluster}")
        
        collect_weaknesses(tree_results, "root")
        
        return weaknesses
        
    except ImportError as e:
        print(f"Could not import original EvalTree functions: {e}")
        return []


def main():
    """Main execution function"""
    print("🔍 Running Original EvalTree Weakness Profiling with DOVE Scores")
    print("="*70)
    
    try:
        # Load data
        tree_data, dove_scores = load_data()
        
        # Calculate confidence intervals
        print("\n📊 Calculating confidence intervals with DOVE robustness scores...")
        tree_results = calculate_confidence_intervals(tree_data, dove_scores)
        
        print(f"✅ Processed tree with {tree_results['size']} questions having DOVE scores")
        
        # Save intermediate results
        print("\n💾 Saving confidence interval results...")
        os.makedirs("results", exist_ok=True)
        with open("results/dove_confidence_intervals.json", "w") as f:
            json.dump(tree_results, f, indent=2)
        
        # Extract weakness profiles at different thresholds
        print("\n🎯 Extracting weakness profiles...")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        all_profiles = {}
        for threshold in thresholds:
            print(f"  Extracting at threshold {threshold}...")
            
            # Make a copy for extraction (since it modifies the tree)
            import copy
            tree_copy = copy.deepcopy(tree_results)
            
            weaknesses = extract_weakness_profile(tree_copy, 
                                                alpha=0.05, 
                                                threshold=threshold, 
                                                direction="lower")
            
            all_profiles[threshold] = weaknesses
            print(f"    Found {len(weaknesses)} weaknesses")
        
        # Save weakness profiles
        print("\n💾 Saving weakness profiles...")
        with open("results/dove_weakness_profiles.json", "w") as f:
            json.dump(all_profiles, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("🎯 ORIGINAL EVALTREE WEAKNESS PROFILING COMPLETE!")
        print("="*70)
        
        print(f"\n📊 Summary:")
        print(f"  Total questions in tree: {tree_data.get('size', 'unknown')}")
        print(f"  Questions with DOVE scores: {tree_results['size']}")
        print(f"  Coverage: {tree_results['size']/tree_data.get('size', 1)*100:.1f}%")
        
        print(f"\n🎯 Weakness Profiles by Threshold:")
        for threshold, weaknesses in all_profiles.items():
            print(f"  Threshold {threshold}: {len(weaknesses)} weaknesses")
        
        print(f"\n📁 Results saved to:")
        print(f"  • results/dove_confidence_intervals.json")
        print(f"  • results/dove_weakness_profiles.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 