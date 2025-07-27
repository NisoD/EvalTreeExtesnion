#!/usr/bin/env python3
"""
Flexible Weakness Profile Analyzer for MMLU DOVE Evaluation

This script can adapt to different data formats and create weakness profiles
by mapping your evaluation scores to the existing MMLU EvalTree structure.
"""

import json
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional

class WeaknessAnalyzer:
    def __init__(self):
        self.dove_scores = {}
        self.eval_tree = {}
        self.dataset = []
        
    def load_dove_scores(self, file_path: str) -> Dict[str, float]:
        """Load DOVE scores from various formats"""
        print(f"Loading DOVE scores from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                # Convert all keys to strings and values to floats
                scores = {str(k): float(v) for k, v in data.items() if v is not None}
                print(f"Loaded {len(scores)} DOVE scores")
                return scores
            else:
                print(f"Unexpected data format in {file_path}")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {}
        except Exception as e:
            print(f"Error loading DOVE scores: {e}")
            return {}
    
    def load_eval_tree(self, file_path: str) -> Dict:
        """Load EvalTree structure"""
        print(f"Loading EvalTree from: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                tree = json.load(f)
            print(f"Loaded EvalTree structure")
            return tree
        except Exception as e:
            print(f"Error loading EvalTree: {e}")
            return {}
    
    def collect_leaf_indices(self, node, leaf_indices=None):
        """Recursively collect all leaf indices from the EvalTree"""
        if leaf_indices is None:
            leaf_indices = []
        
        if isinstance(node, dict):
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for subtree in subtrees:
                        self.collect_leaf_indices(subtree, leaf_indices)
                elif isinstance(subtrees, dict):
                    for subtree in subtrees.values():
                        self.collect_leaf_indices(subtree, leaf_indices)
                elif isinstance(subtrees, int):
                    leaf_indices.append(subtrees)
            elif isinstance(node, int):
                leaf_indices.append(node)
        elif isinstance(node, int):
            leaf_indices.append(node)
        elif isinstance(node, list):
            for item in node:
                self.collect_leaf_indices(item, leaf_indices)
        
        return leaf_indices
    
    def calculate_node_stats(self, node) -> Dict:
        """Calculate statistics for a node based on DOVE scores"""
        leaf_indices = self.collect_leaf_indices(node)
        
        # Filter indices that have DOVE scores
        scored_indices = [idx for idx in leaf_indices if str(idx) in self.dove_scores]
        
        if not scored_indices:
            return {
                'mean_score': None,
                'std_score': None,
                'min_score': None,
                'max_score': None,
                'count': 0,
                'coverage': 0.0,
                'weakness_level': None,
                'leaf_indices': leaf_indices,
                'scored_indices': []
            }
        
        scores = [self.dove_scores[str(idx)] for idx in scored_indices]
        
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Determine weakness level based on mean score
        if mean_score < 0.3:
            weakness_level = "Critical"
        elif mean_score < 0.5:
            weakness_level = "High"
        elif mean_score < 0.7:
            weakness_level = "Moderate"
        else:
            weakness_level = "Low"
        
        return {
            'mean_score': round(mean_score, 4),
            'std_score': round(std_score, 4),
            'min_score': round(min_score, 4),
            'max_score': round(max_score, 4),
            'count': len(scored_indices),
            'coverage': round(len(scored_indices) / len(leaf_indices), 4) if leaf_indices else 0,
            'weakness_level': weakness_level,
            'leaf_indices': leaf_indices,
            'scored_indices': scored_indices[:10]  # Sample for debugging
        }
    
    def annotate_tree(self, node):
        """Recursively annotate tree with weakness statistics"""
        if isinstance(node, dict):
            # Calculate stats for this node
            stats = self.calculate_node_stats(node)
            node['weakness_stats'] = stats
            
            # Process subtrees
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for subtree in subtrees:
                        self.annotate_tree(subtree)
                elif isinstance(subtrees, dict):
                    for subtree in subtrees.values():
                        self.annotate_tree(subtree)
        
        return node
    
    def extract_capabilities(self, node, capabilities=None, path=""):
        """Extract all capabilities with their statistics"""
        if capabilities is None:
            capabilities = []
        
        if isinstance(node, dict) and 'weakness_stats' in node:
            stats = node['weakness_stats']
            if stats['count'] > 0:  # Only include nodes with actual data
                capability_info = {
                    'path': path,
                    'capability': node.get('capability', 'Unknown capability'),
                    'mean_score': stats['mean_score'],
                    'weakness_level': stats['weakness_level'],
                    'count': stats['count'],
                    'coverage': stats['coverage'],
                    'size': node.get('size', 0),
                    'depth': node.get('depth', 0),
                    'std_score': stats['std_score'],
                    'min_score': stats['min_score'],
                    'max_score': stats['max_score']
                }
                capabilities.append(capability_info)
            
            # Process subtrees
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for i, subtree in enumerate(subtrees):
                        self.extract_capabilities(subtree, capabilities, f"{path}/{i}")
                elif isinstance(subtrees, dict):
                    for key, subtree in subtrees.items():
                        self.extract_capabilities(subtree, capabilities, f"{path}/{key}")
        
        return capabilities
    
    def generate_report(self, min_questions=3):
        """Generate comprehensive weakness report"""
        print("\nGenerating weakness profile...")
        
        # Annotate tree with statistics
        annotated_tree = self.annotate_tree(self.eval_tree.copy())
        
        # Extract all capabilities
        all_capabilities = self.extract_capabilities(annotated_tree)
        
        # Filter capabilities with sufficient data
        valid_capabilities = [cap for cap in all_capabilities if cap['count'] >= min_questions]
        
        # Sort by mean score (ascending = worst first)
        valid_capabilities.sort(key=lambda x: x['mean_score'] if x['mean_score'] is not None else 0)
        
        # Calculate overall statistics
        all_scores = list(self.dove_scores.values())
        overall_stats = {
            'total_questions_scored': len(self.dove_scores),
            'total_capabilities_analyzed': len(valid_capabilities),
            'overall_mean_score': np.mean(all_scores) if all_scores else 0,
            'overall_std_score': np.std(all_scores) if all_scores else 0,
            'overall_min_score': np.min(all_scores) if all_scores else 0,
            'overall_max_score': np.max(all_scores) if all_scores else 0
        }
        
        # Count by weakness level
        weakness_counts = {
            'Critical': len([c for c in valid_capabilities if c['weakness_level'] == 'Critical']),
            'High': len([c for c in valid_capabilities if c['weakness_level'] == 'High']),
            'Moderate': len([c for c in valid_capabilities if c['weakness_level'] == 'Moderate']),
            'Low': len([c for c in valid_capabilities if c['weakness_level'] == 'Low'])
        }
        
        report = {
            'summary': {**overall_stats, **weakness_counts},
            'top_weaknesses': valid_capabilities[:20],
            'weakness_by_level': {
                'Critical': [c for c in valid_capabilities if c['weakness_level'] == 'Critical'],
                'High': [c for c in valid_capabilities if c['weakness_level'] == 'High'],
                'Moderate': [c for c in valid_capabilities if c['weakness_level'] == 'Moderate'],
                'Low': [c for c in valid_capabilities if c['weakness_level'] == 'Low']
            },
            'all_capabilities': valid_capabilities,
            'annotated_tree': annotated_tree
        }
        
        return report
    
    def print_summary(self, report):
        """Print human-readable summary"""
        summary = report['summary']
        
        print("\n" + "="*70)
        print("MMLU WEAKNESS PROFILE SUMMARY")
        print("="*70)
        
        print(f"Total Questions Analyzed: {summary['total_questions_scored']}")
        print(f"Overall Mean Score: {summary['overall_mean_score']:.4f} ± {summary['overall_std_score']:.4f}")
        print(f"Score Range: {summary['overall_min_score']:.4f} - {summary['overall_max_score']:.4f}")
        print(f"Total Capability Areas: {summary['total_capabilities_analyzed']}")
        
        print(f"\nWeakness Distribution:")
        print(f"  Critical (< 0.30): {summary['Critical']:3d} areas")
        print(f"  High (0.30-0.49):  {summary['High']:3d} areas")
        print(f"  Moderate (0.50-0.69): {summary['Moderate']:3d} areas")
        print(f"  Low (≥ 0.70):      {summary['Low']:3d} areas")
        
        print(f"\nTOP 10 WEAKEST CAPABILITY AREAS:")
        print("-" * 70)
        
        for i, weakness in enumerate(report['top_weaknesses'][:10], 1):
            print(f"{i:2d}. Score: {weakness['mean_score']:.3f} ± {weakness['std_score']:.3f} | "
                  f"Level: {weakness['weakness_level']:8s} | "
                  f"Questions: {weakness['count']:3d}")
            print(f"    {weakness['capability'][:65]}...")
            print()
    
    def analyze(self, dove_file="MMLU_DOVE.json", tree_file="MMLU.json", min_questions=3):
        """Main analysis function"""
        try:
            # Load data
            self.dove_scores = self.load_dove_scores(dove_file)
            self.eval_tree = self.load_eval_tree(tree_file)
            
            if not self.dove_scores:
                print("No DOVE scores loaded. Please check your data file.")
                return None
            
            if not self.eval_tree:
                print("No EvalTree loaded. Please check your tree file.")
                return None
            
            # Generate report
            report = self.generate_report(min_questions)
            
            # Print summary
            self.print_summary(report)
            
            # Save reports
            with open('weakness_profile_detailed.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            summary_report = {
                'summary': report['summary'],
                'top_20_weaknesses': report['top_weaknesses'],
                'critical_areas': report['weakness_by_level']['Critical']
            }
            
            with open('weakness_profile_summary.json', 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            print(f"\nDetailed report saved to: weakness_profile_detailed.json")
            print(f"Summary report saved to: weakness_profile_summary.json")
            
            return report
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution"""
    analyzer = WeaknessAnalyzer()
    
    # You can customize these file paths
    dove_file = "MMLU_DOVE.json"  # Your DOVE scores file
    tree_file = "MMLU.json"       # Your EvalTree structure file
    
    print("MMLU Weakness Profile Analyzer")
    print("="*40)
    print(f"DOVE scores file: {dove_file}")
    print(f"EvalTree file: {tree_file}")
    
    # Run analysis
    report = analyzer.analyze(dove_file, tree_file, min_questions=3)
    
    if report:
        print("\nAnalysis completed successfully!")
        print("Check the generated JSON files for detailed results.")
    else:
        print("\nAnalysis failed. Please check your data files and try again.")

if __name__ == "__main__":
    main()