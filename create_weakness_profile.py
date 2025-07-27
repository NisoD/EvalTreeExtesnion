#!/usr/bin/env python3
"""
Weakness Profile Generator for MMLU DOVE Evaluation

This script creates a weakness profile by mapping your DOVE scores to the existing
MMLU EvalTree capability structure to identify specific areas of weakness.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

def load_data():
    """Load all necessary data files"""
    print("Loading data files...")
    
    # Load DOVE scores
    with open('MMLU_DOVE.json', 'r') as f:
        dove_scores = json.load(f)
    
    # Load EvalTree structure
    with open('MMLU.json', 'r') as f:
        eval_tree = json.load(f)
    
    # Load original dataset to get question mapping
    with open('Datasets/MMLU/dataset.json', 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dove_scores)} DOVE scores")
    print(f"Loaded dataset with {len(dataset)} questions")
    
    return dove_scores, eval_tree, dataset

def collect_leaf_indices(node, leaf_indices=None):
    """Recursively collect all leaf indices from the EvalTree"""
    if leaf_indices is None:
        leaf_indices = []
    
    if isinstance(node, dict):
        if 'subtrees' in node:
            if isinstance(node['subtrees'], list):
                for subtree in node['subtrees']:
                    collect_leaf_indices(subtree, leaf_indices)
            elif isinstance(node['subtrees'], dict):
                for subtree in node['subtrees'].values():
                    collect_leaf_indices(subtree, leaf_indices)
            else:
                # This is a leaf node with an index
                leaf_indices.append(node['subtrees'])
        elif isinstance(node, int):
            leaf_indices.append(node)
    elif isinstance(node, int):
        leaf_indices.append(node)
    elif isinstance(node, list):
        for item in node:
            collect_leaf_indices(item, leaf_indices)
    
    return leaf_indices

def calculate_node_statistics(node, dove_scores, dataset_size):
    """Calculate statistics for a node based on DOVE scores"""
    leaf_indices = collect_leaf_indices(node)
    
    # Filter indices that have DOVE scores
    scored_indices = [idx for idx in leaf_indices if str(idx) in dove_scores]
    
    if not scored_indices:
        return {
            'mean_score': None,
            'std_score': None,
            'min_score': None,
            'max_score': None,
            'count': 0,
            'coverage': 0.0,
            'weakness_level': None
        }
    
    scores = [dove_scores[str(idx)] for idx in scored_indices]
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
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
        'scored_indices': scored_indices[:10]  # Sample of indices for debugging
    }

def annotate_tree_with_stats(node, dove_scores, dataset_size):
    """Recursively annotate the tree with weakness statistics"""
    if isinstance(node, dict):
        # Calculate stats for this node
        stats = calculate_node_statistics(node, dove_scores, dataset_size)
        node['weakness_stats'] = stats
        
        # Recursively process subtrees
        if 'subtrees' in node:
            if isinstance(node['subtrees'], list):
                for subtree in node['subtrees']:
                    annotate_tree_with_stats(subtree, dove_scores, dataset_size)
            elif isinstance(node['subtrees'], dict):
                for subtree in node['subtrees'].values():
                    annotate_tree_with_stats(subtree, dove_scores, dataset_size)
    
    return node

def find_weakest_capabilities(node, weaknesses=None, path=""):
    """Find the weakest capability areas"""
    if weaknesses is None:
        weaknesses = []
    
    if isinstance(node, dict) and 'weakness_stats' in node:
        stats = node['weakness_stats']
        if stats['count'] > 0:  # Only consider nodes with actual data
            capability = node.get('capability', 'Unknown capability')
            weaknesses.append({
                'path': path,
                'capability': capability,
                'mean_score': stats['mean_score'],
                'weakness_level': stats['weakness_level'],
                'count': stats['count'],
                'coverage': stats['coverage'],
                'size': node.get('size', 0),
                'depth': node.get('depth', 0)
            })
        
        # Recursively process subtrees
        if 'subtrees' in node:
            if isinstance(node['subtrees'], list):
                for i, subtree in enumerate(node['subtrees']):
                    find_weakest_capabilities(subtree, weaknesses, f"{path}/{i}")
            elif isinstance(node['subtrees'], dict):
                for key, subtree in node['subtrees'].items():
                    find_weakest_capabilities(subtree, weaknesses, f"{path}/{key}")
    
    return weaknesses

def generate_weakness_report(dove_scores, eval_tree, dataset):
    """Generate comprehensive weakness report"""
    print("\nGenerating weakness profile...")
    
    # Annotate tree with statistics
    annotated_tree = annotate_tree_with_stats(eval_tree.copy(), dove_scores, len(dataset))
    
    # Find all capabilities with their weakness levels
    all_capabilities = find_weakest_capabilities(annotated_tree)
    
    # Filter out capabilities with insufficient data
    valid_capabilities = [cap for cap in all_capabilities if cap['count'] >= 5]
    
    # Sort by mean score (ascending = worst first)
    valid_capabilities.sort(key=lambda x: x['mean_score'])
    
    # Generate report
    report = {
        'summary': {
            'total_questions_scored': len(dove_scores),
            'total_capabilities_analyzed': len(valid_capabilities),
            'overall_mean_score': np.mean(list(dove_scores.values())),
            'critical_weaknesses': len([c for c in valid_capabilities if c['weakness_level'] == 'Critical']),
            'high_weaknesses': len([c for c in valid_capabilities if c['weakness_level'] == 'High']),
            'moderate_weaknesses': len([c for c in valid_capabilities if c['weakness_level'] == 'Moderate']),
            'low_weaknesses': len([c for c in valid_capabilities if c['weakness_level'] == 'Low'])
        },
        'top_weaknesses': valid_capabilities[:20],  # Top 20 weakest areas
        'weakness_by_level': {
            'Critical': [c for c in valid_capabilities if c['weakness_level'] == 'Critical'],
            'High': [c for c in valid_capabilities if c['weakness_level'] == 'High'],
            'Moderate': [c for c in valid_capabilities if c['weakness_level'] == 'Moderate'],
            'Low': [c for c in valid_capabilities if c['weakness_level'] == 'Low']
        },
        'annotated_tree': annotated_tree
    }
    
    return report

def print_weakness_summary(report):
    """Print a human-readable summary of weaknesses"""
    summary = report['summary']
    
    print("\n" + "="*60)
    print("MMLU WEAKNESS PROFILE SUMMARY")
    print("="*60)
    
    print(f"Total Questions Analyzed: {summary['total_questions_scored']}")
    print(f"Overall Mean Score: {summary['overall_mean_score']:.4f}")
    print(f"Total Capability Areas: {summary['total_capabilities_analyzed']}")
    
    print(f"\nWeakness Distribution:")
    print(f"  Critical (< 0.30): {summary['critical_weaknesses']} areas")
    print(f"  High (0.30-0.49): {summary['high_weaknesses']} areas")
    print(f"  Moderate (0.50-0.69): {summary['moderate_weaknesses']} areas")
    print(f"  Low (≥ 0.70): {summary['low_weaknesses']} areas")
    
    print(f"\nTOP 10 WEAKEST CAPABILITY AREAS:")
    print("-" * 60)
    
    for i, weakness in enumerate(report['top_weaknesses'][:10], 1):
        print(f"{i:2d}. Score: {weakness['mean_score']:.3f} | "
              f"Level: {weakness['weakness_level']:8s} | "
              f"Questions: {weakness['count']:3d}")
        print(f"    {weakness['capability'][:80]}...")
        print()

def main():
    """Main execution function"""
    try:
        # Load data
        dove_scores, eval_tree, dataset = load_data()
        
        # Generate weakness report
        report = generate_weakness_report(dove_scores, eval_tree, dataset)
        
        # Print summary
        print_weakness_summary(report)
        
        # Save detailed report
        with open('weakness_profile_detailed.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed weakness profile saved to: weakness_profile_detailed.json")
        
        # Save just the top weaknesses for easy viewing
        top_weaknesses = {
            'summary': report['summary'],
            'top_20_weaknesses': report['top_weaknesses'],
            'critical_areas': report['weakness_by_level']['Critical']
        }
        
        with open('weakness_profile_summary.json', 'w') as f:
            json.dump(top_weaknesses, f, indent=2)
        
        print(f"Summary weakness profile saved to: weakness_profile_summary.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()