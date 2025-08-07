#!/usr/bin/env python3
"""
Analyze Weakness Profile Results

This script provides detailed analysis of the extracted weakness profile,
including comparisons with DOVE scores and subject-level breakdowns.
"""

import json
import sys
from collections import defaultdict, Counter
import statistics

def analyze_weakness_profile(profile_path: str, combined_tree_path: str):
    """Analyze the weakness profile results in detail."""
    
    # Load the weakness profile
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    # Load the combined tree for comparison
    with open(combined_tree_path, 'r') as f:
        combined_tree = json.load(f)
    
    print("=== WEAKNESS PROFILE ANALYSIS ===")
    print(f"Model: {profile['model_name']}")
    print(f"Extraction Parameters: {profile['extraction_params']}")
    print()
    
    # Summary statistics
    summary = profile['summary']
    print("=== SUMMARY STATISTICS ===")
    print(f"Number of weakness nodes: {summary['num_weaknesses']}")
    print(f"Total questions affected: {summary['total_questions_affected']}")
    print(f"Average accuracy in weaknesses: {summary['avg_accuracy_in_weaknesses']:.3f}")
    print(f"Average DOVE score in weaknesses: {summary['avg_dove_score_in_weaknesses']:.3f}")
    print()
    
    # Get overall statistics for comparison
    overall_stats = get_overall_statistics(combined_tree, profile['model_name'])
    print("=== COMPARISON WITH OVERALL PERFORMANCE ===")
    print(f"Overall accuracy: {overall_stats['overall_accuracy']:.3f}")
    print(f"Overall average DOVE score: {overall_stats['overall_dove_score']:.3f}")
    print(f"Weakness accuracy gap: {overall_stats['overall_accuracy'] - summary['avg_accuracy_in_weaknesses']:.3f}")
    print(f"Weakness DOVE score gap: {overall_stats['overall_dove_score'] - summary['avg_dove_score_in_weaknesses']:.3f}")
    print()
    
    # Detailed analysis of each weakness
    print("=== DETAILED WEAKNESS ANALYSIS ===")
    for i, weakness in enumerate(profile['weakness_nodes'], 1):
        print(f"\nWeakness {i}: {weakness['capability']}")
        print(f"  Size: {weakness['size']} questions")
        print(f"  Accuracy: {weakness['accuracy']:.3f}")
        print(f"  DOVE scores: min={min(weakness['dove_scores']):.3f}, "
              f"max={max(weakness['dove_scores']):.3f}, "
              f"avg={statistics.mean(weakness['dove_scores']):.3f}")
        print(f"  Subjects ({len(weakness['subjects'])}): {', '.join(weakness['subjects'])}")
        
        # Analyze DOVE score distribution
        dove_scores = weakness['dove_scores']
        low_dove = sum(1 for score in dove_scores if score < 0.3)
        high_dove = sum(1 for score in dove_scores if score > 0.7)
        print(f"  DOVE distribution: {low_dove} low (<0.3), "
              f"{len(dove_scores) - low_dove - high_dove} medium, "
              f"{high_dove} high (>0.7)")
    
    # Subject analysis
    print("\n=== SUBJECT ANALYSIS ===")
    subject_weakness_count = Counter()
    subject_question_count = defaultdict(int)
    
    for weakness in profile['weakness_nodes']:
        for subject in weakness['subjects']:
            subject_weakness_count[subject] += 1
            # Count questions per subject in this weakness
            subject_questions = count_questions_per_subject_in_node(weakness['node_data'])
            for subj, count in subject_questions.items():
                subject_question_count[subj] += count
    
    print("Subjects appearing in weaknesses:")
    for subject, weakness_count in subject_weakness_count.most_common():
        question_count = subject_question_count[subject]
        print(f"  {subject}: appears in {weakness_count} weakness(es), "
              f"{question_count} questions affected")
    
    # DOVE score correlation analysis
    print("\n=== DOVE SCORE CORRELATION ===")
    all_weakness_dove_scores = []
    all_weakness_accuracies = []
    
    for weakness in profile['weakness_nodes']:
        avg_dove = statistics.mean(weakness['dove_scores'])
        all_weakness_dove_scores.append(avg_dove)
        all_weakness_accuracies.append(weakness['accuracy'])
    
    if len(all_weakness_dove_scores) > 1:
        correlation = calculate_correlation(all_weakness_dove_scores, all_weakness_accuracies)
        print(f"Correlation between DOVE scores and accuracy in weaknesses: {correlation:.3f}")
    else:
        print("Not enough weakness nodes to calculate correlation")
    
    # Confidence interval analysis
    print("\n=== STATISTICAL SIGNIFICANCE ===")
    for i, weakness in enumerate(profile['weakness_nodes'], 1):
        ci = weakness['confidence_interval'][str(profile['extraction_params']['alpha'])]
        print(f"Weakness {i}: accuracy {weakness['accuracy']:.3f}, "
              f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  Significantly below threshold {profile['extraction_params']['threshold']}: "
              f"{ci[1] < profile['extraction_params']['threshold']}")

def get_overall_statistics(tree, model_name):
    """Calculate overall statistics for the tree."""
    def collect_leaf_data(node):
        leaves = []
        if isinstance(node.get('subtrees'), (int, type(None))) and 'dove_score' in node:
            # Leaf node
            accuracy = None
            if 'ranking' in node:
                for model_data in node['ranking']:
                    if model_data[0] == model_name:
                        accuracy = model_data[1]
                        break
            if accuracy is not None:
                leaves.append({
                    'accuracy': accuracy,
                    'dove_score': node['dove_score']
                })
        elif isinstance(node.get('subtrees'), list):
            for child in node['subtrees']:
                leaves.extend(collect_leaf_data(child))
        return leaves
    
    leaves = collect_leaf_data(tree)
    
    if leaves:
        accuracies = [leaf['accuracy'] for leaf in leaves]
        dove_scores = [leaf['dove_score'] for leaf in leaves]
        
        # Convert accuracies to binary (correct/incorrect) for fair comparison
        binary_accuracies = [1 if acc > 0.5 else 0 for acc in accuracies]
        
        return {
            'overall_accuracy': statistics.mean(binary_accuracies),
            'overall_dove_score': statistics.mean(dove_scores),
            'total_questions': len(leaves)
        }
    else:
        return {'overall_accuracy': 0.0, 'overall_dove_score': 0.0, 'total_questions': 0}

def count_questions_per_subject_in_node(node):
    """Count questions per subject in a given node."""
    subject_counts = defaultdict(int)
    
    def count_in_node(n):
        if isinstance(n.get('subtrees'), (int, type(None))) and 'subject' in n:
            subject_counts[n['subject']] += 1
        elif isinstance(n.get('subtrees'), list):
            for child in n['subtrees']:
                count_in_node(child)
    
    count_in_node(node)
    return dict(subject_counts)

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_weakness_profile.py <profile_path> <combined_tree_path>")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    combined_tree_path = sys.argv[2]
    
    analyze_weakness_profile(profile_path, combined_tree_path)

if __name__ == "__main__":
    main() 