#!/usr/bin/env python3
"""
Sensitivity Analysis: Threshold Robustness for Hierarchical Weakness Identification

This script analyzes how the number and identity of weak subjects changes
across different robustness thresholds to validate our methodology.
"""

import json
import os
import statistics
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_hierarchical_results():
    """Load the subject-level robustness statistics"""
    with open('figures/evaltree_results/subject_robustness_stats.json', 'r') as f:
        subject_stats = json.load(f)
    return subject_stats

def sensitivity_analysis(subject_stats, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analyze sensitivity to different robustness thresholds"""
    
    results = {}
    all_subjects = list(subject_stats.keys())
    
    for threshold in thresholds:
        weak_subjects = []
        for subject, stats in subject_stats.items():
            if stats['mean_robustness'] < threshold:
                weak_subjects.append({
                    'subject': subject,
                    'robustness': stats['mean_robustness'],
                    'question_count': stats['question_count']
                })
        
        # Sort by robustness (weakest first)
        weak_subjects.sort(key=lambda x: x['robustness'])
        
        results[threshold] = {
            'weak_count': len(weak_subjects),
            'weak_percentage': len(weak_subjects) / len(all_subjects) * 100,
            'weak_subjects': weak_subjects[:10]  # Top 10 weakest
        }
    
    return results

def analyze_threshold_stability(subject_stats, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analyze which subjects consistently appear as weak across thresholds"""
    
    weak_at_threshold = {}
    for threshold in thresholds:
        weak_subjects = set()
        for subject, stats in subject_stats.items():
            if stats['mean_robustness'] < threshold:
                weak_subjects.add(subject)
        weak_at_threshold[threshold] = weak_subjects
    
    # Find subjects weak at multiple thresholds
    consistency_analysis = {}
    for subject in subject_stats.keys():
        count = sum(1 for threshold in thresholds if subject in weak_at_threshold[threshold])
        consistency_analysis[subject] = {
            'weak_at_thresholds': count,
            'robustness': subject_stats[subject]['mean_robustness'],
            'consistency_percentage': count / len(thresholds) * 100
        }
    
    return consistency_analysis

def create_threshold_visualization(sensitivity_results):
    """Create visualization of threshold sensitivity"""
    thresholds = sorted(sensitivity_results.keys())
    weak_counts = [sensitivity_results[t]['weak_count'] for t in thresholds]
    weak_percentages = [sensitivity_results[t]['weak_percentage'] for t in thresholds]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute count
    ax1.plot(thresholds, weak_counts, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Robustness Threshold')
    ax1.set_ylabel('Number of Weak Subjects')
    ax1.set_title('Weak Subject Count by Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Percentage
    ax2.plot(thresholds, weak_percentages, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Robustness Threshold')
    ax2.set_ylabel('Weak Subjects (%)')
    ax2.set_title('Weak Subject Percentage by Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== Sensitivity Analysis: Robustness Threshold Impact ===\n")
    
    # Load hierarchical results
    print("Loading hierarchical robustness statistics...")
    subject_stats = load_hierarchical_results()
    print(f"✓ Loaded statistics for {len(subject_stats)} subjects\n")
    
    # Sensitivity analysis across thresholds
    print("Analyzing sensitivity to different robustness thresholds...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    sensitivity_results = sensitivity_analysis(subject_stats, thresholds)
    
    print("=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    for threshold in thresholds:
        result = sensitivity_results[threshold]
        print(f"\nThreshold {threshold}:")
        print(f"  Weak subjects: {result['weak_count']}/57 ({result['weak_percentage']:.1f}%)")
        print("  Weakest subjects:")
        for i, subject_info in enumerate(result['weak_subjects'][:5], 1):
            subject = subject_info['subject']
            robustness = subject_info['robustness']
            count = subject_info['question_count']
            print(f"    {i}. {subject:25} | {robustness:.3f} ({count} questions)")
    
    # Consistency analysis
    print(f"\n" + "=" * 80)
    print("THRESHOLD CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    consistency_analysis = analyze_threshold_stability(subject_stats, thresholds)
    
    # Sort by consistency
    sorted_consistency = sorted(consistency_analysis.items(), 
                              key=lambda x: x[1]['consistency_percentage'], 
                              reverse=True)
    
    print("\nMost consistently weak subjects across thresholds:")
    for subject, info in sorted_consistency[:10]:
        consistency = info['consistency_percentage']
        robustness = info['robustness']
        weak_count = info['weak_at_thresholds']
        print(f"  {subject:25} | {robustness:.3f} | Weak at {weak_count}/5 thresholds ({consistency:.0f}%)")
    
    # Key insights
    print(f"\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Mathematical subjects consistency
    math_subjects = ['college_mathematics', 'abstract_algebra', 'high_school_mathematics', 
                    'high_school_physics', 'college_physics']
    math_consistency = []
    
    for subject in math_subjects:
        if subject in consistency_analysis:
            consistency = consistency_analysis[subject]['consistency_percentage']
            math_consistency.append(consistency)
            print(f"Mathematical subject: {subject:25} | {consistency:.0f}% consistency")
    
    if math_consistency:
        avg_math_consistency = statistics.mean(math_consistency)
        print(f"\nAverage mathematical subject consistency: {avg_math_consistency:.1f}%")
    
    # Threshold recommendation
    print(f"\nThreshold Recommendation Analysis:")
    print(f"  0.3: {sensitivity_results[0.3]['weak_count']} subjects ({sensitivity_results[0.3]['weak_percentage']:.1f}%) - Very sensitive")
    print(f"  0.4: {sensitivity_results[0.4]['weak_count']} subjects ({sensitivity_results[0.4]['weak_percentage']:.1f}%) - Moderate")
    print(f"  0.5: {sensitivity_results[0.5]['weak_count']} subjects ({sensitivity_results[0.5]['weak_percentage']:.1f}%) - Balanced ✓")
    print(f"  0.6: {sensitivity_results[0.6]['weak_count']} subjects ({sensitivity_results[0.6]['weak_percentage']:.1f}%) - Conservative")
    print(f"  0.7: {sensitivity_results[0.7]['weak_count']} subjects ({sensitivity_results[0.7]['weak_percentage']:.1f}%) - Very conservative")
    
    # Create visualization
    print(f"\nCreating threshold sensitivity visualization...")
    create_threshold_visualization(sensitivity_results)
    
    # Save results
    os.makedirs('figures/evaltree_results', exist_ok=True)
    
    with open('figures/evaltree_results/sensitivity_analysis.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        json_ready_results = {}
        for threshold, result in sensitivity_results.items():
            json_ready_results[threshold] = {
                'weak_count': result['weak_count'],
                'weak_percentage': result['weak_percentage'],
                'weak_subjects': result['weak_subjects']
            }
        json.dump(json_ready_results, f, indent=2)
    
    with open('figures/evaltree_results/consistency_analysis.json', 'w') as f:
        json.dump(consistency_analysis, f, indent=2)
    
    print("✓ Saved sensitivity analysis results")
    print("✓ Saved consistency analysis results")
    print("✓ Created threshold sensitivity visualization")
    print(f"\nSensitivity analysis complete! 🎉")

if __name__ == "__main__":
    main() 