#!/usr/bin/env python3
"""
DOVE vs Binary Evaluation Comparison for MMLU

This script compares your DOVE permutation-based evaluation against 
traditional binary (correct/incorrect) evaluation to show where they 
diverge and intersect.
"""

import json
import statistics
from collections import defaultdict

def load_data():
    """Load DOVE scores and MMLU dataset"""
    print("📂 Loading data for comparison...")
    
    # Load DOVE scores
    with open('MMLU_DOVE.json') as f:
        dove_scores = json.load(f)
    
    # Load MMLU dataset to get correct answers
    with open('Datasets/MMLU/dataset.json') as f:
        mmlu_data = json.load(f)
    
    print(f"✅ Loaded {len(dove_scores)} DOVE scores")
    print(f"✅ Loaded {len(mmlu_data)} MMLU questions")
    
    return dove_scores, mmlu_data

def extract_correct_answers(mmlu_data):
    """Extract correct answers from MMLU dataset"""
    correct_answers = {}
    
    for i, item in enumerate(mmlu_data):
        question = item.get('question', '')
        # Extract correct answer from multiple choice format
        # Typically questions end with options A, B, C, D
        if 'A.' in question and 'B.' in question:
            # This is a multiple choice question
            # For this comparison, we'll simulate binary evaluation
            # In reality, you'd need the actual model predictions
            correct_answers[i] = 'A'  # Placeholder - would need actual correct answers
    
    return correct_answers

def simulate_binary_evaluation(dove_scores, threshold=0.5):
    """
    Simulate binary evaluation from DOVE scores
    
    DOVE scores represent the probability/confidence of correctness
    We can convert them to binary by applying a threshold
    """
    binary_results = {}
    
    for idx_str, dove_score in dove_scores.items():
        # Convert DOVE score to binary prediction
        # Higher DOVE score = more likely correct
        binary_results[idx_str] = 1 if dove_score >= threshold else 0
    
    return binary_results

def analyze_capability_differences(dove_scores, binary_results, eval_tree):
    """Compare DOVE vs binary evaluation at capability level"""
    
    def collect_leaf_indices(node, indices=None):
        if indices is None:
            indices = []
        
        if isinstance(node, dict):
            if 'subtrees' in node:
                subtrees = node['subtrees']
                if isinstance(subtrees, list):
                    for subtree in subtrees:
                        collect_leaf_indices(subtree, indices)
                elif isinstance(subtrees, dict):
                    for subtree in subtrees.values():
                        collect_leaf_indices(subtree, indices)
                elif isinstance(subtrees, int):
                    indices.append(subtrees)
        elif isinstance(node, int):
            indices.append(node)
        elif isinstance(node, list):
            for item in node:
                collect_leaf_indices(item, indices)
        
        return indices
    
    def analyze_node(node, path=""):
        if not isinstance(node, dict) or 'capability' not in node:
            return []
        
        # Get all questions in this capability
        leaf_indices = collect_leaf_indices(node)
        
        # Filter to questions we have data for
        available_indices = [idx for idx in leaf_indices if str(idx) in dove_scores]
        
        if len(available_indices) < 3:  # Need minimum questions for reliable comparison
            return []
        
        # Calculate DOVE performance
        dove_values = [dove_scores[str(idx)] for idx in available_indices]
        dove_mean = statistics.mean(dove_values)
        dove_std = statistics.stdev(dove_values) if len(dove_values) > 1 else 0
        
        # Calculate Binary performance  
        binary_values = [binary_results[str(idx)] for idx in available_indices]
        binary_accuracy = statistics.mean(binary_values)
        
        # Calculate difference
        difference = dove_mean - binary_accuracy
        
        capability_analysis = {
            'path': path,
            'capability': node.get('capability', 'Unknown'),
            'question_count': len(available_indices),
            'dove_mean': round(dove_mean, 4),
            'dove_std': round(dove_std, 4),
            'binary_accuracy': round(binary_accuracy, 4),
            'difference': round(difference, 4),
            'coverage': len(available_indices) / len(leaf_indices) if leaf_indices else 0
        }
        
        results = [capability_analysis]
        
        # Recursively analyze subtrees
        if 'subtrees' in node:
            subtrees = node['subtrees']
            if isinstance(subtrees, list):
                for i, subtree in enumerate(subtrees):
                    results.extend(analyze_node(subtree, f"{path}/{i}"))
            elif isinstance(subtrees, dict):
                for key, subtree in subtrees.items():
                    results.extend(analyze_node(subtree, f"{path}/{key}"))
        
        return results
    
    return analyze_node(eval_tree)

def print_comparison_summary(dove_scores, binary_results, capability_analyses):
    """Print comprehensive comparison summary"""
    
    print("\n" + "="*80)
    print("📊 DOVE vs BINARY EVALUATION COMPARISON")
    print("="*80)
    
    # Overall comparison
    dove_values = list(dove_scores.values())
    binary_values = list(binary_results.values())
    
    dove_overall = statistics.mean(dove_values)
    binary_overall = statistics.mean(binary_values)
    
    print(f"🌍 OVERALL PERFORMANCE:")
    print(f"   DOVE Mean Score:     {dove_overall:.4f} ± {statistics.stdev(dove_values):.4f}")
    print(f"   Binary Accuracy:     {binary_overall:.4f}")
    print(f"   Overall Difference:  {dove_overall - binary_overall:.4f}")
    print()
    
    # Distribution analysis
    print(f"📈 SCORE DISTRIBUTIONS:")
    
    # DOVE distribution
    dove_ranges = {
        'Very Low (0.0-0.2)': len([x for x in dove_values if 0.0 <= x < 0.2]),
        'Low (0.2-0.4)': len([x for x in dove_values if 0.2 <= x < 0.4]),
        'Medium (0.4-0.6)': len([x for x in dove_values if 0.4 <= x < 0.6]),
        'High (0.6-0.8)': len([x for x in dove_values if 0.6 <= x < 0.8]),
        'Very High (0.8-1.0)': len([x for x in dove_values if 0.8 <= x <= 1.0])
    }
    
    print(f"   DOVE Distribution:")
    for range_name, count in dove_ranges.items():
        pct = count / len(dove_values) * 100
        print(f"     {range_name}: {count:4d} ({pct:5.1f}%)")
    
    # Binary distribution
    correct_count = sum(binary_values)
    incorrect_count = len(binary_values) - correct_count
    
    print(f"   Binary Distribution:")
    print(f"     Correct:   {correct_count:4d} ({correct_count/len(binary_values)*100:5.1f}%)")
    print(f"     Incorrect: {incorrect_count:4d} ({incorrect_count/len(binary_values)*100:5.1f}%)")
    print()
    
    # Capability-level differences
    valid_analyses = [a for a in capability_analyses if a['question_count'] >= 5]
    
    if valid_analyses:
        print(f"🎯 CAPABILITY-LEVEL ANALYSIS ({len(valid_analyses)} areas):")
        
        # Sort by absolute difference
        valid_analyses.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        print(f"   Largest Divergences (DOVE vs Binary):")
        print(f"   {'Rank':<4} {'DOVE':<6} {'Binary':<6} {'Diff':<7} {'Questions':<9} {'Capability'}")
        print(f"   {'-'*4} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*50}")
        
        for i, analysis in enumerate(valid_analyses[:15], 1):
            capability_short = analysis['capability'][:45] + "..." if len(analysis['capability']) > 45 else analysis['capability']
            print(f"   {i:<4} {analysis['dove_mean']:<6.3f} {analysis['binary_accuracy']:<6.3f} "
                  f"{analysis['difference']:+7.3f} {analysis['question_count']:<9} {capability_short}")
        
        print()
        
        # Convergence analysis
        small_diff = [a for a in valid_analyses if abs(a['difference']) < 0.05]
        large_diff = [a for a in valid_analyses if abs(a['difference']) >= 0.2]
        
        print(f"🔍 CONVERGENCE ANALYSIS:")
        print(f"   Similar results (|diff| < 0.05): {len(small_diff)} areas ({len(small_diff)/len(valid_analyses)*100:.1f}%)")
        print(f"   Large differences (|diff| ≥ 0.20): {len(large_diff)} areas ({len(large_diff)/len(valid_analyses)*100:.1f}%)")
        
        if small_diff:
            print(f"   📍 Areas where DOVE ≈ Binary (examples):")
            for analysis in small_diff[:5]:
                print(f"     • {analysis['capability'][:60]}... (diff: {analysis['difference']:+.3f})")
        
        if large_diff:
            print(f"   📍 Areas where DOVE ≠ Binary (examples):")
            for analysis in large_diff[:5]:
                print(f"     • {analysis['capability'][:60]}... (diff: {analysis['difference']:+.3f})")

def main():
    """Main comparison analysis"""
    print("🔍 DOVE vs Binary Evaluation Comparison")
    print("="*50)
    
    # Load data
    dove_scores, mmlu_data = load_data()
    
    # Load EvalTree
    with open('MMLU.json') as f:
        content = f.read()
        if len(content) > 2000000:  # Handle large file
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
        eval_tree = json.loads(content)
    
    print("✅ Loaded EvalTree structure")
    
    # Convert DOVE to binary using threshold
    print(f"\n🔄 Converting DOVE scores to binary (threshold=0.5)...")
    binary_results = simulate_binary_evaluation(dove_scores, threshold=0.5)
    
    # Analyze capability differences
    print(f"🔍 Analyzing capability-level differences...")
    capability_analyses = analyze_capability_differences(dove_scores, binary_results, eval_tree)
    
    # Print comparison
    print_comparison_summary(dove_scores, binary_results, capability_analyses)
    
    # Save detailed comparison
    comparison_report = {
        'overall_stats': {
            'dove_mean': statistics.mean(dove_scores.values()),
            'dove_std': statistics.stdev(dove_scores.values()),
            'binary_accuracy': statistics.mean(binary_results.values()),
            'total_questions': len(dove_scores)
        },
        'capability_analyses': capability_analyses
    }
    
    with open('dove_vs_binary_comparison.json', 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\n✅ Detailed comparison saved to: dove_vs_binary_comparison.json")
    
    print(f"\n🔑 KEY INSIGHTS:")
    print(f"• DOVE provides nuanced confidence scores (0.0-1.0)")
    print(f"• Binary gives simple correct/incorrect (0 or 1)")
    print(f"• DOVE can identify partial understanding and uncertainty")
    print(f"• Areas with large differences show where confidence matters")

if __name__ == "__main__":
    main()