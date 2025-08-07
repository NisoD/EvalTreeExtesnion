#!/usr/bin/env python3
"""
Simple Model Comparison: Direct DOVE Score Analysis

This script directly compares DOVE robustness scores between Llama 3.1 8B and OLMOE
without relying on tree structure question indices.
"""

import json
import os
import statistics
import numpy as np
from collections import defaultdict

def load_scores():
    """Load DOVE scores for both models"""
    
    # Load Llama DOVE scores
    with open("data/MMLU_DOVE.json", "r") as f:
        llama_scores = json.load(f)
    
    # Load OLMOE scores
    with open("data/olmoe_mmlu_final_scores.json", "r") as f:
        olmoe_scores = json.load(f)
    
    print(f"Loaded {len(llama_scores)} Llama 3.1 8B DOVE scores")
    print(f"Loaded {len(olmoe_scores)} OLMOE DOVE scores")
    
    return llama_scores, olmoe_scores

def find_common_questions(llama_scores, olmoe_scores):
    """Find questions that have scores for both models"""
    
    llama_questions = set(llama_scores.keys())
    olmoe_questions = set(olmoe_scores.keys())
    
    common_questions = llama_questions & olmoe_questions
    
    print(f"Found {len(common_questions)} questions with scores from both models")
    print(f"Coverage: {len(common_questions)/max(len(llama_questions), len(olmoe_questions)):.1%}")
    
    return common_questions

def compare_models_directly(llama_scores, olmoe_scores, common_questions):
    """Compare models on overlapping questions"""
    
    comparisons = []
    
    for question_id in common_questions:
        llama_score = llama_scores[question_id]
        olmoe_score = olmoe_scores[question_id]
        
        comparisons.append({
            "question_id": int(question_id),
            "llama_score": llama_score,
            "olmoe_score": olmoe_score,
            "difference": olmoe_score - llama_score,
            "abs_difference": abs(olmoe_score - llama_score)
        })
    
    # Sort by question ID for consistency
    comparisons.sort(key=lambda x: x["question_id"])
    
    return comparisons

def analyze_score_distributions(llama_scores, olmoe_scores):
    """Analyze and compare score distributions"""
    
    llama_vals = list(llama_scores.values())
    olmoe_vals = list(olmoe_scores.values())
    
    distributions = {
        "llama": {
            "count": len(llama_vals),
            "mean": statistics.mean(llama_vals),
            "median": statistics.median(llama_vals),
            "std": statistics.stdev(llama_vals),
            "min": min(llama_vals),
            "max": max(llama_vals),
            "q25": np.percentile(llama_vals, 25),
            "q75": np.percentile(llama_vals, 75)
        },
        "olmoe": {
            "count": len(olmoe_vals),
            "mean": statistics.mean(olmoe_vals),
            "median": statistics.median(olmoe_vals),
            "std": statistics.stdev(olmoe_vals),
            "min": min(olmoe_vals),
            "max": max(olmoe_vals),
            "q25": np.percentile(olmoe_vals, 25),
            "q75": np.percentile(olmoe_vals, 75)
        }
    }
    
    return distributions

def categorize_performance_gaps(comparisons):
    """Categorize the performance differences"""
    
    differences = [comp["difference"] for comp in comparisons]
    abs_differences = [comp["abs_difference"] for comp in comparisons]
    
    # Count different types of performance gaps
    olmoe_stronger = sum(1 for d in differences if d > 0.1)  # OLMOE significantly better
    llama_stronger = sum(1 for d in differences if d < -0.1)  # Llama significantly better
    similar = sum(1 for d in differences if abs(d) <= 0.1)  # Similar performance
    
    # Find extreme cases
    large_gaps = [comp for comp in comparisons if comp["abs_difference"] > 0.5]
    
    analysis = {
        "total_comparisons": len(comparisons),
        "olmoe_stronger": olmoe_stronger,
        "llama_stronger": llama_stronger,
        "similar_performance": similar,
        "mean_difference": statistics.mean(differences),
        "median_difference": statistics.median(differences),
        "std_difference": statistics.stdev(differences),
        "mean_abs_difference": statistics.mean(abs_differences),
        "large_gaps_count": len(large_gaps),
        "large_gaps": large_gaps[:10]  # Top 10 largest gaps
    }
    
    return analysis

def extract_weakness_patterns(comparisons, llama_threshold=0.4, olmoe_threshold=0.3):
    """Extract patterns in weaknesses between models"""
    
    # Questions where both models are weak (different thresholds due to different distributions)
    both_weak = [comp for comp in comparisons 
                 if comp["llama_score"] < llama_threshold and comp["olmoe_score"] < olmoe_threshold]
    
    # Questions where only one model is weak
    llama_only_weak = [comp for comp in comparisons 
                       if comp["llama_score"] < llama_threshold and comp["olmoe_score"] >= olmoe_threshold]
    
    olmoe_only_weak = [comp for comp in comparisons 
                       if comp["llama_score"] >= llama_threshold and comp["olmoe_score"] < olmoe_threshold]
    
    # Questions where both models are strong
    both_strong = [comp for comp in comparisons 
                   if comp["llama_score"] >= llama_threshold and comp["olmoe_score"] >= olmoe_threshold]
    
    weakness_analysis = {
        "thresholds": {
            "llama_threshold": llama_threshold,
            "olmoe_threshold": olmoe_threshold
        },
        "both_weak": len(both_weak),
        "llama_only_weak": len(llama_only_weak),
        "olmoe_only_weak": len(olmoe_only_weak),
        "both_strong": len(both_strong),
        "both_weak_questions": [comp["question_id"] for comp in both_weak[:20]],  # Sample
        "llama_only_weak_questions": [comp["question_id"] for comp in llama_only_weak[:20]],
        "olmoe_only_weak_questions": [comp["question_id"] for comp in olmoe_only_weak[:20]]
    }
    
    return weakness_analysis

def main():
    """Main execution function"""
    print("🔍 Simple Multi-Model Robustness Comparison")
    print("="*60)
    
    # Load scores
    llama_scores, olmoe_scores = load_scores()
    
    # Find common questions
    common_questions = find_common_questions(llama_scores, olmoe_scores)
    
    if len(common_questions) == 0:
        print("❌ No common questions found between models!")
        return
    
    # Analyze score distributions
    print("\n📊 Analyzing score distributions...")
    distributions = analyze_score_distributions(llama_scores, olmoe_scores)
    
    print(f"Llama 3.1 8B: μ={distributions['llama']['mean']:.3f} ± {distributions['llama']['std']:.3f}")
    print(f"OLMOE: μ={distributions['olmoe']['mean']:.3f} ± {distributions['olmoe']['std']:.3f}")
    
    # Compare models directly
    print("\n🔄 Comparing models on common questions...")
    comparisons = compare_models_directly(llama_scores, olmoe_scores, common_questions)
    
    # Analyze performance gaps
    print("\n📈 Analyzing performance gaps...")
    gap_analysis = categorize_performance_gaps(comparisons)
    
    print(f"Total comparisons: {gap_analysis['total_comparisons']}")
    print(f"OLMOE stronger: {gap_analysis['olmoe_stronger']} ({gap_analysis['olmoe_stronger']/gap_analysis['total_comparisons']:.1%})")
    print(f"Llama stronger: {gap_analysis['llama_stronger']} ({gap_analysis['llama_stronger']/gap_analysis['total_comparisons']:.1%})")
    print(f"Similar performance: {gap_analysis['similar_performance']} ({gap_analysis['similar_performance']/gap_analysis['total_comparisons']:.1%})")
    print(f"Mean difference (OLMOE - Llama): {gap_analysis['mean_difference']:.3f}")
    print(f"Mean absolute difference: {gap_analysis['mean_abs_difference']:.3f}")
    
    # Extract weakness patterns
    print("\n🎯 Analyzing weakness patterns...")
    weakness_analysis = extract_weakness_patterns(comparisons)
    
    total = weakness_analysis['both_weak'] + weakness_analysis['llama_only_weak'] + weakness_analysis['olmoe_only_weak'] + weakness_analysis['both_strong']
    
    print(f"Both models weak: {weakness_analysis['both_weak']} ({weakness_analysis['both_weak']/total:.1%})")
    print(f"Only Llama weak: {weakness_analysis['llama_only_weak']} ({weakness_analysis['llama_only_weak']/total:.1%})")
    print(f"Only OLMOE weak: {weakness_analysis['olmoe_only_weak']} ({weakness_analysis['olmoe_only_weak']/total:.1%})")
    print(f"Both models strong: {weakness_analysis['both_strong']} ({weakness_analysis['both_strong']/total:.1%})")
    
    # Save results
    os.makedirs("figures/evaltree_results", exist_ok=True)
    
    results = {
        "distributions": distributions,
        "gap_analysis": gap_analysis,
        "weakness_analysis": weakness_analysis,
        "sample_comparisons": comparisons[:100]  # Save first 100 for inspection
    }
    
    with open("figures/evaltree_results/simple_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ COMPARISON COMPLETE!")
    print(f"📊 Results saved to figures/evaltree_results/simple_model_comparison.json")
    
    # Print key findings for paper
    print("\n🔑 KEY FINDINGS FOR PAPER:")
    print(f"• Analyzed {len(common_questions)} questions with both model scores")
    print(f"• OLMOE shows {gap_analysis['mean_difference']:.3f} average robustness difference vs Llama")
    print(f"• Models show similar performance on {gap_analysis['similar_performance']/gap_analysis['total_comparisons']:.1%} of questions")
    print(f"• Cross-model weakness consistency: {weakness_analysis['both_weak']/total:.1%} of questions weak for both models")

if __name__ == "__main__":
    main() 