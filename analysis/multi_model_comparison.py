#!/usr/bin/env python3
"""
Multi-Model Robustness Comparison Analysis

Refined analysis comparing Llama 3.1 8B and OLMOE weakness profiles
with appropriate thresholds for each model's score distribution.
"""

import json
import os
import sys
import numpy as np
import statistics
from collections import defaultdict
import copy

def load_data():
    """Load MMLU tree structure and both model scores"""
    print("Loading MMLU tree structure and robustness scores...")
    
    # Load MMLU tree structure
    with open("data/MMLU.json", "r") as f:
        tree_data = json.load(f)
    
    # Load Llama DOVE scores (original)
    with open("data/MMLU_DOVE.json", "r") as f:
        llama_scores = json.load(f)
    
    # Load OLMOE scores
    with open("data/olmoe_mmlu_final_scores.json", "r") as f:
        olmoe_scores = json.load(f)
    
    print(f"Loaded tree with {tree_data.get('size', 'unknown')} total questions")
    print(f"Loaded {len(llama_scores)} Llama 3.1 8B DOVE scores")
    print(f"Loaded {len(olmoe_scores)} OLMOE DOVE scores")
    
    return tree_data, llama_scores, olmoe_scores

def collect_question_indices(node):
    """Recursively collect all question indices from a subtree"""
    indices = []
    
    if isinstance(node, dict):
        if "question_indices" in node:
            indices.extend(node["question_indices"])
        
        if "subtrees" in node:
            subtrees = node["subtrees"]
            if isinstance(subtrees, list):
                for subtree in subtrees:
                    indices.extend(collect_question_indices(subtree))
            elif isinstance(subtrees, dict):
                for subtree in subtrees.values():
                    indices.extend(collect_question_indices(subtree))
    
    return indices

def calculate_confidence_intervals(tree, scores, model_name, alpha=0.05):
    """Calculate confidence intervals for all nodes using robustness scores"""
    
    def calculate_node(node):
        node_results = copy.deepcopy(node)
        
        # Get question indices for this node
        question_indices = collect_question_indices(node)
        
        # Extract robustness scores for these questions
        node_scores = []
        for idx in question_indices:
            if str(idx) in scores:
                node_scores.append(scores[str(idx)])
        
        # Calculate statistics if we have enough data
        if len(node_scores) >= 3:
            mean_score = statistics.mean(node_scores)
            std_score = statistics.stdev(node_scores)
            n = len(node_scores)
            
            # 95% confidence interval
            margin = 1.96 * (std_score / np.sqrt(n))
            ci_lower = mean_score - margin
            ci_upper = mean_score + margin
            
            node_results.update({
                "mean_robustness": round(mean_score, 4),
                "std_robustness": round(std_score, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "sample_size": n,
                "model": model_name
            })
        
        # Process subtrees
        if "subtrees" in node:
            if isinstance(node["subtrees"], list):
                node_results["subtrees"] = [calculate_node(subtree) for subtree in node["subtrees"]]
            elif isinstance(node["subtrees"], dict):
                node_results["subtrees"] = {k: calculate_node(v) for k, v in node["subtrees"].items()}
        
        return node_results
    
    return calculate_node(tree)

def extract_weaknesses_simple(node, threshold=0.5, path="root", min_size=3):
    """Extract weakness nodes using upper confidence bound threshold"""
    weaknesses = []
    
    # Check if this node is weak
    if ("ci_upper" in node and 
        node["ci_upper"] < threshold and 
        node.get("sample_size", 0) >= min_size):
        
        weaknesses.append({
            "capability": node.get("capability", "Unknown"),
            "path": path,
            "mean_robustness": node["mean_robustness"],
            "ci_upper": node["ci_upper"],
            "sample_size": node["sample_size"],
            "questions_covered": len(collect_question_indices(node))
        })
        
        # Don't recurse into subtrees if parent is already extracted
        return weaknesses
    
    # Recurse into subtrees if not extracted
    if "subtrees" in node:
        if isinstance(node["subtrees"], list):
            for i, subtree in enumerate(node["subtrees"]):
                sub_weaknesses = extract_weaknesses_simple(
                    subtree, threshold, f"{path}/subtree_{i}", min_size
                )
                weaknesses.extend(sub_weaknesses)
        elif isinstance(node["subtrees"], dict):
            for cluster, subtree in node["subtrees"].items():
                sub_weaknesses = extract_weaknesses_simple(
                    subtree, threshold, f"{path}/{cluster}", min_size
                )
                weaknesses.extend(sub_weaknesses)
    
    return weaknesses

def analyze_score_distributions(llama_scores, olmoe_scores):
    """Analyze score distributions to determine appropriate thresholds"""
    
    llama_vals = list(llama_scores.values())
    olmoe_vals = list(olmoe_scores.values())
    
    # Calculate percentiles for threshold setting
    llama_percentiles = {
        10: np.percentile(llama_vals, 10),
        25: np.percentile(llama_vals, 25),
        50: np.percentile(llama_vals, 50),
        75: np.percentile(llama_vals, 75),
        90: np.percentile(llama_vals, 90)
    }
    
    olmoe_percentiles = {
        10: np.percentile(olmoe_vals, 10),
        25: np.percentile(olmoe_vals, 25),
        50: np.percentile(olmoe_vals, 50),
        75: np.percentile(olmoe_vals, 75),
        90: np.percentile(olmoe_vals, 90)
    }
    
    return {
        "llama": {
            "mean": statistics.mean(llama_vals),
            "median": statistics.median(llama_vals),
            "std": statistics.stdev(llama_vals),
            "percentiles": llama_percentiles
        },
        "olmoe": {
            "mean": statistics.mean(olmoe_vals),
            "median": statistics.median(olmoe_vals),
            "std": statistics.stdev(olmoe_vals),
            "percentiles": olmoe_percentiles
        }
    }

def compare_capabilities_directly(llama_results, olmoe_results):
    """Compare capabilities between models using matched capability paths"""
    
    def extract_all_capabilities(node, path="root"):
        capabilities = []
        
        if "mean_robustness" in node and node.get("sample_size", 0) >= 3:
            capabilities.append({
                "capability": node.get("capability", "Unknown"),
                "path": path,
                "mean_robustness": node["mean_robustness"],
                "ci_upper": node["ci_upper"],
                "ci_lower": node["ci_lower"],
                "sample_size": node["sample_size"]
            })
        
        if "subtrees" in node:
            if isinstance(node["subtrees"], list):
                for i, subtree in enumerate(node["subtrees"]):
                    capabilities.extend(extract_all_capabilities(subtree, f"{path}/subtree_{i}"))
            elif isinstance(node["subtrees"], dict):
                for cluster, subtree in node["subtrees"].items():
                    capabilities.extend(extract_all_capabilities(subtree, f"{path}/{cluster}"))
        
        return capabilities
    
    llama_caps = extract_all_capabilities(llama_results)
    olmoe_caps = extract_all_capabilities(olmoe_results)
    
    # Create lookup dictionaries
    llama_dict = {cap["path"]: cap for cap in llama_caps}
    olmoe_dict = {cap["path"]: cap for cap in olmoe_caps}
    
    # Find common capabilities
    common_paths = set(llama_dict.keys()) & set(olmoe_dict.keys())
    
    comparisons = []
    for path in common_paths:
        llama_cap = llama_dict[path]
        olmoe_cap = olmoe_dict[path]
        
        comparisons.append({
            "capability": llama_cap["capability"],
            "path": path,
            "llama_mean": llama_cap["mean_robustness"],
            "olmoe_mean": olmoe_cap["mean_robustness"],
            "difference": olmoe_cap["mean_robustness"] - llama_cap["mean_robustness"],
            "llama_ci_upper": llama_cap["ci_upper"],
            "olmoe_ci_upper": olmoe_cap["ci_upper"],
            "llama_sample_size": llama_cap["sample_size"],
            "olmoe_sample_size": olmoe_cap["sample_size"]
        })
    
    return comparisons

def generate_multi_model_summary(llama_profiles, olmoe_profiles, comparisons, score_stats):
    """Generate comprehensive summary of multi-model analysis"""
    
    # Use median-based thresholds for fair comparison
    llama_threshold = score_stats["llama"]["percentiles"][50]  # median
    olmoe_threshold = score_stats["olmoe"]["percentiles"][50]  # median
    
    # Extract weaknesses at median thresholds
    llama_weaknesses = []
    olmoe_weaknesses = []
    
    # Find the closest threshold to median in our profiles
    for threshold in sorted(llama_profiles.keys()):
        if threshold >= llama_threshold:
            llama_weaknesses = llama_profiles[threshold]
            break
    
    for threshold in sorted(olmoe_profiles.keys()):
        if threshold >= olmoe_threshold:
            olmoe_weaknesses = olmoe_profiles[threshold]
            break
    
    # Calculate differences
    differences = [comp["difference"] for comp in comparisons]
    
    summary = {
        "score_distributions": score_stats,
        "thresholds_used": {
            "llama_median_threshold": llama_threshold,
            "olmoe_median_threshold": olmoe_threshold
        },
        "total_comparisons": len(comparisons),
        "performance_comparison": {
            "mean_difference": statistics.mean(differences),
            "median_difference": statistics.median(differences),
            "std_difference": statistics.stdev(differences),
            "olmoe_stronger_count": sum(1 for d in differences if d > 0),
            "llama_stronger_count": sum(1 for d in differences if d < 0),
            "similar_count": sum(1 for d in differences if abs(d) < 0.05)
        },
        "weakness_profiles": {
            "llama_weaknesses_count": len(llama_weaknesses),
            "olmoe_weaknesses_count": len(olmoe_weaknesses),
            "llama_questions_covered": sum(w.get("questions_covered", 0) for w in llama_weaknesses),
            "olmoe_questions_covered": sum(w.get("questions_covered", 0) for w in olmoe_weaknesses)
        }
    }
    
    return summary

def main():
    """Main execution function"""
    print("🔍 Multi-Model Robustness Analysis: Llama vs OLMOE")
    print("="*70)
    
    # Load data
    tree_data, llama_scores, olmoe_scores = load_data()
    
    # Analyze score distributions
    print("\n📊 Analyzing score distributions...")
    score_stats = analyze_score_distributions(llama_scores, olmoe_scores)
    
    print(f"Llama 3.1 8B: μ={score_stats['llama']['mean']:.3f}, σ={score_stats['llama']['std']:.3f}")
    print(f"OLMOE: μ={score_stats['olmoe']['mean']:.3f}, σ={score_stats['olmoe']['std']:.3f}")
    
    # Calculate confidence intervals for both models
    print("\n📈 Calculating confidence intervals...")
    llama_results = calculate_confidence_intervals(tree_data, llama_scores, "Llama-3.1-8B")
    olmoe_results = calculate_confidence_intervals(tree_data, olmoe_scores, "OLMOE")
    
    # Compare capabilities directly
    print("\n🔄 Comparing matched capabilities...")
    comparisons = compare_capabilities_directly(llama_results, olmoe_results)
    print(f"Found {len(comparisons)} matched capabilities for comparison")
    
    # Extract weakness profiles using percentile-based thresholds
    llama_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]  # Higher thresholds for Llama
    olmoe_thresholds = [0.15, 0.25, 0.35, 0.45, 0.55]  # Lower thresholds for OLMOE
    
    print("\n🎯 Extracting weakness profiles...")
    
    llama_profiles = {}
    for threshold in llama_thresholds:
        weaknesses = extract_weaknesses_simple(llama_results, threshold)
        llama_profiles[threshold] = weaknesses
        print(f"  Llama @ {threshold}: {len(weaknesses)} weaknesses")
    
    olmoe_profiles = {}
    for threshold in olmoe_thresholds:
        weaknesses = extract_weaknesses_simple(olmoe_results, threshold)
        olmoe_profiles[threshold] = weaknesses
        print(f"  OLMOE @ {threshold}: {len(weaknesses)} weaknesses")
    
    # Generate comprehensive summary
    print("\n📋 Generating summary...")
    summary = generate_multi_model_summary(llama_profiles, olmoe_profiles, comparisons, score_stats)
    
    # Save all results
    os.makedirs("figures/evaltree_results", exist_ok=True)
    
    with open("figures/evaltree_results/multi_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open("figures/evaltree_results/capability_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)
    
    with open("figures/evaltree_results/llama_weakness_profiles.json", "w") as f:
        json.dump(llama_profiles, f, indent=2)
    
    with open("figures/evaltree_results/olmoe_weakness_profiles.json", "w") as f:
        json.dump(olmoe_profiles, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 MULTI-MODEL ANALYSIS SUMMARY")
    print("="*70)
    
    perf = summary["performance_comparison"]
    print(f"Total Capability Comparisons: {summary['total_comparisons']}")
    print(f"Mean Robustness Difference (OLMOE - Llama): {perf['mean_difference']:.3f}")
    print(f"OLMOE Stronger: {perf['olmoe_stronger_count']} ({perf['olmoe_stronger_count']/summary['total_comparisons']:.1%})")
    print(f"Llama Stronger: {perf['llama_stronger_count']} ({perf['llama_stronger_count']/summary['total_comparisons']:.1%})")
    print(f"Similar Performance: {perf['similar_count']} ({perf['similar_count']/summary['total_comparisons']:.1%})")
    
    wp = summary["weakness_profiles"]
    print(f"\nWeakness Profile Comparison:")
    print(f"Llama Weaknesses: {wp['llama_weaknesses_count']} ({wp['llama_questions_covered']} questions)")
    print(f"OLMOE Weaknesses: {wp['olmoe_weaknesses_count']} ({wp['olmoe_questions_covered']} questions)")
    
    print("\n✅ ANALYSIS COMPLETE! Results saved to figures/evaltree_results/")

if __name__ == "__main__":
    main() 