#!/usr/bin/env python3
"""
OLMOE Weakness Analysis: Multi-Model Robustness-Aware Evaluation

This script analyzes OLMOE robustness data and compares findings with Llama 3.1 8B
to demonstrate the generalizability of our DOVE-EvalTree integration approach.
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

def compare_model_performances(llama_results, olmoe_results):
    """Compare weakness profiles between models"""
    
    def extract_capability_data(results, model_name):
        capabilities = []
        
        def traverse(node, path="root"):
            if "mean_robustness" in node and node.get("sample_size", 0) >= 3:
                capabilities.append({
                    "capability": node.get("capability", "Unknown"),
                    "path": path,
                    "mean_robustness": node["mean_robustness"],
                    "ci_upper": node["ci_upper"],
                    "sample_size": node["sample_size"],
                    "model": model_name
                })
            
            if "subtrees" in node:
                if isinstance(node["subtrees"], list):
                    for i, subtree in enumerate(node["subtrees"]):
                        traverse(subtree, f"{path}/subtree_{i}")
                elif isinstance(node["subtrees"], dict):
                    for cluster, subtree in node["subtrees"].items():
                        traverse(subtree, f"{path}/{cluster}")
        
        traverse(results)
        return capabilities
    
    llama_caps = extract_capability_data(llama_results, "Llama-3.1-8B")
    olmoe_caps = extract_capability_data(olmoe_results, "OLMOE")
    
    # Match capabilities by path for direct comparison
    matched_capabilities = []
    llama_dict = {cap["path"]: cap for cap in llama_caps}
    olmoe_dict = {cap["path"]: cap for cap in olmoe_caps}
    
    common_paths = set(llama_dict.keys()) & set(olmoe_dict.keys())
    
    for path in common_paths:
        llama_cap = llama_dict[path]
        olmoe_cap = olmoe_dict[path]
        
        matched_capabilities.append({
            "capability": llama_cap["capability"],
            "path": path,
            "llama_robustness": llama_cap["mean_robustness"],
            "olmoe_robustness": olmoe_cap["mean_robustness"],
            "robustness_diff": olmoe_cap["mean_robustness"] - llama_cap["mean_robustness"],
            "llama_sample_size": llama_cap["sample_size"],
            "olmoe_sample_size": olmoe_cap["sample_size"]
        })
    
    return matched_capabilities

def analyze_weakness_consistency(llama_weaknesses, olmoe_weaknesses, threshold=0.5):
    """Analyze consistency of weakness identification across models"""
    
    # Group by path for comparison
    llama_weak_paths = {w["path"] for w in llama_weaknesses}
    olmoe_weak_paths = {w["path"] for w in olmoe_weaknesses}
    
    # Calculate consistency metrics
    common_weaknesses = llama_weak_paths & olmoe_weak_paths
    llama_only = llama_weak_paths - olmoe_weak_paths
    olmoe_only = olmoe_weak_paths - llama_weak_paths
    
    consistency_stats = {
        "total_llama_weaknesses": len(llama_weaknesses),
        "total_olmoe_weaknesses": len(olmoe_weaknesses),
        "common_weaknesses": len(common_weaknesses),
        "llama_only_weaknesses": len(llama_only),
        "olmoe_only_weaknesses": len(olmoe_only),
        "consistency_rate": len(common_weaknesses) / max(len(llama_weak_paths), len(olmoe_weak_paths)) if llama_weak_paths or olmoe_weak_paths else 0
    }
    
    return consistency_stats, common_weaknesses, llama_only, olmoe_only

def main():
    """Main execution function"""
    print("🔍 OLMOE Multi-Model Robustness Analysis")
    print("="*60)
    
    # Load data
    tree_data, llama_scores, olmoe_scores = load_data()
    
    # Calculate confidence intervals for both models
    print("\n📊 Calculating confidence intervals...")
    llama_results = calculate_confidence_intervals(tree_data, llama_scores, "Llama-3.1-8B")
    olmoe_results = calculate_confidence_intervals(tree_data, olmoe_scores, "OLMOE")
    
    print(f"✅ Processed Llama tree with {llama_results['size']} questions")
    print(f"✅ Processed OLMOE tree with {olmoe_results['size']} questions")
    
    # Save results
    os.makedirs("figures/evaltree_results", exist_ok=True)
    
    with open("figures/evaltree_results/llama_confidence_intervals.json", "w") as f:
        json.dump(llama_results, f, indent=2)
    
    with open("figures/evaltree_results/olmoe_confidence_intervals.json", "w") as f:
        json.dump(olmoe_results, f, indent=2)
    
    # Extract weakness profiles at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n🎯 Extracting weakness profiles...")
    llama_profiles = {}
    olmoe_profiles = {}
    
    for threshold in thresholds:
        print(f"  Threshold {threshold}:")
        
        llama_weaknesses = extract_weaknesses_simple(llama_results, threshold)
        olmoe_weaknesses = extract_weaknesses_simple(olmoe_results, threshold)
        
        llama_profiles[threshold] = llama_weaknesses
        olmoe_profiles[threshold] = olmoe_weaknesses
        
        print(f"    Llama: {len(llama_weaknesses)} weaknesses")
        print(f"    OLMOE: {len(olmoe_weaknesses)} weaknesses")
        
        # Analyze consistency at this threshold
        consistency_stats, common, llama_only, olmoe_only = analyze_weakness_consistency(
            llama_weaknesses, olmoe_weaknesses, threshold
        )
        
        print(f"    Consistency: {consistency_stats['consistency_rate']:.1%} ({len(common)} common)")
    
    # Save weakness profiles
    with open("figures/evaltree_results/llama_weakness_profiles.json", "w") as f:
        json.dump(llama_profiles, f, indent=2)
    
    with open("figures/evaltree_results/olmoe_weakness_profiles.json", "w") as f:
        json.dump(olmoe_profiles, f, indent=2)
    
    # Detailed analysis at threshold 0.5
    print("\n📈 Detailed Analysis (Threshold 0.5):")
    llama_weak_05 = llama_profiles[0.5]
    olmoe_weak_05 = olmoe_profiles[0.5]
    
    # Calculate total questions covered
    llama_questions = sum(w["questions_covered"] for w in llama_weak_05)
    olmoe_questions = sum(w["questions_covered"] for w in olmoe_weak_05)
    
    print(f"Llama 3.1 8B: {len(llama_weak_05)} weaknesses ({llama_questions} questions)")
    print(f"OLMOE: {len(olmoe_weak_05)} weaknesses ({olmoe_questions} questions)")
    
    # Compare model performances
    print("\n🔄 Comparing Model Performances...")
    matched_capabilities = compare_model_performances(llama_results, olmoe_results)
    
    if matched_capabilities:
        # Calculate performance statistics
        robustness_diffs = [cap["robustness_diff"] for cap in matched_capabilities]
        
        print(f"Compared {len(matched_capabilities)} capabilities:")
        print(f"  Mean robustness difference (OLMOE - Llama): {statistics.mean(robustness_diffs):.3f}")
        print(f"  Std robustness difference: {statistics.stdev(robustness_diffs):.3f}")
        
        # Count where each model is stronger
        olmoe_stronger = sum(1 for diff in robustness_diffs if diff > 0)
        llama_stronger = sum(1 for diff in robustness_diffs if diff < 0)
        
        print(f"  OLMOE stronger: {olmoe_stronger} ({olmoe_stronger/len(matched_capabilities):.1%})")
        print(f"  Llama stronger: {llama_stronger} ({llama_stronger/len(matched_capabilities):.1%})")
        
        # Save comparison data
        with open("figures/evaltree_results/model_comparison.json", "w") as f:
            json.dump({
                "matched_capabilities": matched_capabilities,
                "summary_stats": {
                    "total_comparisons": len(matched_capabilities),
                    "mean_difference": statistics.mean(robustness_diffs),
                    "std_difference": statistics.stdev(robustness_diffs),
                    "olmoe_stronger_count": olmoe_stronger,
                    "llama_stronger_count": llama_stronger
                }
            }, f, indent=2)
    
    # Final consistency analysis
    print("\n🎯 Cross-Model Weakness Consistency:")
    consistency_stats, common_weaknesses, llama_only, olmoe_only = analyze_weakness_consistency(
        llama_weak_05, olmoe_weak_05
    )
    
    for key, value in consistency_stats.items():
        if "rate" in key:
            print(f"  {key.replace('_', ' ').title()}: {value:.1%}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nCommon weaknesses identified by both models:")
    for path in sorted(common_weaknesses):
        # Find the capability name
        for w in llama_weak_05:
            if w["path"] == path:
                print(f"  - {w['capability']}")
                break
    
    print("\n" + "="*60)
    print("✅ MULTI-MODEL ANALYSIS COMPLETE!")
    print(f"📊 Results saved to figures/evaltree_results/")

if __name__ == "__main__":
    main() 