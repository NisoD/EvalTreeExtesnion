#!/usr/bin/env python3
"""
Simple Weakness Extraction for DOVE-EvalTree Integration

This script extracts weakness profiles from our DOVE confidence intervals
using a simplified version of the EvalTree algorithm.
"""

import json
import copy

def load_confidence_intervals():
    """Load the confidence intervals we computed"""
    with open("results/dove_confidence_intervals.json", "r") as f:
        return json.load(f)

def extract_weaknesses_simple(node, alpha=0.05, threshold=0.5, path="root", min_size=20):
    """
    Simple weakness extraction algorithm
    
    Args:
        node: Tree node with confidence intervals
        alpha: Significance level
        threshold: Weakness threshold
        path: Current path in tree
        min_size: Minimum node size to consider
    
    Returns:
        List of weakness descriptions
    """
    weaknesses = []
    
    # Handle None nodes
    if node is None:
        return weaknesses
    
    # Handle leaf nodes (integers)
    if isinstance(node, int):
        return weaknesses  # Leaf nodes don't have weaknesses to extract
    
    # Handle non-dict nodes
    if not isinstance(node, dict):
        return weaknesses
    
    # Check if this node is a weakness
    if (node.get("confidence_interval") is not None and 
        node.get("size", 0) >= min_size):
        
        confidence_interval = node["confidence_interval"].get(str(alpha))
        if confidence_interval:
            upper_bound = confidence_interval[1]
            
            # Node is weak if upper bound is below threshold
            if upper_bound < threshold:
                # Check if we should go deeper (if children are not weak)
                should_extract_here = True
                
                if "subtrees" in node and isinstance(node["subtrees"], list):
                    for i, child in enumerate(node["subtrees"]):
                        if isinstance(child, dict) and (child.get("confidence_interval") is not None and 
                            child.get("size", 0) >= min_size):
                            child_interval = child["confidence_interval"].get(str(alpha))
                            if child_interval and child_interval[1] < threshold:
                                should_extract_here = False
                                break
                
                if should_extract_here:
                    weaknesses.append({
                        "path": path,
                        "size": node.get("size", 0),
                        "mean_score": node.get("sum_metrics", 0) / max(1, node.get("size", 1)),
                        "confidence_interval": confidence_interval,
                        "upper_bound": upper_bound
                    })
                    return weaknesses  # Don't recurse if we extracted this node
    
    # Recurse into children
    if "subtrees" in node and isinstance(node["subtrees"], list):
        for i, child in enumerate(node["subtrees"]):
            child_path = f"{path}_subtree_{i}"
            weaknesses.extend(extract_weaknesses_simple(child, alpha, threshold, child_path, min_size))
    elif "subtrees" in node and isinstance(node["subtrees"], dict):
        for cluster, child in node["subtrees"].items():
            child_path = f"{path}_{cluster}"
            weaknesses.extend(extract_weaknesses_simple(child, alpha, threshold, child_path, min_size))
    
    return weaknesses

def extract_at_multiple_thresholds():
    """Extract weaknesses at multiple thresholds"""
    
    print("🔍 Loading confidence intervals...")
    tree_data = load_confidence_intervals()
    
    print(f"✅ Loaded tree with {tree_data['size']} questions")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n🎯 Extracting weaknesses at threshold {threshold}...")
        
        weaknesses = extract_weaknesses_simple(tree_data, 
                                             alpha=0.05, 
                                             threshold=threshold, 
                                             min_size=20)
        
        print(f"  Found {len(weaknesses)} weaknesses")
        
        # Sort by severity (lowest upper bound first)
        weaknesses.sort(key=lambda x: x["upper_bound"])
        
        all_results[threshold] = weaknesses
        
        # Show top 5 weaknesses
        print(f"  Top 5 weaknesses:")
        for i, weakness in enumerate(weaknesses[:5], 1):
            print(f"    {i}. {weakness['path']} (size: {weakness['size']}, "
                  f"upper_bound: {weakness['upper_bound']:.3f})")
    
    return all_results

def create_weakness_summary(all_results):
    """Create a summary of weakness patterns"""
    
    print("\n" + "="*70)
    print("🎯 WEAKNESS PROFILE SUMMARY")
    print("="*70)
    
    for threshold, weaknesses in all_results.items():
        print(f"\n📊 Threshold {threshold}:")
        print(f"  Total weaknesses: {len(weaknesses)}")
        
        if weaknesses:
            # Statistics
            upper_bounds = [w["upper_bound"] for w in weaknesses]
            sizes = [w["size"] for w in weaknesses]
            
            print(f"  Severity range: {min(upper_bounds):.3f} - {max(upper_bounds):.3f}")
            print(f"  Size range: {min(sizes)} - {max(sizes)} questions")
            print(f"  Total questions affected: {sum(sizes)}")
            
            # Most severe weaknesses
            print(f"  Most severe weaknesses:")
            for i, weakness in enumerate(weaknesses[:3], 1):
                mean_score = weakness["mean_score"]
                print(f"    {i}. {weakness['path']}")
                print(f"       Size: {weakness['size']}, Mean: {mean_score:.3f}, "
                      f"Upper bound: {weakness['upper_bound']:.3f}")

def main():
    """Main execution"""
    print("🔍 Simple Weakness Extraction from DOVE-EvalTree Integration")
    print("="*70)
    
    try:
        # Extract weaknesses
        all_results = extract_at_multiple_thresholds()
        
        # Save results
        print("\n💾 Saving results...")
        with open("results/simple_weakness_profiles.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary
        create_weakness_summary(all_results)
        
        print("\n" + "="*70)
        print("✅ SIMPLE WEAKNESS EXTRACTION COMPLETE!")
        print("="*70)
        print("📁 Results saved to: results/simple_weakness_profiles.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 