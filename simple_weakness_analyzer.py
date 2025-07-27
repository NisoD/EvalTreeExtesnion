#!/usr/bin/env python3
"""
Simple Weakness Profile Analyzer for MMLU DOVE Evaluation
Uses only Python standard library - no external dependencies

ADDRESSES YOUR KEY CONCERNS:
1. ✅ Missing questions are EXCLUDED (not treated as score 0)
2. ✅ Prevents false weakness detection
3. ✅ Uses your MMLU.json tree structure
4. ✅ Handles partial coverage properly
"""

import json
import statistics
import os
from collections import defaultdict

def load_dove_scores(file_path="MMLU_DOVE.json"):
    """Load DOVE scores"""
    print(f"📂 Loading DOVE scores from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to proper format
        scores = {str(k): float(v) for k, v in data.items() if v is not None}
        
        indices = [int(k) for k in scores.keys()]
        score_values = list(scores.values())
        
        print(f"✅ Loaded {len(scores)} DOVE scores")
        print(f"📊 Score range: {min(score_values):.4f} to {max(score_values):.4f}")
        print(f"📊 Mean score: {statistics.mean(score_values):.4f}")
        print(f"📋 Index range: {min(indices)} to {max(indices)}")
        print(f"📋 Missing questions: {max(indices) + 1 - len(scores)} out of {max(indices) + 1}")
        print(f"📋 Coverage: {len(scores) / (max(indices) + 1) * 100:.1f}%")
        
        return scores
        
    except Exception as e:
        print(f"❌ Error loading DOVE scores: {e}")
        return {}

def load_eval_tree(file_path="MMLU.json"):
    """Load EvalTree structure with error handling"""
    print(f"📂 Loading EvalTree from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Handle potentially corrupted large files
        if len(content) > 2000000:  # > 2MB
            print("⚠️  Large file detected, extracting valid JSON...")
            # Find the end of the first complete JSON object
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
        
        tree = json.loads(content)
        print(f"✅ Loaded EvalTree structure")
        
        if isinstance(tree, dict):
            print(f"🌳 Tree keys: {list(tree.keys())}")
            if 'size' in tree:
                print(f"🌳 Tree size: {tree['size']}")
            if 'capability' in tree:
                print(f"🌳 Root capability: {tree['capability'][:80]}...")
        
        return tree
        
    except Exception as e:
        print(f"❌ Error loading EvalTree: {e}")
        return {}

def collect_leaf_indices(node, indices=None):
    """Recursively collect all leaf indices from tree"""
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
        elif 'description' in node and 'subtrees' in node:
            # Handle Stage 4 format
            subtrees = node['subtrees']
            if isinstance(subtrees, int):
                indices.append(subtrees)
            elif isinstance(subtrees, list):
                for subtree in subtrees:
                    collect_leaf_indices(subtree, indices)
    elif isinstance(node, int):
        indices.append(node)
    elif isinstance(node, list):
        for item in node:
            collect_leaf_indices(item, indices)
    
    return indices

def calculate_stats(node, dove_scores):
    """Calculate statistics for a capability node"""
    # Get all leaf indices for this capability
    leaf_indices = collect_leaf_indices(node)
    
    # CRITICAL: Only include indices that have DOVE scores
    # This prevents false weakness from missing questions
    scored_indices = [idx for idx in leaf_indices if str(idx) in dove_scores]
    missing_indices = [idx for idx in leaf_indices if str(idx) not in dove_scores]
    
    if not scored_indices:
        return {
            'mean_score': None,
            'count': 0,
            'total_questions': len(leaf_indices),
            'missing_count': len(missing_indices),
            'coverage': 0.0,
            'weakness_level': None,
            'reliable': False
        }
    
    # Calculate statistics from available scores
    scores = [dove_scores[str(idx)] for idx in scored_indices]
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    # Coverage: percentage of questions we have scores for
    coverage = len(scored_indices) / len(leaf_indices) if leaf_indices else 0
    
    # Reliability: need good coverage and minimum questions
    reliable = coverage >= 0.3 and len(scored_indices) >= 3
    
    # Weakness level
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
        'min_score': round(min(scores), 4),
        'max_score': round(max(scores), 4),
        'count': len(scored_indices),
        'total_questions': len(leaf_indices),
        'missing_count': len(missing_indices),
        'coverage': round(coverage, 4),
        'weakness_level': weakness_level,
        'reliable': reliable
    }

def analyze_tree(node, dove_scores, capabilities=None, path=""):
    """Recursively analyze the tree and collect capability statistics"""
    if capabilities is None:
        capabilities = []
    
    if isinstance(node, dict):
        # Calculate stats for this node
        stats = calculate_stats(node, dove_scores)
        
        # Only include reliable analyses
        if stats['reliable'] and stats['count'] > 0:
            capability_name = node.get('capability', node.get('description', 'Unknown capability'))
            
            capability_info = {
                'path': path,
                'capability': capability_name,
                'mean_score': stats['mean_score'],
                'weakness_level': stats['weakness_level'],
                'count': stats['count'],
                'total_questions': stats['total_questions'],
                'missing_count': stats['missing_count'],
                'coverage': stats['coverage'],
                'std_score': stats['std_score'],
                'min_score': stats['min_score'],
                'max_score': stats['max_score']
            }
            capabilities.append(capability_info)
        
        # Recursively analyze subtrees
        if 'subtrees' in node:
            subtrees = node['subtrees']
            if isinstance(subtrees, list):
                for i, subtree in enumerate(subtrees):
                    analyze_tree(subtree, dove_scores, capabilities, f"{path}/{i}")
            elif isinstance(subtrees, dict):
                for key, subtree in subtrees.items():
                    analyze_tree(subtree, dove_scores, capabilities, f"{path}/{key}")
    
    return capabilities

def generate_weakness_report(dove_scores, eval_tree):
    """Generate comprehensive weakness report"""
    print("\n🔍 Generating weakness profile...")
    
    # Analyze all capabilities
    all_capabilities = analyze_tree(eval_tree, dove_scores)
    
    # Sort by mean score (worst first)
    all_capabilities.sort(key=lambda x: x['mean_score'])
    
    # Calculate overall statistics
    all_scores = list(dove_scores.values())
    total_missing = sum(cap['missing_count'] for cap in all_capabilities)
    
    # Count by weakness level
    weakness_counts = {
        'Critical': len([c for c in all_capabilities if c['weakness_level'] == 'Critical']),
        'High': len([c for c in all_capabilities if c['weakness_level'] == 'High']),
        'Moderate': len([c for c in all_capabilities if c['weakness_level'] == 'Moderate']),
        'Low': len([c for c in all_capabilities if c['weakness_level'] == 'Low'])
    }
    
    report = {
        'summary': {
            'total_questions_scored': len(dove_scores),
            'total_capabilities_analyzed': len(all_capabilities),
            'overall_mean_score': round(statistics.mean(all_scores), 4),
            'overall_std_score': round(statistics.stdev(all_scores), 4),
            'overall_min_score': round(min(all_scores), 4),
            'overall_max_score': round(max(all_scores), 4),
            **weakness_counts
        },
        'top_weaknesses': all_capabilities[:20],
        'weakness_by_level': {
            'Critical': [c for c in all_capabilities if c['weakness_level'] == 'Critical'],
            'High': [c for c in all_capabilities if c['weakness_level'] == 'High'],
            'Moderate': [c for c in all_capabilities if c['weakness_level'] == 'Moderate'],
            'Low': [c for c in all_capabilities if c['weakness_level'] == 'Low']
        },
        'all_capabilities': all_capabilities
    }
    
    return report

def print_summary(report):
    """Print human-readable summary"""
    summary = report['summary']
    
    print("\n" + "="*80)
    print("🎯 MMLU WEAKNESS PROFILE SUMMARY")
    print("="*80)
    
    print(f"📊 DATA OVERVIEW:")
    print(f"  Questions with DOVE scores: {summary['total_questions_scored']:,}")
    print(f"  Capability areas analyzed:  {summary['total_capabilities_analyzed']}")
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"  Mean Score: {summary['overall_mean_score']:.4f} ± {summary['overall_std_score']:.4f}")
    print(f"  Score Range: {summary['overall_min_score']:.4f} - {summary['overall_max_score']:.4f}")
    
    print(f"\n⚠️  WEAKNESS DISTRIBUTION:")
    print(f"  Critical (< 30%):   {summary['Critical']:3d} areas")
    print(f"  High (30-49%):      {summary['High']:3d} areas")
    print(f"  Moderate (50-69%):  {summary['Moderate']:3d} areas")
    print(f"  Low (≥ 70%):        {summary['Low']:3d} areas")
    
    print(f"\n🔍 TOP 10 WEAKEST CAPABILITY AREAS:")
    print("-" * 80)
    
    for i, weakness in enumerate(report['top_weaknesses'][:10], 1):
        coverage_pct = weakness['coverage'] * 100
        print(f"{i:2d}. Score: {weakness['mean_score']:.3f} ± {weakness['std_score']:.3f} | "
              f"Level: {weakness['weakness_level']:8s} | "
              f"Q: {weakness['count']:3d}/{weakness['total_questions']:3d} "
              f"({coverage_pct:.0f}%)")
        print(f"    {weakness['capability'][:70]}...")
        print()

def main():
    """Main execution"""
    print("🔍 MMLU Simple Weakness Profile Analyzer")
    print("="*50)
    print("✅ Handles missing DOVE scores correctly")
    print("✅ Prevents false weakness detection")
    print("✅ No external dependencies")
    print()
    
    # Load data
    dove_scores = load_dove_scores("MMLU_DOVE.json")
    eval_tree = load_eval_tree("MMLU.json")
    
    if not dove_scores:
        print("❌ No DOVE scores loaded. Check your MMLU_DOVE.json file.")
        return
    
    if not eval_tree:
        print("❌ No EvalTree loaded. Check your MMLU.json file.")
        return
    
    # Generate report
    report = generate_weakness_report(dove_scores, eval_tree)
    
    # Print summary
    print_summary(report)
    
    # Save reports
    with open('weakness_profile_simple.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    summary_report = {
        'summary': report['summary'],
        'top_20_weaknesses': report['top_weaknesses'],
        'critical_areas': report['weakness_by_level']['Critical']
    }
    
    with open('weakness_profile_summary_simple.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\n✅ Analysis completed!")
    print(f"📄 Detailed report: weakness_profile_simple.json")
    print(f"📄 Summary report: weakness_profile_summary_simple.json")
    
    print(f"\n🔑 KEY POINTS:")
    print(f"• Missing questions are EXCLUDED from analysis (not treated as 0)")
    print(f"• Only capability areas with ≥30% coverage and ≥3 questions are included")
    print(f"• This prevents false weakness detection from missing data")

if __name__ == "__main__":
    main()