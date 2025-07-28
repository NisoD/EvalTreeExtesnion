#!/usr/bin/env python3
"""
DOVE-Based Hierarchical Weakness Profile Generator for MMLU

This script creates a hierarchical weakness profile by mapping DOVE scores to the real
MMLU EvalTree capability structure from MMLU.json to identify specific areas of weakness.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import statistics
import matplotlib.patches as mpatches

# Set style for high-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_dove_scores(dove_file='MMLU_DOVE.json'):
    """Load DOVE robustness scores"""
    print(f"Loading DOVE scores from {dove_file}...")
    with open(dove_file, 'r') as f:
        dove_scores = json.load(f)
    print(f"Loaded {len(dove_scores)} DOVE scores")
    return dove_scores

def load_eval_tree(tree_file='MMLU.json'):
    """Load the real EvalTree structure from MMLU.json"""
    print(f"Loading EvalTree structure from {tree_file}...")
    
    try:
        with open(tree_file, 'r') as f:
            content = f.read()
            # Try to load the complete JSON
            eval_tree = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        # Try to extract the first valid JSON object if file is truncated
        try:
            with open(tree_file, 'r') as f:
                content = f.read()
                # Find the first complete JSON object
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    eval_tree = json.loads(content[:end_pos])
                else:
                    raise ValueError("Could not find complete JSON object")
        except Exception as e2:
            print(f"Failed to parse truncated JSON: {e2}")
            raise e2
    
    print(f"Loaded EvalTree with size {eval_tree.get('size', 'unknown')}")
    return eval_tree

def collect_leaf_indices_with_paths(node, dove_scores, path="", depth=0):
    """
    Recursively collect all leaf indices from the EvalTree with their capability paths
    Returns: Dict[int, Dict] where key is question_index and value contains path info
    """
    leaf_data = {}
    
    if isinstance(node, dict):
        current_capability = node.get('capability', f'Level_{depth}')
        current_path = f"{path} > {current_capability}" if path else current_capability
        
        if 'subtrees' in node:
            subtrees = node['subtrees']
            
            # Check if this is a leaf node (subtrees is an integer)
            if isinstance(subtrees, int):
                # This is a leaf node with a question index
                if str(subtrees) in dove_scores:
                    leaf_data[subtrees] = {
                        'path': current_path,
                        'capability': current_capability,
                        'dove_score': dove_scores[str(subtrees)],
                        'depth': depth
                    }
            elif isinstance(subtrees, list):
                # Recursively process subtree list
                for subtree in subtrees:
                    child_data = collect_leaf_indices_with_paths(subtree, dove_scores, current_path, depth + 1)
                    leaf_data.update(child_data)
            elif isinstance(subtrees, dict):
                # Process as single subtree
                child_data = collect_leaf_indices_with_paths(subtrees, dove_scores, current_path, depth + 1)
                leaf_data.update(child_data)
    
    return leaf_data

def build_hierarchical_structure(leaf_data):
    """Build hierarchical structure from leaf data"""
    print("Building hierarchical structure from real EvalTree...")
    
    # Group by capability paths
    capability_groups = defaultdict(list)
    path_hierarchy = defaultdict(set)
    
    for question_idx, data in leaf_data.items():
        path = data['path']
        capability = data['capability']
        dove_score = data['dove_score']
        
        capability_groups[capability].append(dove_score)
        
        # Build path hierarchy
        path_parts = [p.strip() for p in path.split('>')]
        for i in range(len(path_parts)):
            parent_path = ' > '.join(path_parts[:i+1])
            if i > 0:
                parent = ' > '.join(path_parts[:i])
                path_hierarchy[parent].add(parent_path)
    
    # Calculate statistics for each capability
    capability_stats = {}
    for capability, scores in capability_groups.items():
        if scores:
            capability_stats[capability] = {
                'mean_score': statistics.mean(scores),
                'std_score': statistics.stdev(scores) if len(scores) > 1 else 0,
                'count': len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'weakness_level': get_weakness_level(statistics.mean(scores)),
                'scores': scores
            }
    
    print(f"Built hierarchy with {len(capability_stats)} capabilities")
    return capability_stats, path_hierarchy, capability_groups

def get_weakness_level(score):
    """Classify weakness level based on DOVE score"""
    if score < 0.3:
        return "Critical"
    elif score < 0.5:
        return "High"
    elif score < 0.7:
        return "Moderate"
    else:
        return "Low"

def create_hierarchical_tree_visualization(capability_stats, path_hierarchy, output_file='dove_hierarchical_tree.png'):
    """Create a hierarchical tree visualization using NetworkX"""
    print("Creating hierarchical tree visualization...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges based on path hierarchy
    all_paths = set()
    for parent, children in path_hierarchy.items():
        all_paths.add(parent)
        all_paths.update(children)
        
        for child in children:
            G.add_edge(parent, child)
    
    # Add root if not present
    roots = [node for node in G.nodes() if G.in_degree(node) == 0]
    if not roots and all_paths:
        # Find the shortest path as root
        root = min(all_paths, key=lambda x: len(x.split('>')))
        roots = [root]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    
    if not G.nodes():
        # Fallback: create a simple capability-based visualization
        print("No hierarchical structure found, creating capability-based visualization...")
        return create_capability_visualization(capability_stats, output_file)
    
    # Create hierarchical layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except:
        pos = nx.random_layout(G, seed=42)
    
    # Color nodes based on DOVE scores
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        # Find the capability name (last part of path)
        capability_name = node.split('>')[-1].strip()
        
        if capability_name in capability_stats:
            score = capability_stats[capability_name]['mean_score']
            
            # Color based on weakness level
            if score < 0.3:
                color = '#d32f2f'  # Critical - Dark red
            elif score < 0.5:
                color = '#f57c00'  # High - Orange
            elif score < 0.7:
                color = '#fbc02d'  # Moderate - Yellow
            else:
                color = '#388e3c'  # Low - Green
                
            # Size based on number of questions
            size = min(3000, max(500, capability_stats[capability_name]['count'] * 50))
        else:
            color = '#9e9e9e'  # Gray for intermediate nodes
            size = 1000
            
        node_colors.append(color)
        node_sizes.append(size)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, alpha=0.4, ax=ax)
    
    # Add labels for leaf nodes only (to avoid clutter)
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    leaf_labels = {}
    for node in leaf_nodes:
        capability_name = node.split('>')[-1].strip()
        if capability_name in capability_stats:
            score = capability_stats[capability_name]['mean_score']
            leaf_labels[node] = f"{capability_name[:20]}...\n{score:.3f}" if len(capability_name) > 20 else f"{capability_name}\n{score:.3f}"
    
    nx.draw_networkx_labels(G, pos, leaf_labels, font_size=8, font_weight='bold', ax=ax)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#d32f2f', label='Critical Weakness (< 0.30)'),
        mpatches.Patch(color='#f57c00', label='High Weakness (0.30-0.49)'),
        mpatches.Patch(color='#fbc02d', label='Moderate Weakness (0.50-0.69)'),
        mpatches.Patch(color='#388e3c', label='Low Weakness (≥ 0.70)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Set title
    ax.set_title('DOVE-Based Hierarchical Weakness Profile\nReal MMLU EvalTree Structure', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Hierarchical tree visualization saved to {output_file}")
    
    return fig

def create_capability_visualization(capability_stats, output_file='dove_capability_tree.png'):
    """Create a simpler capability-based visualization as fallback"""
    print("Creating capability-based visualization...")
    
    # Sort capabilities by score (weakest first)
    sorted_capabilities = sorted(capability_stats.items(), key=lambda x: x[1]['mean_score'])
    
    # Take top 20 weakest capabilities for visualization
    top_weak = sorted_capabilities[:20]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Create horizontal bar chart
    capabilities = [item[0][:40] + "..." if len(item[0]) > 40 else item[0] for item, _ in top_weak]
    scores = [stats['mean_score'] for _, stats in top_weak]
    colors = ['#d32f2f' if score < 0.3 else '#f57c00' if score < 0.5 else 
             '#fbc02d' if score < 0.7 else '#388e3c' for score in scores]
    
    bars = ax.barh(capabilities, scores, color=colors, alpha=0.8)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', va='center', fontweight='bold')
    
    ax.set_xlabel('DOVE Score (Robustness)', fontweight='bold')
    ax.set_title('Top 20 Weakest Capabilities\nDOVE-Based Assessment', fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Weakness Threshold')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#d32f2f', label='Critical (< 0.30)'),
        mpatches.Patch(color='#f57c00', label='High (0.30-0.49)'),
        mpatches.Patch(color='#fbc02d', label='Moderate (0.50-0.69)'),
        mpatches.Patch(color='#388e3c', label='Low (≥ 0.70)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Capability visualization saved to {output_file}")
    
    return fig

def create_summary_statistics(capability_stats, dove_scores):
    """Create summary statistics visualization"""
    print("Creating summary statistics...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top weakest capabilities
    sorted_capabilities = sorted(capability_stats.items(), key=lambda x: x[1]['mean_score'])
    top_15 = sorted_capabilities[:15]
    
    cap_names = [item[0][:25] + "..." if len(item[0]) > 25 else item[0] for item, _ in top_15]
    cap_scores = [stats['mean_score'] for _, stats in top_15]
    colors = ['#d32f2f' if score < 0.3 else '#f57c00' if score < 0.5 else 
             '#fbc02d' if score < 0.7 else '#388e3c' for score in cap_scores]
    
    bars1 = ax1.barh(cap_names, cap_scores, color=colors, alpha=0.8)
    ax1.set_xlabel('DOVE Score')
    ax1.set_title('Top 15 Weakest Capabilities', fontweight='bold')
    
    # Add value labels
    for bar, score in zip(bars1, cap_scores):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 2. Weakness level distribution
    weakness_counts = Counter()
    for stats in capability_stats.values():
        weakness_counts[stats['weakness_level']] += 1
    
    levels = ['Critical', 'High', 'Moderate', 'Low']
    counts = [weakness_counts[level] for level in levels]
    level_colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c']
    
    ax2.pie(counts, labels=levels, colors=level_colors, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Weakness Level Distribution', fontweight='bold')
    
    # 3. DOVE score distribution
    all_dove_scores = [float(score) for score in dove_scores.values()]
    ax3.hist(all_dove_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=np.mean(all_dove_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_dove_scores):.3f}')
    ax3.set_xlabel('DOVE Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Overall DOVE Score Distribution', fontweight='bold')
    ax3.legend()
    
    # 4. Question count by capability (top 15)
    question_counts = [stats['count'] for _, stats in top_15]
    bars4 = ax4.barh(cap_names, question_counts, color='lightcoral', alpha=0.8)
    ax4.set_xlabel('Number of Questions')
    ax4.set_title('Question Count (Top 15 Weakest)', fontweight='bold')
    
    # Add value labels
    for bar, count in zip(bars4, question_counts):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dove_hierarchical_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('dove_hierarchical_summary.pdf', dpi=300, bbox_inches='tight')
    print("Summary statistics saved to dove_hierarchical_summary.png/pdf")
    
    return fig

def generate_weakness_report(capability_stats, dove_scores, leaf_data):
    """Generate comprehensive weakness report"""
    print("Generating detailed weakness report...")
    
    # Overall statistics
    all_dove_scores = [float(score) for score in dove_scores.values()]
    
    report = {
        'overall_statistics': {
            'total_dove_questions': len(dove_scores),
            'total_mapped_questions': len(leaf_data),
            'mean_dove_score': np.mean(all_dove_scores),
            'std_dove_score': np.std(all_dove_scores),
            'total_capabilities': len(capability_stats)
        },
        'weakness_analysis': {},
        'top_weaknesses': [],
        'critical_areas': [],
        'recommendations': []
    }
    
    # Analyze each capability
    for capability, stats in capability_stats.items():
        report['weakness_analysis'][capability] = {
            'mean_score': stats['mean_score'],
            'std_score': stats['std_score'],
            'question_count': stats['count'],
            'weakness_level': stats['weakness_level'],
            'min_score': stats['min_score'],
            'max_score': stats['max_score']
        }
    
    # Top weaknesses
    sorted_capabilities = sorted(capability_stats.items(), key=lambda x: x[1]['mean_score'])
    report['top_weaknesses'] = [
        {
            'capability': capability,
            'score': stats['mean_score'],
            'level': stats['weakness_level'],
            'questions': stats['count']
        }
        for capability, stats in sorted_capabilities[:20]
    ]
    
    # Critical areas
    critical_capabilities = [(cap, stats) for cap, stats in capability_stats.items() 
                           if stats['weakness_level'] == 'Critical']
    report['critical_areas'] = [
        {
            'capability': capability,
            'score': stats['mean_score'],
            'questions': stats['count']
        }
        for capability, stats in critical_capabilities
    ]
    
    # Recommendations
    if critical_capabilities:
        worst_capability = min(critical_capabilities, key=lambda x: x[1]['mean_score'])
        report['recommendations'].append(
            f"Immediate attention needed for '{worst_capability[0]}' (score: {worst_capability[1]['mean_score']:.3f})"
        )
    
    weakest_capability = sorted_capabilities[0]
    report['recommendations'].extend([
        f"Focus on weakest area: '{weakest_capability[0]}' (score: {weakest_capability[1]['mean_score']:.3f})",
        f"Target {len([c for c in capability_stats.values() if c['weakness_level'] in ['Critical', 'High']])} high-priority capabilities",
        "Consider robustness training for input perturbation sensitivity",
        "Generate targeted questions for identified weak capabilities"
    ])
    
    # Save report
    with open('dove_hierarchical_weakness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Detailed report saved to dove_hierarchical_weakness_report.json")
    return report

def print_weakness_summary(capability_stats, dove_scores, leaf_data):
    """Print a human-readable summary of weaknesses"""
    print("\n" + "="*80)
    print("DOVE-BASED HIERARCHICAL WEAKNESS PROFILE SUMMARY")
    print("="*80)
    
    all_dove_scores = [float(score) for score in dove_scores.values()]
    
    print(f"Total DOVE Questions: {len(dove_scores)}")
    print(f"Mapped to Hierarchy: {len(leaf_data)}")
    print(f"Overall Mean DOVE Score: {np.mean(all_dove_scores):.4f}")
    print(f"Total Capabilities Found: {len(capability_stats)}")
    
    # Weakness distribution
    weakness_counts = Counter()
    for stats in capability_stats.values():
        weakness_counts[stats['weakness_level']] += 1
    
    print(f"\nWeakness Distribution:")
    print(f"  Critical (< 0.30): {weakness_counts['Critical']} capabilities")
    print(f"  High (0.30-0.49): {weakness_counts['High']} capabilities")
    print(f"  Moderate (0.50-0.69): {weakness_counts['Moderate']} capabilities")
    print(f"  Low (≥ 0.70): {weakness_counts['Low']} capabilities")
    
    # Top weaknesses
    sorted_capabilities = sorted(capability_stats.items(), key=lambda x: x[1]['mean_score'])
    
    print(f"\nTOP 15 WEAKEST CAPABILITIES:")
    print("-" * 80)
    
    for i, (capability, stats) in enumerate(sorted_capabilities[:15], 1):
        capability_short = capability[:50] + "..." if len(capability) > 50 else capability
        print(f"{i:2d}. Score: {stats['mean_score']:.3f} | "
              f"Level: {stats['weakness_level']:8s} | "
              f"Questions: {stats['count']:3d} | {capability_short}")

def main():
    """Main execution function"""
    print("Starting DOVE-Based Hierarchical Weakness Profiling with Real MMLU Structure...")
    
    try:
        # Load data
        dove_scores = load_dove_scores()
        eval_tree = load_eval_tree()
        
        # Extract leaf indices with their hierarchical paths
        leaf_data = collect_leaf_indices_with_paths(eval_tree, dove_scores)
        print(f"Mapped {len(leaf_data)} questions to hierarchical structure")
        
        if not leaf_data:
            print("No questions could be mapped to the hierarchy. Check data compatibility.")
            return
        
        # Build hierarchical structure
        capability_stats, path_hierarchy, capability_groups = build_hierarchical_structure(leaf_data)
        
        # Create visualizations
        create_hierarchical_tree_visualization(capability_stats, path_hierarchy)
        create_summary_statistics(capability_stats, dove_scores)
        
        # Generate report and print summary
        generate_weakness_report(capability_stats, dove_scores, leaf_data)
        print_weakness_summary(capability_stats, dove_scores, leaf_data)
        
        print("\n" + "="*80)
        print("DOVE HIERARCHICAL PROFILING COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  • dove_hierarchical_tree.png/pdf - Hierarchical tree visualization")
        print("  • dove_hierarchical_summary.png/pdf - Summary statistics")
        print("  • dove_hierarchical_weakness_report.json - Detailed report")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()