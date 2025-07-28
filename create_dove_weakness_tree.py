#!/usr/bin/env python3
"""
DOVE-Based Weakness Tree Generator

Creates a hierarchical weakness profile tree that maintains the same structure as MMLU.json
but replaces rankings with DOVE-based weakness scores for each subtree/cluster.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import statistics
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Set style for high-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_data():
    """Load DOVE scores and question mappings"""
    print("Loading data files...")
    
    # Load DOVE scores (question_id -> score)
    with open('MMLU_DOVE.json', 'r') as f:
        dove_scores = json.load(f)
    
    # Load MMLU.json structure
    try:
        with open('MMLU.json', 'r') as f:
            content = f.read()
            eval_tree = json.loads(content)
    except json.JSONDecodeError:
        # Handle truncated file
        with open('MMLU.json', 'r') as f:
            content = f.read()
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
            eval_tree = json.loads(content[:end_pos])
    
    print(f"Loaded {len(dove_scores)} DOVE scores")
    print(f"Loaded EvalTree with size {eval_tree.get('size', 'unknown')}")
    
    return dove_scores, eval_tree

def collect_question_indices(node):
    """Recursively collect all question indices from a subtree"""
    indices = []
    
    if isinstance(node, dict):
        if 'subtrees' in node:
            subtrees = node['subtrees']
            
            if isinstance(subtrees, int):
                # Leaf node with question index
                indices.append(subtrees)
            elif isinstance(subtrees, list):
                # Recurse through list of subtrees
                for subtree in subtrees:
                    indices.extend(collect_question_indices(subtree))
            elif isinstance(subtrees, dict):
                # Single subtree dict
                indices.extend(collect_question_indices(subtrees))
    
    return indices

def calculate_subtree_weakness(node, dove_scores):
    """Calculate weakness statistics for a subtree"""
    question_indices = collect_question_indices(node)
    
    # Get DOVE scores for questions in this subtree
    subtree_scores = []
    for idx in question_indices:
        if str(idx) in dove_scores:
            subtree_scores.append(dove_scores[str(idx)])
    
    if not subtree_scores:
        return None
    
    mean_score = statistics.mean(subtree_scores)
    
    # Determine weakness level
    if mean_score < 0.3:
        weakness_level = "Critical"
    elif mean_score < 0.5:
        weakness_level = "High"  
    elif mean_score < 0.7:
        weakness_level = "Moderate"
    else:
        weakness_level = "Low"
    
    return {
        'mean_dove_score': round(mean_score, 4),
        'std_dove_score': round(statistics.stdev(subtree_scores) if len(subtree_scores) > 1 else 0, 4),
        'question_count': len(subtree_scores),
        'total_questions': len(question_indices),
        'coverage': round(len(subtree_scores) / len(question_indices) if question_indices else 0, 4),
        'weakness_level': weakness_level,
        'min_score': round(min(subtree_scores), 4),
        'max_score': round(max(subtree_scores), 4)
    }

def build_weakness_tree(node, dove_scores, depth=0):
    """Build weakness tree maintaining MMLU.json structure"""
    if not isinstance(node, dict):
        return node
    
    # Create new node with weakness statistics
    weakness_node = {
        'capability': node.get('capability', f'Level_{depth}'),
        'size': node.get('size', 0),
        'depth': depth,
        'weakness_stats': calculate_subtree_weakness(node, dove_scores)
    }
    
    # Process subtrees
    if 'subtrees' in node:
        subtrees = node['subtrees']
        
        if isinstance(subtrees, int):
            # Leaf node - keep the question index but add DOVE score if available
            weakness_node['subtrees'] = subtrees
            if str(subtrees) in dove_scores:
                weakness_node['dove_score'] = dove_scores[str(subtrees)]
        elif isinstance(subtrees, list):
            # Process list of subtrees
            weakness_node['subtrees'] = []
            for subtree in subtrees:
                processed_subtree = build_weakness_tree(subtree, dove_scores, depth + 1)
                weakness_node['subtrees'].append(processed_subtree)
        elif isinstance(subtrees, dict):
            # Single subtree
            weakness_node['subtrees'] = build_weakness_tree(subtrees, dove_scores, depth + 1)
    
    return weakness_node

def extract_readable_hierarchy(node, max_depth=4):
    """Extract readable hierarchy for visualization (limit depth for readability)"""
    hierarchy = []
    
    def traverse(current_node, path="", current_depth=0):
        if not isinstance(current_node, dict) or current_depth >= max_depth:
            return
        
        capability = current_node.get('capability', f'Node_{current_depth}')
        stats = current_node.get('weakness_stats')
        
        if stats and stats['question_count'] >= 3:  # Only include nodes with sufficient data
            node_info = {
                'path': path,
                'capability': capability,
                'depth': current_depth,
                'stats': stats,
                'full_path': f"{path} > {capability}" if path else capability
            }
            hierarchy.append(node_info)
        
        # Recurse into subtrees
        if 'subtrees' in current_node and isinstance(current_node['subtrees'], list):
            for i, subtree in enumerate(current_node['subtrees']):
                new_path = f"{path} > {capability}" if path else capability
                traverse(subtree, new_path, current_depth + 1)
    
    traverse(node)
    return hierarchy

def create_readable_tree_visualization(hierarchy, output_file='dove_weakness_tree_readable.png'):
    """Create a readable hierarchical tree visualization"""
    print("Creating readable tree visualization...")
    
    # Filter to most significant nodes (top weaknesses at each level)
    hierarchy_by_depth = defaultdict(list)
    for node in hierarchy:
        hierarchy_by_depth[node['depth']].append(node)
    
    # Select top weakest nodes at each depth for readability
    selected_nodes = []
    for depth, nodes in hierarchy_by_depth.items():
        # Sort by weakness (lowest DOVE score first) and take top N
        nodes_sorted = sorted(nodes, key=lambda x: x['stats']['mean_dove_score'])
        if depth == 0:
            selected_nodes.extend(nodes_sorted[:1])  # Root
        elif depth == 1:
            selected_nodes.extend(nodes_sorted[:6])  # Top 6 at level 1
        elif depth == 2:
            selected_nodes.extend(nodes_sorted[:12])  # Top 12 at level 2
        else:
            selected_nodes.extend(nodes_sorted[:15])  # Top 15 at deeper levels
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in selected_nodes:
        node_id = node['full_path']
        G.add_node(node_id, **node)
        
        # Add edge to parent if not root
        if node['depth'] > 0:
            path_parts = node['full_path'].split(' > ')
            if len(path_parts) > 1:
                parent_path = ' > '.join(path_parts[:-1])
                if parent_path in [n['full_path'] for n in selected_nodes]:
                    G.add_edge(parent_path, node_id)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Create hierarchical layout
    pos = create_hierarchical_layout(G, selected_nodes)
    
    # Color and size nodes based on weakness
    node_colors = []
    node_sizes = []
    
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        mean_score = node_data['stats']['mean_dove_score']
        question_count = node_data['stats']['question_count']
        
        # Color based on weakness level
        if mean_score < 0.3:
            color = '#d32f2f'  # Critical - Dark red
        elif mean_score < 0.5:
            color = '#f57c00'  # High - Orange
        elif mean_score < 0.7:
            color = '#fbc02d'  # Moderate - Yellow
        else:
            color = '#388e3c'  # Low - Green
        
        # Size based on question count (log scale for readability)
        size = min(4000, max(800, np.log(question_count + 1) * 500))
        
        node_colors.append(color)
        node_sizes.append(size)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, alpha=0.6, width=2, ax=ax)
    
    # Add labels with capability names and scores
    labels = {}
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        capability = node_data['capability']
        mean_score = node_data['stats']['mean_dove_score']
        question_count = node_data['stats']['question_count']
        
        # Truncate long capability names
        if len(capability) > 30:
            capability_short = capability[:27] + "..."
        else:
            capability_short = capability
        
        labels[node_id] = f"{capability_short}\n{mean_score:.3f} ({question_count}q)"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#d32f2f', label='Critical Weakness (< 0.30)'),
        mpatches.Patch(color='#f57c00', label='High Weakness (0.30-0.49)'),
        mpatches.Patch(color='#fbc02d', label='Moderate Weakness (0.50-0.69)'),
        mpatches.Patch(color='#388e3c', label='Low Weakness (≥ 0.70)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Set title
    ax.set_title('DOVE-Based Weakness Profile Tree\nHierarchical Clustering with Robustness Scores', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Readable tree visualization saved to {output_file}")
    
    return fig

def create_hierarchical_layout(G, selected_nodes):
    """Create a clean hierarchical layout"""
    pos = {}
    
    # Group nodes by depth
    depth_groups = defaultdict(list)
    for node in selected_nodes:
        if node['full_path'] in G.nodes():
            depth_groups[node['depth']].append(node['full_path'])
    
    # Position nodes by depth
    for depth, nodes in depth_groups.items():
        y = 4 - depth  # Higher depth = lower on graph
        x_positions = np.linspace(-6, 6, len(nodes))
        
        for i, node_id in enumerate(nodes):
            pos[node_id] = (x_positions[i], y)
    
    return pos

def create_weakness_summary(weakness_tree):
    """Create summary statistics from weakness tree"""
    print("Creating weakness summary...")
    
    def collect_stats(node, all_stats):
        if isinstance(node, dict) and 'weakness_stats' in node and node['weakness_stats']:
            stats = node['weakness_stats']
            if stats['question_count'] >= 3:  # Only include reliable statistics
                all_stats.append({
                    'capability': node.get('capability', 'Unknown'),
                    'depth': node.get('depth', 0),
                    **stats
                })
        
        if isinstance(node, dict) and 'subtrees' in node and isinstance(node['subtrees'], list):
            for subtree in node['subtrees']:
                collect_stats(subtree, all_stats)
    
    all_stats = []
    collect_stats(weakness_tree, all_stats)
    
    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top weakest capabilities
    sorted_capabilities = sorted(all_stats, key=lambda x: x['mean_dove_score'])[:15]
    
    cap_names = [item['capability'][:25] + "..." if len(item['capability']) > 25 
                else item['capability'] for item in sorted_capabilities]
    cap_scores = [item['mean_dove_score'] for item in sorted_capabilities]
    colors = ['#d32f2f' if score < 0.3 else '#f57c00' if score < 0.5 else 
             '#fbc02d' if score < 0.7 else '#388e3c' for score in cap_scores]
    
    bars1 = ax1.barh(cap_names, cap_scores, color=colors, alpha=0.8)
    ax1.set_xlabel('DOVE Score (Robustness)')
    ax1.set_title('Top 15 Weakest Capability Clusters', fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Weakness Threshold')
    
    # Add value labels
    for bar, score in zip(bars1, cap_scores):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 2. Weakness distribution by depth
    depth_weakness = defaultdict(list)
    for stat in all_stats:
        depth_weakness[stat['depth']].append(stat['weakness_level'])
    
    depths = sorted(depth_weakness.keys())
    weakness_counts = {level: [] for level in ['Critical', 'High', 'Moderate', 'Low']}
    
    for depth in depths:
        level_counts = {level: depth_weakness[depth].count(level) for level in weakness_counts.keys()}
        for level in weakness_counts.keys():
            weakness_counts[level].append(level_counts[level])
    
    x = np.arange(len(depths))
    width = 0.2
    colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c']
    
    for i, (level, counts) in enumerate(weakness_counts.items()):
        ax2.bar(x + i * width, counts, width, label=level, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Tree Depth')
    ax2.set_ylabel('Number of Capabilities')
    ax2.set_title('Weakness Distribution by Tree Depth', fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f'Depth {d}' for d in depths])
    ax2.legend()
    
    # 3. Score distribution
    all_scores = [stat['mean_dove_score'] for stat in all_stats]
    ax3.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=np.mean(all_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_scores):.3f}')
    ax3.set_xlabel('DOVE Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Capability Weakness Scores', fontweight='bold')
    ax3.legend()
    
    # 4. Question count vs weakness
    question_counts = [stat['question_count'] for stat in sorted_capabilities]
    ax4.barh(cap_names, question_counts, color='lightcoral', alpha=0.8)
    ax4.set_xlabel('Number of Questions')
    ax4.set_title('Question Count (Top 15 Weakest)', fontweight='bold')
    
    # Add value labels
    for i, count in enumerate(question_counts):
        ax4.text(count + 0.5, i, str(count), va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dove_weakness_tree_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('dove_weakness_tree_summary.pdf', dpi=300, bbox_inches='tight')
    print("Weakness summary saved to dove_weakness_tree_summary.png/pdf")
    
    return fig, all_stats

def save_weakness_tree_json(weakness_tree, output_file='dove_weakness_tree.json'):
    """Save the weakness tree in JSON format similar to MMLU.json"""
    print(f"Saving weakness tree to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(weakness_tree, f, indent=2)
    
    print(f"Weakness tree saved to {output_file}")

def print_summary(weakness_tree, all_stats):
    """Print human-readable summary"""
    print("\n" + "="*80)
    print("DOVE-BASED WEAKNESS TREE SUMMARY")
    print("="*80)
    
    print(f"Total Capability Clusters Analyzed: {len(all_stats)}")
    print(f"Mean DOVE Score Across Clusters: {np.mean([s['mean_dove_score'] for s in all_stats]):.4f}")
    
    # Weakness distribution
    weakness_counts = {'Critical': 0, 'High': 0, 'Moderate': 0, 'Low': 0}
    for stat in all_stats:
        weakness_counts[stat['weakness_level']] += 1
    
    print(f"\nWeakness Distribution:")
    print(f"  Critical (< 0.30): {weakness_counts['Critical']} clusters")
    print(f"  High (0.30-0.49): {weakness_counts['High']} clusters")
    print(f"  Moderate (0.50-0.69): {weakness_counts['Moderate']} clusters")
    print(f"  Low (≥ 0.70): {weakness_counts['Low']} clusters")
    
    # Top weaknesses
    sorted_stats = sorted(all_stats, key=lambda x: x['mean_dove_score'])
    
    print(f"\nTOP 10 WEAKEST CAPABILITY CLUSTERS:")
    print("-" * 80)
    
    for i, stat in enumerate(sorted_stats[:10], 1):
        capability = stat['capability'][:60] + "..." if len(stat['capability']) > 60 else stat['capability']
        print(f"{i:2d}. {capability}")
        print(f"    Score: {stat['mean_dove_score']:.3f} | Level: {stat['weakness_level']} | "
              f"Questions: {stat['question_count']} | Depth: {stat['depth']}")
        print()

def main():
    """Main execution function"""
    print("Starting DOVE-Based Weakness Tree Generation...")
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Build weakness tree maintaining MMLU.json structure
        print("Building weakness tree with DOVE scores...")
        weakness_tree = build_weakness_tree(eval_tree, dove_scores)
        
        # Extract readable hierarchy
        print("Extracting readable hierarchy...")
        hierarchy = extract_readable_hierarchy(weakness_tree)
        
        if not hierarchy:
            print("No readable hierarchy could be extracted. Check data compatibility.")
            return
        
        # Create visualizations
        create_readable_tree_visualization(hierarchy)
        fig, all_stats = create_weakness_summary(weakness_tree)
        
        # Save weakness tree JSON
        save_weakness_tree_json(weakness_tree)
        
        # Print summary
        print_summary(weakness_tree, all_stats)
        
        print("\n" + "="*80)
        print("DOVE WEAKNESS TREE GENERATION COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  • dove_weakness_tree_readable.png/pdf - Readable tree visualization")
        print("  • dove_weakness_tree_summary.png/pdf - Summary statistics")
        print("  • dove_weakness_tree.json - Weakness tree in MMLU.json format")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 