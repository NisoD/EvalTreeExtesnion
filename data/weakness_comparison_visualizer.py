#!/usr/bin/env python3
"""
Weakness Profile Comparison Visualizer

Creates visualizations comparing accuracy vs DOVE scores in weakness areas,
helping to understand the relationship between performance and robustness.
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import statistics

def extract_weakness_data(profile_path, combined_tree_path, model_name="Llama-3.1-8B-Instruct"):
    """Extract data for visualization from weakness profile and tree."""
    
    # Load the weakness profile
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    # Load the combined tree
    with open(combined_tree_path, 'r') as f:
        tree = json.load(f)
    
    # Extract data for each weakness node
    weakness_data = []
    
    for weakness in profile['weakness_nodes']:
        # Get individual question data from this weakness
        question_data = extract_questions_from_node(weakness['node_data'], model_name)
        
        weakness_info = {
            'capability': weakness['capability'],
            'size': weakness['size'],
            'node_accuracy': weakness['accuracy'],
            'node_dove_avg': statistics.mean(weakness['dove_scores']),
            'questions': question_data,
            'subjects': weakness['subjects']
        }
        weakness_data.append(weakness_info)
    
    # Also get overall dataset statistics for comparison
    overall_data = extract_overall_data(tree, model_name)
    
    return weakness_data, overall_data

def extract_questions_from_node(node, model_name):
    """Extract individual question data (accuracy, DOVE score) from a node."""
    questions = []
    
    def extract_from_node(n):
        if isinstance(n.get('subtrees'), (int, type(None))) and 'dove_score' in n:
            # Leaf node with question
            accuracy = None
            if 'ranking' in n:
                for model_data in n['ranking']:
                    if model_data[0] == model_name:
                        accuracy = model_data[1]
                        break
            
            if accuracy is not None:
                questions.append({
                    'accuracy': accuracy,
                    'dove_score': n['dove_score'],
                    'subject': n.get('subject', 'unknown'),
                    'question_text': n.get('input', '')[:100] + '...'
                })
        elif isinstance(n.get('subtrees'), list):
            for child in n['subtrees']:
                extract_from_node(child)
    
    extract_from_node(node)
    return questions

def extract_overall_data(tree, model_name):
    """Extract overall dataset statistics for comparison."""
    def collect_all_questions(node):
        questions = []
        if isinstance(node.get('subtrees'), (int, type(None))) and 'dove_score' in node:
            # Leaf node
            accuracy = None
            if 'ranking' in node:
                for model_data in node['ranking']:
                    if model_data[0] == model_name:
                        accuracy = model_data[1]
                        break
            
            if accuracy is not None:
                questions.append({
                    'accuracy': accuracy,
                    'dove_score': node['dove_score'],
                    'subject': node.get('subject', 'unknown')
                })
        elif isinstance(node.get('subtrees'), list):
            for child in node['subtrees']:
                questions.extend(collect_all_questions(child))
        return questions
    
    return collect_all_questions(tree)

def create_weakness_comparison_plot(weakness_data, overall_data, output_path):
    """Create comprehensive comparison visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weakness Profile: Accuracy vs DOVE Score Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Individual questions in weakness areas vs overall
    ax1 = axes[0, 0]
    
    # Plot overall data as background
    overall_acc = [q['accuracy'] for q in overall_data]
    overall_dove = [q['dove_score'] for q in overall_data]
    ax1.scatter(overall_acc, overall_dove, alpha=0.3, c='lightgray', s=20, label='All Questions')
    
    # Plot weakness questions
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, weakness in enumerate(weakness_data):
        questions = weakness['questions']
        acc_scores = [q['accuracy'] for q in questions]
        dove_scores = [q['dove_score'] for q in questions]
        
        ax1.scatter(acc_scores, dove_scores, 
                   c=colors[i % len(colors)], s=50, alpha=0.8,
                   label=f"Weakness {i+1} ({weakness['size']} q)")
    
    ax1.set_xlabel('Accuracy Score')
    ax1.set_ylabel('DOVE Score (Robustness)')
    ax1.set_title('Individual Questions: Accuracy vs DOVE Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Plot 2: Node-level comparison
    ax2 = axes[0, 1]
    
    node_acc = [w['node_accuracy'] for w in weakness_data]
    node_dove = [w['node_dove_avg'] for w in weakness_data]
    sizes = [w['size'] for w in weakness_data]
    
    # Create bubble plot where size represents number of questions
    scatter = ax2.scatter(node_acc, node_dove, s=[s*10 for s in sizes], 
                         alpha=0.6, c=range(len(weakness_data)), cmap='viridis')
    
    # Add labels for each weakness
    for i, weakness in enumerate(weakness_data):
        ax2.annotate(f"W{i+1}", 
                    (weakness['node_accuracy'], weakness['node_dove_avg']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('Node Accuracy')
    ax2.set_ylabel('Average DOVE Score')
    ax2.set_title('Weakness Nodes: Accuracy vs Average DOVE Score\n(Bubble size = # questions)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    ax3 = axes[1, 0]
    
    # Create histograms
    bins = np.linspace(0, 1, 21)
    
    # Overall accuracy distribution
    ax3.hist(overall_acc, bins=bins, alpha=0.5, label='Overall Accuracy', density=True, color='lightblue')
    
    # Weakness accuracy distribution
    all_weakness_acc = []
    for weakness in weakness_data:
        all_weakness_acc.extend([q['accuracy'] for q in weakness['questions']])
    
    ax3.hist(all_weakness_acc, bins=bins, alpha=0.7, label='Weakness Accuracy', density=True, color='red')
    
    ax3.set_xlabel('Accuracy Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Accuracy Distribution: Overall vs Weaknesses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DOVE score distribution comparison
    ax4 = axes[1, 1]
    
    # Overall DOVE distribution
    ax4.hist(overall_dove, bins=bins, alpha=0.5, label='Overall DOVE', density=True, color='lightgreen')
    
    # Weakness DOVE distribution
    all_weakness_dove = []
    for weakness in weakness_data:
        all_weakness_dove.extend([q['dove_score'] for q in weakness['questions']])
    
    ax4.hist(all_weakness_dove, bins=bins, alpha=0.7, label='Weakness DOVE', density=True, color='orange')
    
    ax4.set_xlabel('DOVE Score')
    ax4.set_ylabel('Density')
    ax4.set_title('DOVE Score Distribution: Overall vs Weaknesses')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return fig

def create_detailed_weakness_analysis(weakness_data, output_path):
    """Create detailed analysis plot for each weakness."""
    
    n_weaknesses = len(weakness_data)
    fig, axes = plt.subplots(n_weaknesses, 2, figsize=(12, 4*n_weaknesses))
    
    if n_weaknesses == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Detailed Weakness Analysis', fontsize=16, fontweight='bold')
    
    for i, weakness in enumerate(weakness_data):
        questions = weakness['questions']
        acc_scores = [q['accuracy'] for q in questions]
        dove_scores = [q['dove_score'] for q in questions]
        subjects = [q['subject'] for q in questions]
        
        # Scatter plot for this weakness
        ax_scatter = axes[i, 0]
        
        # Color by subject
        unique_subjects = list(set(subjects))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_subjects)))
        
        for j, subject in enumerate(unique_subjects):
            subject_mask = [s == subject for s in subjects]
            subject_acc = [acc for acc, mask in zip(acc_scores, subject_mask) if mask]
            subject_dove = [dove for dove, mask in zip(dove_scores, subject_mask) if mask]
            
            ax_scatter.scatter(subject_acc, subject_dove, 
                             c=[colors[j]], label=subject, s=50, alpha=0.7)
        
        ax_scatter.set_xlabel('Accuracy')
        ax_scatter.set_ylabel('DOVE Score')
        ax_scatter.set_title(f'Weakness {i+1}: {weakness["capability"][:50]}...')
        ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_scatter.grid(True, alpha=0.3)
        
        # Correlation analysis
        ax_corr = axes[i, 1]
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(acc_scores, dove_scores, bins=10)
        ax_corr.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='Blues', aspect='auto')
        
        # Calculate and display correlation
        if len(acc_scores) > 1:
            correlation = np.corrcoef(acc_scores, dove_scores)[0, 1]
            ax_corr.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax_corr.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        ax_corr.set_xlabel('Accuracy')
        ax_corr.set_ylabel('DOVE Score')
        ax_corr.set_title(f'Density Plot - Weakness {i+1}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to {output_path}")
    
    return fig

def print_statistical_summary(weakness_data, overall_data):
    """Print statistical summary of the comparison."""
    
    print("\n=== STATISTICAL SUMMARY ===")
    
    # Overall statistics
    overall_acc = [q['accuracy'] for q in overall_data]
    overall_dove = [q['dove_score'] for q in overall_data]
    
    print(f"Overall Dataset:")
    print(f"  Accuracy: mean={statistics.mean(overall_acc):.3f}, std={statistics.stdev(overall_acc):.3f}")
    print(f"  DOVE: mean={statistics.mean(overall_dove):.3f}, std={statistics.stdev(overall_dove):.3f}")
    print(f"  Correlation: {np.corrcoef(overall_acc, overall_dove)[0,1]:.3f}")
    
    # Weakness statistics
    all_weakness_acc = []
    all_weakness_dove = []
    
    for i, weakness in enumerate(weakness_data):
        questions = weakness['questions']
        acc_scores = [q['accuracy'] for q in questions]
        dove_scores = [q['dove_score'] for q in questions]
        
        all_weakness_acc.extend(acc_scores)
        all_weakness_dove.extend(dove_scores)
        
        print(f"\nWeakness {i+1}: {weakness['capability'][:50]}...")
        print(f"  Size: {len(questions)} questions")
        print(f"  Accuracy: mean={statistics.mean(acc_scores):.3f}, std={statistics.stdev(acc_scores) if len(acc_scores) > 1 else 0:.3f}")
        print(f"  DOVE: mean={statistics.mean(dove_scores):.3f}, std={statistics.stdev(dove_scores) if len(dove_scores) > 1 else 0:.3f}")
        if len(acc_scores) > 1:
            print(f"  Correlation: {np.corrcoef(acc_scores, dove_scores)[0,1]:.3f}")
    
    print(f"\nAll Weaknesses Combined:")
    print(f"  Accuracy: mean={statistics.mean(all_weakness_acc):.3f}, std={statistics.stdev(all_weakness_acc):.3f}")
    print(f"  DOVE: mean={statistics.mean(all_weakness_dove):.3f}, std={statistics.stdev(all_weakness_dove):.3f}")
    print(f"  Correlation: {np.corrcoef(all_weakness_acc, all_weakness_dove)[0,1]:.3f}")
    
    # Comparison
    print(f"\n=== WEAKNESS vs OVERALL COMPARISON ===")
    print(f"Accuracy gap: {statistics.mean(overall_acc) - statistics.mean(all_weakness_acc):.3f}")
    print(f"DOVE gap: {statistics.mean(overall_dove) - statistics.mean(all_weakness_dove):.3f}")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize weakness profile comparison")
    parser.add_argument("--profile", required=True, help="Path to weakness profile JSON")
    parser.add_argument("--tree", required=True, help="Path to combined tree JSON")
    parser.add_argument("--output", default="weakness_comparison", help="Output file prefix")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct", help="Model name")
    
    args = parser.parse_args()
    
    print("Extracting weakness data...")
    weakness_data, overall_data = extract_weakness_data(args.profile, args.tree, args.model)
    
    print("Creating comparison visualization...")
    create_weakness_comparison_plot(weakness_data, overall_data, f"{args.output}_comparison.png")
    
    print("Creating detailed analysis...")
    create_detailed_weakness_analysis(weakness_data, f"{args.output}_detailed.png")
    
    print_statistical_summary(weakness_data, overall_data)
    
    print(f"\nVisualizations created:")
    print(f"  - {args.output}_comparison.png: Overall comparison")
    print(f"  - {args.output}_detailed.png: Detailed per-weakness analysis")

if __name__ == "__main__":
    main() 