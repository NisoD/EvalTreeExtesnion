#!/usr/bin/env python3
"""
Simple Accuracy vs DOVE Score Scatter Plot

Creates a focused visualization showing the relationship between 
accuracy (X-axis) and DOVE scores (Y-axis) for weakness analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import statistics

def create_accuracy_dove_scatter(profile_path, tree_path, model_name="Llama-3.1-8B-Instruct"):
    """Create the main scatter plot: Accuracy (X) vs DOVE Score (Y)."""
    
    # Load data
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    with open(tree_path, 'r') as f:
        tree = json.load(f)
    
    # Extract overall data
    def collect_all_data(node):
        data = []
        if isinstance(node.get('subtrees'), (int, type(None))) and 'dove_score' in node:
            accuracy = None
            if 'ranking' in node:
                for model_data in node['ranking']:
                    if model_data[0] == model_name:
                        accuracy = model_data[1]
                        break
            
            if accuracy is not None:
                data.append({
                    'accuracy': accuracy,
                    'dove_score': node['dove_score'],
                    'subject': node.get('subject', 'unknown'),
                    'is_weakness': False
                })
        elif isinstance(node.get('subtrees'), list):
            for child in node['subtrees']:
                data.extend(collect_all_data(child))
        return data
    
    all_data = collect_all_data(tree)
    
    # Extract weakness data
    weakness_questions = []
    for weakness in profile['weakness_nodes']:
        def extract_weakness_questions(node):
            questions = []
            if isinstance(node.get('subtrees'), (int, type(None))) and 'dove_score' in node:
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
                        'subject': node.get('subject', 'unknown'),
                        'is_weakness': True,
                        'weakness_name': weakness['capability']
                    })
            elif isinstance(node.get('subtrees'), list):
                for child in node['subtrees']:
                    questions.extend(extract_weakness_questions(child))
            return questions
        
        weakness_questions.extend(extract_weakness_questions(weakness['node_data']))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot all questions as background
    overall_acc = [d['accuracy'] for d in all_data]
    overall_dove = [d['dove_score'] for d in all_data]
    
    ax.scatter(overall_acc, overall_dove, 
               alpha=0.4, c='lightgray', s=30, label=f'All Questions (n={len(all_data)})')
    
    # Plot weakness questions with different colors for each weakness
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    weakness_names = list(set([q['weakness_name'] for q in weakness_questions]))
    
    for i, weakness_name in enumerate(weakness_names):
        weakness_subset = [q for q in weakness_questions if q['weakness_name'] == weakness_name]
        weak_acc = [q['accuracy'] for q in weakness_subset]
        weak_dove = [q['dove_score'] for q in weakness_subset]
        
        ax.scatter(weak_acc, weak_dove, 
                   c=colors[i % len(colors)], s=80, alpha=0.8,
                   label=f'Weakness {i+1} (n={len(weakness_subset)})', 
                   edgecolors='black', linewidth=0.5)
    
    # Add trend lines
    # Overall trend
    z_overall = np.polyfit(overall_acc, overall_dove, 1)
    p_overall = np.poly1d(z_overall)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p_overall(x_line), "k--", alpha=0.5, linewidth=2, 
            label=f'Overall Trend (r={np.corrcoef(overall_acc, overall_dove)[0,1]:.3f})')
    
    # Weakness trend
    if len(weakness_questions) > 1:
        weak_all_acc = [q['accuracy'] for q in weakness_questions]
        weak_all_dove = [q['dove_score'] for q in weakness_questions]
        z_weak = np.polyfit(weak_all_acc, weak_all_dove, 1)
        p_weak = np.poly1d(z_weak)
        ax.plot(x_line, p_weak(x_line), "r-", alpha=0.8, linewidth=2,
                label=f'Weakness Trend (r={np.corrcoef(weak_all_acc, weak_all_dove)[0,1]:.3f})')
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='DOVE = 0.5')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Accuracy = 0.5')
    
    # Customize plot
    ax.set_xlabel('Accuracy Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('DOVE Score (Robustness)', fontsize=14, fontweight='bold')
    ax.set_title(f'Accuracy vs DOVE Score Analysis\n{model_name}', 
                 fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add statistics text box
    overall_stats = f"""Overall Statistics:
    Mean Accuracy: {statistics.mean(overall_acc):.3f}
    Mean DOVE: {statistics.mean(overall_dove):.3f}
    Correlation: {np.corrcoef(overall_acc, overall_dove)[0,1]:.3f}
    
    Weakness Statistics:
    Mean Accuracy: {statistics.mean([q['accuracy'] for q in weakness_questions]):.3f}
    Mean DOVE: {statistics.mean([q['dove_score'] for q in weakness_questions]):.3f}
    Correlation: {np.corrcoef([q['accuracy'] for q in weakness_questions], [q['dove_score'] for q in weakness_questions])[0,1]:.3f}
    """
    
    ax.text(0.02, 0.98, overall_stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python simple_accuracy_dove_plot.py <profile_path> <tree_path> <output_path>")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    tree_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("Creating Accuracy vs DOVE Score scatter plot...")
    fig, ax = create_accuracy_dove_scatter(profile_path, tree_path)
    
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("Done!")
    
    # Show some key insights
    print("\n=== KEY INSIGHTS ===")
    print("1. Red/Blue points = Weakness areas (low accuracy + low DOVE)")
    print("2. Gray points = All other questions")
    print("3. Diagonal would indicate perfect accuracy-DOVE correlation")
    print("4. Lower-left quadrant = Both low accuracy and low robustness")
    print("5. Upper-right quadrant = Both high accuracy and high robustness")

if __name__ == "__main__":
    main() 