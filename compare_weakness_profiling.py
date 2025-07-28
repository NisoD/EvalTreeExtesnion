#!/usr/bin/env python3
"""
Robustness in Weakness Profiling: MMLU vs DOVE Comparison

Compares weak branches identified by:
1. Traditional MMLU accuracy-based profiling
2. DOVE robustness-based profiling

Shows how robustness analysis reveals different weakness patterns
for targeted question generation.
"""

import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

def load_data():
    """Load DOVE scores and MMLU tree structure"""
    print("Loading data files...")

    # Load DOVE scores
    with open("MMLU_DOVE.json", "r") as f:
        dove_scores = json.load(f)

    # Load MMLU.json structure (handle truncation)
    try:
        with open("MMLU.json", "r") as f:
            content = f.read()
            eval_tree = json.loads(content)
    except json.JSONDecodeError:
        with open("MMLU.json", "r") as f:
            content = f.read()
            brace_count = 0
            for i, char in enumerate(content):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[: i + 1]
                        break
            eval_tree = json.loads(content)

    print(f"Loaded {len(dove_scores)} DOVE scores")
    print(f"Loaded EvalTree with size {eval_tree.get('size', 'unknown')}")

    return dove_scores, eval_tree


def collect_question_indices(node):
    """Recursively collect all question indices from a subtree"""
    indices = []

    if isinstance(node, dict):
        if "subtrees" in node:
            subtrees = node["subtrees"]

            if isinstance(subtrees, int):
                indices.append(subtrees)
            elif isinstance(subtrees, list):
                for subtree in subtrees:
                    indices.extend(collect_question_indices(subtree))
            elif isinstance(subtrees, dict):
                indices.extend(collect_question_indices(subtree))

    return indices


def extract_mmlu_weaknesses(node, threshold=0.6, level=0, max_levels=3):
    """Extract weak branches from original MMLU tree using ranking/accuracy data"""
    weak_branches = []
    
    def traverse(current_node, current_level, path=""):
        if not isinstance(current_node, dict) or current_level >= max_levels:
            return
            
        capability = current_node.get("capability", "Unknown")
        
        # Check if this node has ranking data (accuracy scores)
        if "ranking" in current_node and isinstance(current_node["ranking"], dict):
            # Extract accuracy scores from ranking
            ranking_data = current_node["ranking"]
            accuracy_scores = []
            
            for model_name, score in ranking_data.items():
                if isinstance(score, (int, float)):
                    accuracy_scores.append(score)
            
            if accuracy_scores:
                mean_accuracy = statistics.mean(accuracy_scores)
                
                # If mean accuracy is below threshold, it's a weak branch
                if mean_accuracy < threshold:
                    question_indices = collect_question_indices(current_node)
                    weak_branches.append({
                        "capability": capability,
                        "accuracy": round(mean_accuracy, 3),
                        "level": current_level,
                        "path": path,
                        "question_count": len(question_indices),
                        "method": "Traditional MMLU"
                    })
        
        # Recurse into subtrees
        if "subtrees" in current_node and isinstance(current_node["subtrees"], list):
            for subtree in current_node["subtrees"]:
                new_path = f"{path} → {capability}" if path else capability
                traverse(subtree, current_level + 1, new_path)
    
    traverse(node, level)
    return weak_branches


def extract_dove_weaknesses(node, dove_scores, threshold=0.5, level=0, max_levels=3):
    """Extract weak branches from DOVE robustness analysis"""
    weak_branches = []
    
    def traverse(current_node, current_level, path=""):
        if not isinstance(current_node, dict) or current_level >= max_levels:
            return
            
        capability = current_node.get("capability", "Unknown")
        
        # Get question indices and calculate DOVE stats
        question_indices = collect_question_indices(current_node)
        subtree_scores = []
        for idx in question_indices:
            if str(idx) in dove_scores:
                subtree_scores.append(dove_scores[str(idx)])
        
        if subtree_scores and len(subtree_scores) >= 3:
            mean_robustness = statistics.mean(subtree_scores)
            
            # If mean robustness is below threshold, it's a weak branch
            if mean_robustness < threshold:
                weak_branches.append({
                    "capability": capability,
                    "robustness": round(mean_robustness, 3),
                    "level": current_level,
                    "path": path,
                    "question_count": len(subtree_scores),
                    "method": "DOVE Robustness"
                })
        
        # Recurse into subtrees
        if "subtrees" in current_node and isinstance(current_node["subtrees"], list):
            for subtree in current_node["subtrees"]:
                new_path = f"{path} → {capability}" if path else capability
                traverse(subtree, current_level + 1, new_path)
    
    traverse(node, level)
    return weak_branches


def shorten_capability(text, max_words=4):
    """Shorten capability names for visualization"""
    if not text:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Remove common words
    skip_words = {"and", "or", "the", "a", "an", "of", "for", "in", "on", "at", "to", "with", "by", "from", "analyzing", "evaluating", "synthesizing"}
    important_words = [w for w in words if w.lower() not in skip_words]
    
    if len(important_words) >= max_words:
        return " ".join(important_words[:max_words])
    else:
        return " ".join(words[:max_words])


def create_comparison_visualization(mmlu_weak, dove_weak):
    """Create comprehensive comparison visualization"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Robustness in Weakness Profiling: Traditional vs DOVE Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Top Weaknesses Comparison (Bar Plot)
    ax1.set_title('Top 10 Weakest Capabilities: Method Comparison', fontweight='bold')
    
    # Get top 10 from each method
    mmlu_top = sorted(mmlu_weak, key=lambda x: x.get('accuracy', 1))[:10]
    dove_top = sorted(dove_weak, key=lambda x: x.get('robustness', 1))[:10]
    
    # Create comparison data
    mmlu_names = [shorten_capability(item['capability'], 3) for item in mmlu_top]
    mmlu_scores = [item.get('accuracy', 0) for item in mmlu_top]
    dove_names = [shorten_capability(item['capability'], 3) for item in dove_top]
    dove_scores = [item.get('robustness', 0) for item in dove_top]
    
    y_pos_mmlu = np.arange(len(mmlu_names))
    y_pos_dove = np.arange(len(dove_names)) + len(mmlu_names) + 1
    
    bars1 = ax1.barh(y_pos_mmlu, mmlu_scores, alpha=0.8, color='skyblue', label='Traditional MMLU')
    bars2 = ax1.barh(y_pos_dove, dove_scores, alpha=0.8, color='coral', label='DOVE Robustness')
    
    ax1.set_yticks(list(y_pos_mmlu) + list(y_pos_dove))
    ax1.set_yticklabels(mmlu_names + dove_names, fontsize=9)
    ax1.set_xlabel('Weakness Score (Lower = Weaker)', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Score Distribution Comparison
    ax2.set_title('Weakness Score Distributions', fontweight='bold')
    
    mmlu_all_scores = [item.get('accuracy', 0) for item in mmlu_weak if 'accuracy' in item]
    dove_all_scores = [item.get('robustness', 0) for item in dove_weak if 'robustness' in item]
    
    ax2.hist(mmlu_all_scores, bins=15, alpha=0.7, color='skyblue', label=f'Traditional MMLU (n={len(mmlu_all_scores)})', density=True)
    ax2.hist(dove_all_scores, bins=15, alpha=0.7, color='coral', label=f'DOVE Robustness (n={len(dove_all_scores)})', density=True)
    ax2.set_xlabel('Weakness Score', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Method Overlap Analysis
    ax3.set_title('Weakness Detection Method Overlap', fontweight='bold')
    
    # Find overlapping capabilities (simplified matching by first few words)
    mmlu_caps = set(shorten_capability(item['capability'], 2) for item in mmlu_weak)
    dove_caps = set(shorten_capability(item['capability'], 2) for item in dove_weak)
    
    overlap = len(mmlu_caps.intersection(dove_caps))
    mmlu_only = len(mmlu_caps - dove_caps)
    dove_only = len(dove_caps - mmlu_caps)
    
    labels = ['Both Methods', 'Traditional Only', 'DOVE Only']
    sizes = [overlap, mmlu_only, dove_only]
    colors = ['lightgreen', 'skyblue', 'coral']
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    # 4. Question Coverage Analysis
    ax4.set_title('Question Coverage by Weakness Detection Method', fontweight='bold')
    
    mmlu_questions = sum(item.get('question_count', 0) for item in mmlu_weak)
    dove_questions = sum(item.get('question_count', 0) for item in dove_weak)
    
    methods = ['Traditional\nMMLU', 'DOVE\nRobustness']
    question_counts = [mmlu_questions, dove_questions]
    colors = ['skyblue', 'coral']
    
    bars = ax4.bar(methods, question_counts, color=colors, alpha=0.8)
    ax4.set_ylabel('Total Questions in Weak Branches', fontweight='bold')
    ax4.set_title('Question Coverage by Method', fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, question_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robustness_weakness_profiling_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('robustness_weakness_profiling_comparison.pdf', bbox_inches='tight')
    print("✅ Saved comparison visualization as PNG and PDF")
    
    return fig


def create_detailed_comparison_table(mmlu_weak, dove_weak):
    """Create detailed comparison table"""
    
    print("\n" + "="*100)
    print("DETAILED WEAKNESS PROFILING COMPARISON")
    print("="*100)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Traditional MMLU weak branches: {len(mmlu_weak)}")
    print(f"DOVE Robustness weak branches: {len(dove_weak)}")
    
    if mmlu_weak:
        mmlu_scores = [item.get('accuracy', 0) for item in mmlu_weak if 'accuracy' in item]
        if mmlu_scores:
            print(f"Traditional MMLU - Mean weakness: {statistics.mean(mmlu_scores):.3f}")
            print(f"Traditional MMLU - Weakest score: {min(mmlu_scores):.3f}")
    
    if dove_weak:
        dove_scores = [item.get('robustness', 0) for item in dove_weak if 'robustness' in item]
        if dove_scores:
            print(f"DOVE Robustness - Mean weakness: {statistics.mean(dove_scores):.3f}")
            print(f"DOVE Robustness - Weakest score: {min(dove_scores):.3f}")
    
    print(f"\n🔍 TOP 10 WEAKEST - TRADITIONAL MMLU:")
    print("-" * 80)
    mmlu_sorted = sorted(mmlu_weak, key=lambda x: x.get('accuracy', 1))[:10]
    for i, item in enumerate(mmlu_sorted, 1):
        capability = item['capability'][:60] + "..." if len(item['capability']) > 60 else item['capability']
        print(f"{i:2d}. {capability}")
        print(f"    Accuracy: {item.get('accuracy', 'N/A')} | Questions: {item.get('question_count', 0)} | Level: {item.get('level', 0)}")
    
    print(f"\n🎯 TOP 10 WEAKEST - DOVE ROBUSTNESS:")
    print("-" * 80)
    dove_sorted = sorted(dove_weak, key=lambda x: x.get('robustness', 1))[:10]
    for i, item in enumerate(dove_sorted, 1):
        capability = item['capability'][:60] + "..." if len(item['capability']) > 60 else item['capability']
        print(f"{i:2d}. {capability}")
        print(f"    Robustness: {item.get('robustness', 'N/A')} | Questions: {item.get('question_count', 0)} | Level: {item.get('level', 0)}")
    
    # Find unique insights from each method
    print(f"\n💡 UNIQUE INSIGHTS:")
    print("-" * 80)
    
    mmlu_caps = {shorten_capability(item['capability'], 3) for item in mmlu_weak}
    dove_caps = {shorten_capability(item['capability'], 3) for item in dove_weak}
    
    mmlu_unique = mmlu_caps - dove_caps
    dove_unique = dove_caps - mmlu_caps
    
    print(f"Weaknesses found ONLY by Traditional MMLU ({len(mmlu_unique)}):")
    for cap in list(mmlu_unique)[:5]:
        print(f"  • {cap}")
    
    print(f"\nWeaknesses found ONLY by DOVE Robustness ({len(dove_unique)}):")
    for cap in list(dove_unique)[:5]:
        print(f"  • {cap}")


def main():
    """Main execution function"""
    print("🔍 Comparing Weakness Profiling: Traditional MMLU vs DOVE Robustness")
    print("="*80)
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract weak branches using both methods
        print("\n📋 Extracting weak branches from Traditional MMLU tree...")
        mmlu_weak_branches = extract_mmlu_weaknesses(eval_tree, threshold=0.6)
        
        print("📋 Extracting weak branches from DOVE robustness analysis...")
        dove_weak_branches = extract_dove_weaknesses(eval_tree, dove_scores, threshold=0.5)
        
        print(f"\n✅ Extracted {len(mmlu_weak_branches)} weak branches from Traditional MMLU")
        print(f"✅ Extracted {len(dove_weak_branches)} weak branches from DOVE analysis")
        
        # Create visualizations
        print("\n📊 Creating comparison visualization...")
        fig = create_comparison_visualization(mmlu_weak_branches, dove_weak_branches)
        
        # Create detailed comparison
        create_detailed_comparison_table(mmlu_weak_branches, dove_weak_branches)
        
        print("\n" + "="*80)
        print("🎯 ROBUSTNESS IN WEAKNESS PROFILING - ANALYSIS COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("  • robustness_weakness_profiling_comparison.png")
        print("  • robustness_weakness_profiling_comparison.pdf")
        
        print("\n🔬 Key Research Insights:")
        print("  • Traditional MMLU focuses on accuracy-based weaknesses")
        print("  • DOVE reveals robustness-based weaknesses (input sensitivity)")
        print("  • Different methods identify different weak spots")
        print("  • Robustness analysis enables targeted question generation")
        print("  • Combined approach provides comprehensive weakness profiling")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 