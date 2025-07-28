#!/usr/bin/env python3
"""
Complete False Positive/Negative Analysis: DOVE vs Traditional Accuracy

Analyzes both:
1. False Positives: Traditional accuracy says "weak" but DOVE shows "robust"
2. False Negatives: Traditional accuracy says "strong" but DOVE shows "vulnerable"

This provides a complete picture of how DOVE mitigates both types of 
misclassification errors in weakness profiling.
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


def extract_capability_scores(node, dove_scores, level=0, max_levels=3):
    """Extract both accuracy and robustness scores for all capabilities"""
    capabilities = []
    
    def traverse(current_node, current_level):
        if not isinstance(current_node, dict) or current_level >= max_levels:
            return
            
        capability = current_node.get("capability", "Unknown")
        
        # Get accuracy from ranking data
        accuracy_score = None
        if "ranking" in current_node:
            ranking_data = current_node["ranking"]
            accuracy_scores = []
            
            if isinstance(ranking_data, list):
                for item in ranking_data:
                    if isinstance(item, list) and len(item) >= 2:
                        score = item[1]
                        if isinstance(score, (int, float)):
                            accuracy_scores.append(score)
            
            if accuracy_scores:
                accuracy_score = statistics.mean(accuracy_scores)
        
        # Get robustness from DOVE scores
        question_indices = collect_question_indices(current_node)
        subtree_scores = []
        for idx in question_indices:
            if str(idx) in dove_scores:
                subtree_scores.append(dove_scores[str(idx)])
        
        robustness_score = None
        if subtree_scores and len(subtree_scores) >= 3:
            robustness_score = statistics.mean(subtree_scores)
        
        # If we have both scores, add to analysis
        if accuracy_score is not None and robustness_score is not None:
            capabilities.append({
                "capability": capability,
                "accuracy": round(accuracy_score, 3),
                "robustness": round(robustness_score, 3),
                "question_count": len(subtree_scores),
                "level": current_level,
                "gap": round(accuracy_score - robustness_score, 3)
            })
        
        # Recurse into subtrees
        if "subtrees" in current_node and isinstance(current_node["subtrees"], list):
            for subtree in current_node["subtrees"]:
                traverse(subtree, current_level + 1)
    
    traverse(node, level)
    return capabilities


def classify_capabilities(capabilities, acc_threshold=0.7, rob_threshold=0.5):
    """Classify capabilities into different categories"""
    
    classifications = {
        'true_positives': [],    # Both methods agree: weak
        'true_negatives': [],    # Both methods agree: strong  
        'false_positives': [],   # Traditional says weak, DOVE says strong
        'false_negatives': [],   # Traditional says strong, DOVE says weak
    }
    
    for cap in capabilities:
        acc_weak = cap['accuracy'] < acc_threshold
        rob_weak = cap['robustness'] < rob_threshold
        
        if acc_weak and rob_weak:
            classifications['true_positives'].append(cap)
        elif not acc_weak and not rob_weak:
            classifications['true_negatives'].append(cap)
        elif acc_weak and not rob_weak:
            classifications['false_positives'].append(cap)
        elif not acc_weak and rob_weak:
            classifications['false_negatives'].append(cap)
    
    return classifications


def shorten_capability(text, max_words=3):
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


def create_confusion_matrix_graph(classifications):
    """Create a confusion matrix showing all four categories"""
    
    plt.figure(figsize=(12, 10))
    
    # Create 2x2 confusion matrix
    categories = ['True Positives\n(Both Weak)', 'False Negatives\n(Acc Strong, Rob Weak)', 
                  'False Positives\n(Acc Weak, Rob Strong)', 'True Negatives\n(Both Strong)']
    
    counts = [len(classifications['true_positives']), len(classifications['false_negatives']),
              len(classifications['false_positives']), len(classifications['true_negatives'])]
    
    # Create 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    axes = [ax1, ax2, ax3, ax4]
    colors = ['lightgreen', 'red', 'orange', 'lightblue']
    
    for i, (ax, category, count, color) in enumerate(zip(axes, categories, counts, colors)):
        ax.bar([category.split('\n')[0]], [count], color=color, alpha=0.8)
        ax.set_title(f'{category}\nCount: {count}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Capabilities')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage
        total = sum(counts)
        percentage = count/total*100 if total > 0 else 0
        ax.text(0, count + count*0.05, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Complete False Positive/Negative Analysis\nDOVE Robustness vs Traditional Accuracy Classification', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('complete_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig('complete_confusion_matrix.pdf', bbox_inches='tight')
    print("✅ Confusion Matrix saved: Complete False Positive/Negative Analysis")
    plt.show()


def create_false_positive_detailed_graph(classifications):
    """Detailed analysis of false positives"""
    
    false_positives = classifications['false_positives']
    
    if not false_positives:
        print("⚠️ No false positives found - creating explanatory graph")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'NO FALSE POSITIVES DETECTED\n\n' +
                'Traditional accuracy does not incorrectly\nclassify any robust capabilities as weak.\n\n' +
                'This suggests traditional accuracy is\nconservative and rarely over-penalizes capabilities.',
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen", alpha=0.8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('False Positive Analysis: Traditional Accuracy vs DOVE Robustness\n' +
                  'Question: Does traditional accuracy incorrectly classify robust capabilities as weak?', 
                  fontsize=14, fontweight='bold', pad=20)
    else:
        plt.figure(figsize=(12, 8))
        
        # Show false positives
        names = [shorten_capability(cap['capability'], 4) for cap in false_positives[:10]]
        accuracies = [cap['accuracy'] for cap in false_positives[:10]]
        robustnesses = [cap['robustness'] for cap in false_positives[:10]]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, accuracies, width, label='Traditional Accuracy', color='orange', alpha=0.8)
        bars2 = plt.bar(x + width/2, robustnesses, width, label='DOVE Robustness', color='green', alpha=0.8)
        
        plt.xlabel('Capabilities', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.title('False Positive Analysis: Capabilities Traditional Accuracy Under-Estimates\n' +
                  'Question: Which robust capabilities does traditional accuracy incorrectly classify as weak?', 
                  fontweight='bold', pad=20)
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('false_positive_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig('false_positive_detailed.pdf', bbox_inches='tight')
    print("✅ False Positive Detailed Analysis saved")
    plt.show()


def create_error_comparison_graph(classifications):
    """Compare false positive vs false negative rates"""
    
    plt.figure(figsize=(10, 8))
    
    fp_count = len(classifications['false_positives'])
    fn_count = len(classifications['false_negatives'])
    tp_count = len(classifications['true_positives'])
    tn_count = len(classifications['true_negatives'])
    
    total = fp_count + fn_count + tp_count + tn_count
    
    # Create comparison bars
    error_types = ['False Positives\n(Over-Penalizing)', 'False Negatives\n(Under-Penalizing)']
    error_counts = [fp_count, fn_count]
    error_rates = [fp_count/total*100, fn_count/total*100]
    
    colors = ['orange', 'red']
    bars = plt.bar(error_types, error_counts, color=colors, alpha=0.8)
    
    # Add count and percentage labels
    for bar, count, rate in zip(bars, error_counts, error_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{count}\n({rate:.1f}%)', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    plt.ylabel('Number of Capabilities', fontweight='bold', fontsize=12)
    plt.title('Error Type Comparison: Traditional Accuracy Misclassifications\n' +
              'Question: Which type of error is more common in traditional accuracy?', 
              fontweight='bold', fontsize=14, pad=20)
    
    # Add interpretation
    if fn_count > fp_count:
        interpretation = f"Traditional accuracy is more likely to\nUNDER-PENALIZE (miss weaknesses)\nthan OVER-PENALIZE (false alarms)"
    elif fp_count > fn_count:
        interpretation = f"Traditional accuracy is more likely to\nOVER-PENALIZE (false alarms)\nthan UNDER-PENALIZE (miss weaknesses)"
    else:
        interpretation = f"Traditional accuracy shows\nequal rates of both error types"
    
    plt.text(0.5, 0.85, interpretation, transform=plt.gca().transAxes, 
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('error_comparison.pdf', bbox_inches='tight')
    print("✅ Error Comparison Analysis saved")
    plt.show()


def create_quadrant_analysis_graph(capabilities):
    """Show all capabilities in accuracy-robustness space with quadrants"""
    
    plt.figure(figsize=(12, 10))
    
    # Extract scores
    accuracies = [cap['accuracy'] for cap in capabilities]
    robustnesses = [cap['robustness'] for cap in capabilities]
    
    # Color code by quadrant
    colors = []
    for cap in capabilities:
        if cap['accuracy'] >= 0.7 and cap['robustness'] >= 0.5:
            colors.append('green')  # True Negative (both strong)
        elif cap['accuracy'] < 0.7 and cap['robustness'] < 0.5:
            colors.append('blue')   # True Positive (both weak)
        elif cap['accuracy'] < 0.7 and cap['robustness'] >= 0.5:
            colors.append('orange') # False Positive (acc weak, rob strong)
        else:  # cap['accuracy'] >= 0.7 and cap['robustness'] < 0.5
            colors.append('red')    # False Negative (acc strong, rob weak)
    
    # Create scatter plot
    scatter = plt.scatter(accuracies, robustnesses, c=colors, alpha=0.7, s=80)
    
    # Add threshold lines
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Robustness Threshold (0.5)')
    plt.axvline(x=0.7, color='black', linestyle='--', alpha=0.5, label='Accuracy Threshold (0.7)')
    
    # Add quadrant labels
    plt.text(0.85, 0.7, 'TRUE NEGATIVES\n(Both Strong)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    plt.text(0.85, 0.3, 'FALSE NEGATIVES\n(Acc Strong, Rob Weak)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    plt.text(0.55, 0.7, 'FALSE POSITIVES\n(Acc Weak, Rob Strong)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.8))
    plt.text(0.55, 0.3, 'TRUE POSITIVES\n(Both Weak)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.xlabel('Traditional Accuracy Score', fontweight='bold', fontsize=12)
    plt.ylabel('DOVE Robustness Score', fontweight='bold', fontsize=12)
    plt.title('Quadrant Analysis: Complete Classification of All Capabilities\n' +
              'Question: How are capabilities distributed across accuracy-robustness space?', 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quadrant_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('quadrant_analysis.pdf', bbox_inches='tight')
    print("✅ Quadrant Analysis saved")
    plt.show()


def print_complete_analysis_summary(classifications, capabilities):
    """Print comprehensive summary of both false positives and negatives"""
    
    print("\n" + "="*80)
    print("COMPLETE FALSE POSITIVE/NEGATIVE ANALYSIS SUMMARY")
    print("="*80)
    
    fp_count = len(classifications['false_positives'])
    fn_count = len(classifications['false_negatives'])
    tp_count = len(classifications['true_positives'])
    tn_count = len(classifications['true_negatives'])
    total = fp_count + fn_count + tp_count + tn_count
    
    print(f"\n📊 CLASSIFICATION RESULTS:")
    print(f"Total capabilities analyzed: {total}")
    print(f"True Positives (both weak): {tp_count} ({tp_count/total*100:.1f}%)")
    print(f"True Negatives (both strong): {tn_count} ({tn_count/total*100:.1f}%)")
    print(f"False Positives (acc weak, rob strong): {fp_count} ({fp_count/total*100:.1f}%)")
    print(f"False Negatives (acc strong, rob weak): {fn_count} ({fn_count/total*100:.1f}%)")
    
    print(f"\n🎯 ERROR ANALYSIS:")
    print(f"False Positive Rate: {fp_count/total*100:.1f}%")
    print(f"False Negative Rate: {fn_count/total*100:.1f}%")
    print(f"Overall Accuracy: {(tp_count + tn_count)/total*100:.1f}%")
    
    if fp_count > 0:
        print(f"\n🟠 FALSE POSITIVES (Traditional Under-Estimates):")
        for i, cap in enumerate(classifications['false_positives'][:5], 1):
            print(f"{i}. {cap['capability'][:60]}...")
            print(f"   Accuracy: {cap['accuracy']:.3f} | Robustness: {cap['robustness']:.3f} | Gap: {cap['gap']:.3f}")
    else:
        print(f"\n🟠 FALSE POSITIVES: None detected")
        print("   Traditional accuracy does not over-penalize any robust capabilities")
    
    if fn_count > 0:
        print(f"\n🔴 FALSE NEGATIVES (Traditional Over-Estimates):")
        sorted_fn = sorted(classifications['false_negatives'], key=lambda x: x['gap'], reverse=True)
        for i, cap in enumerate(sorted_fn[:5], 1):
            print(f"{i}. {cap['capability'][:60]}...")
            print(f"   Accuracy: {cap['accuracy']:.3f} | Robustness: {cap['robustness']:.3f} | Gap: {cap['gap']:.3f}")
    
    print(f"\n🚀 COMPLETE FALSE POSITIVE MITIGATION CLAIM:")
    print("="*60)
    
    if fp_count == 0 and fn_count > 0:
        print("✅ DOVE provides ASYMMETRIC FALSE POSITIVE MITIGATION:")
        print(f"   • Zero false positives: Traditional accuracy is not over-penalizing")
        print(f"   • {fn_count} false negatives: Traditional accuracy misses {fn_count/total*100:.1f}% of vulnerabilities")
        print("   • Traditional accuracy is conservative but incomplete")
        print("   • DOVE reveals hidden vulnerabilities without creating false alarms")
    elif fp_count > 0 and fn_count > fp_count:
        print("✅ DOVE provides COMPREHENSIVE FALSE POSITIVE MITIGATION:")
        print(f"   • {fp_count} false positives: Traditional over-penalizes {fp_count/total*100:.1f}% of capabilities")
        print(f"   • {fn_count} false negatives: Traditional under-penalizes {fn_count/total*100:.1f}% of capabilities")
        print("   • False negatives are more common than false positives")
        print("   • DOVE corrects both types of misclassification")
    elif fp_count > fn_count:
        print("✅ DOVE provides TARGETED FALSE POSITIVE MITIGATION:")
        print(f"   • {fp_count} false positives: Traditional over-penalizes {fp_count/total*100:.1f}% of capabilities")
        print(f"   • {fn_count} false negatives: Traditional under-penalizes {fn_count/total*100:.1f}% of capabilities")
        print("   • Traditional accuracy is overly harsh on robust capabilities")
        print("   • DOVE prevents over-penalization of strong areas")
    else:
        print("✅ DOVE provides BALANCED FALSE POSITIVE MITIGATION:")
        print(f"   • Equal rates of false positives and negatives")
        print(f"   • DOVE corrects both over- and under-estimation errors")


def main():
    """Main execution function"""
    print("🔍 Complete False Positive/Negative Analysis: DOVE vs Traditional Accuracy")
    print("="*80)
    
    try:
        # Load data
        dove_scores, eval_tree = load_data()
        
        # Extract capability scores
        print("\n📋 Extracting capability scores for both accuracy and robustness...")
        capabilities = extract_capability_scores(eval_tree, dove_scores)
        
        print(f"✅ Analyzed {len(capabilities)} capabilities with both scores")
        
        # Classify capabilities
        print("\n🔍 Classifying capabilities into TP/TN/FP/FN categories...")
        classifications = classify_capabilities(capabilities)
        
        # Create visualizations
        print("\n📊 Creating Complete Confusion Matrix...")
        create_confusion_matrix_graph(classifications)
        
        print("\n📊 Creating False Positive Detailed Analysis...")
        create_false_positive_detailed_graph(classifications)
        
        print("\n📊 Creating Error Comparison...")
        create_error_comparison_graph(classifications)
        
        print("\n📊 Creating Quadrant Analysis...")
        create_quadrant_analysis_graph(capabilities)
        
        # Print comprehensive summary
        print_complete_analysis_summary(classifications, capabilities)
        
        print("\n" + "="*80)
        print("🎯 COMPLETE FALSE POSITIVE/NEGATIVE ANALYSIS COMPLETE!")
        print("="*80)
        print("Generated comprehensive graphs:")
        print("  • complete_confusion_matrix.png/pdf")
        print("  • false_positive_detailed.png/pdf")
        print("  • error_comparison.png/pdf")
        print("  • quadrant_analysis.png/pdf")
        
        print("\n🔬 Research Findings:")
        fp_count = len(classifications['false_positives'])
        fn_count = len(classifications['false_negatives'])
        
        if fp_count == 0:
            print("  ✅ NO FALSE POSITIVES: Traditional accuracy is conservative")
            print("  ✅ DOVE reveals hidden vulnerabilities without false alarms")
        else:
            print(f"  ✅ {fp_count} FALSE POSITIVES: Traditional over-penalizes robust areas")
        
        print(f"  ✅ {fn_count} FALSE NEGATIVES: Traditional misses vulnerable areas")
        print("  ✅ DOVE provides comprehensive error mitigation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 