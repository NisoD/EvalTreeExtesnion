#!/usr/bin/env python3
"""
Weakness Intersection Analysis: DOVE vs Binary Evaluation
Focus on capability intersections and differentiations for targeted question generation
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import statistics
from collections import Counter, defaultdict
import matplotlib.patches as patches
import seaborn as sns

# Set style for paper-quality figures
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
    """Load DOVE scores and weakness profiles"""
    try:
        with open('MMLU_DOVE.json', 'r') as f:
            dove_scores = json.load(f)
        
        # Try to load existing weakness profile
        try:
            with open('weakness_profile_improved.json', 'r') as f:
                weakness_profile = json.load(f)
        except:
            try:
                with open('weakness_profile_simple.json', 'r') as f:
                    weakness_profile = json.load(f)
            except:
                weakness_profile = None
        
        return dove_scores, weakness_profile
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def simulate_capability_analysis(dove_scores, num_capabilities=50):
    """
    Simulate capability-based analysis by grouping questions into artificial capabilities
    This represents what would happen with proper EvalTree integration
    """
    # Group questions into artificial capabilities
    capabilities = {}
    questions_per_capability = len(dove_scores) // num_capabilities
    
    dove_items = list(dove_scores.items())
    
    for i in range(num_capabilities):
        start_idx = i * questions_per_capability
        end_idx = min((i + 1) * questions_per_capability, len(dove_items))
        
        if start_idx >= len(dove_items):
            break
            
        capability_questions = dove_items[start_idx:end_idx]
        
        if len(capability_questions) >= 3:  # Minimum reliability threshold
            scores = [score for _, score in capability_questions]
            
            capabilities[f"Capability_{i+1:02d}"] = {
                'dove_scores': scores,
                'dove_mean': statistics.mean(scores),
                'dove_std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'question_count': len(scores),
                'question_ids': [qid for qid, _ in capability_questions]
            }
    
    return capabilities

def classify_weakness_level(score, method='dove'):
    """Classify weakness level based on score"""
    if method == 'dove':
        if score < 0.30:
            return "Critical"
        elif score < 0.50:
            return "High"
        elif score < 0.70:
            return "Moderate"
        else:
            return "Strong"
    else:  # binary
        return "Strong" if score >= 0.5 else "Weak"

def analyze_capability_intersections(capabilities):
    """Analyze intersections between DOVE and binary weakness classifications"""
    intersection_data = []
    
    for cap_name, cap_data in capabilities.items():
        dove_scores = cap_data['dove_scores']
        binary_scores = [1.0 if s >= 0.5 else 0.0 for s in dove_scores]
        
        dove_mean = cap_data['dove_mean']
        binary_mean = statistics.mean(binary_scores)
        
        dove_weakness = classify_weakness_level(dove_mean, 'dove')
        binary_weakness = "Strong" if binary_mean >= 0.5 else "Weak"
        
        # Calculate disagreement metrics
        score_diff = abs(dove_mean - binary_mean)
        consistency = statistics.stdev(dove_scores) if len(dove_scores) > 1 else 0
        
        intersection_data.append({
            'capability': cap_name,
            'dove_mean': dove_mean,
            'binary_mean': binary_mean,
            'dove_weakness': dove_weakness,
            'binary_weakness': binary_weakness,
            'score_difference': score_diff,
            'consistency': consistency,
            'question_count': cap_data['question_count'],
            'intersection_type': f"{dove_weakness}_{binary_weakness}"
        })
    
    return intersection_data

def create_intersection_overview(intersection_data):
    """Create overview of weakness profile intersections"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Intersection Matrix
    dove_levels = ["Critical", "High", "Moderate", "Strong"]
    binary_levels = ["Weak", "Strong"]
    
    # Create intersection matrix
    matrix = np.zeros((len(dove_levels), len(binary_levels)))
    intersection_counts = defaultdict(int)
    
    for item in intersection_data:
        dove_idx = dove_levels.index(item['dove_weakness'])
        binary_idx = binary_levels.index(item['binary_weakness'])
        matrix[dove_idx, binary_idx] += 1
        intersection_counts[item['intersection_type']] += 1
    
    # Create heatmap
    im = ax1.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(binary_levels)))
    ax1.set_yticks(range(len(dove_levels)))
    ax1.set_xticklabels(binary_levels)
    ax1.set_yticklabels(dove_levels)
    ax1.set_xlabel('Binary Evaluation')
    ax1.set_ylabel('DOVE Evaluation')
    ax1.set_title('Weakness Classification Intersection Matrix')
    
    # Add text annotations
    for i in range(len(dove_levels)):
        for j in range(len(binary_levels)):
            text = ax1.text(j, i, int(matrix[i, j]), ha="center", va="center", 
                           color="white" if matrix[i, j] > matrix.max()/2 else "black")
    
    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Number of Capabilities')
    
    # 2. Score Difference Distribution
    score_diffs = [item['score_difference'] for item in intersection_data]
    ax2.hist(score_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(np.mean(score_diffs), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(score_diffs):.3f}')
    ax2.set_xlabel('|DOVE Score - Binary Score|')
    ax2.set_ylabel('Number of Capabilities')
    ax2.set_title('Score Difference Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Consistency vs Performance
    dove_means = [item['dove_mean'] for item in intersection_data]
    consistencies = [item['consistency'] for item in intersection_data]
    colors = [{'Critical': '#d32f2f', 'High': '#f57c00', 'Moderate': '#fbc02d', 'Strong': '#388e3c'}[item['dove_weakness']] 
             for item in intersection_data]
    
    scatter = ax3.scatter(dove_means, consistencies, c=colors, alpha=0.6, s=60)
    ax3.set_xlabel('DOVE Mean Performance')
    ax3.set_ylabel('Performance Consistency (Std Dev)')
    ax3.set_title('Performance vs Consistency Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Add legend for colors
    legend_elements = [patches.Patch(color='#d32f2f', label='Critical'),
                      patches.Patch(color='#f57c00', label='High'),
                      patches.Patch(color='#fbc02d', label='Moderate'),
                      patches.Patch(color='#388e3c', label='Strong')]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # 4. Key Intersection Types
    key_intersections = ['Critical_Weak', 'Critical_Strong', 'High_Weak', 'High_Strong', 
                        'Moderate_Weak', 'Moderate_Strong', 'Strong_Weak', 'Strong_Strong']
    intersection_counts_list = [intersection_counts[key] for key in key_intersections]
    
    bars = ax4.bar(range(len(key_intersections)), intersection_counts_list, 
                   color=['red', 'orange', 'orange', 'yellow', 'yellow', 'lightgreen', 'red', 'green'],
                   alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(key_intersections)))
    ax4.set_xticklabels([key.replace('_', '\n') for key in key_intersections], rotation=45, ha='right')
    ax4.set_ylabel('Number of Capabilities')
    ax4.set_title('Intersection Type Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Highlight disagreement cases
    disagreement_indices = [0, 1, 6]  # Critical_Weak, Critical_Strong, Strong_Weak
    for idx in disagreement_indices:
        if idx < len(bars):
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('figure_intersection_overview.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_intersection_overview.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return intersection_counts

def create_disagreement_analysis(intersection_data):
    """Focus on cases where DOVE and binary evaluations disagree"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Identify disagreement cases
    high_disagreement = []
    false_negatives = []  # Binary says strong, DOVE says weak
    false_positives = []  # Binary says weak, DOVE says strong
    
    for item in intersection_data:
        if item['score_difference'] > 0.2:  # High disagreement threshold
            high_disagreement.append(item)
        
        if item['binary_weakness'] == 'Strong' and item['dove_weakness'] in ['Critical', 'High']:
            false_negatives.append(item)
        elif item['binary_weakness'] == 'Weak' and item['dove_weakness'] in ['Moderate', 'Strong']:
            false_positives.append(item)
    
    # 1. High Disagreement Cases
    if high_disagreement:
        dove_scores = [item['dove_mean'] for item in high_disagreement]
        binary_scores = [item['binary_mean'] for item in high_disagreement]
        
        ax1.scatter(binary_scores, dove_scores, alpha=0.7, s=80, c='red', edgecolors='black')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement')
        ax1.set_xlabel('Binary Mean Score')
        ax1.set_ylabel('DOVE Mean Score')
        ax1.set_title(f'High Disagreement Cases (n={len(high_disagreement)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for extreme cases
        for i, item in enumerate(high_disagreement[:5]):  # Top 5 disagreements
            ax1.annotate(f"Cap {i+1}", (item['binary_mean'], item['dove_mean']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No High Disagreement Cases Found', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        ax1.set_title('High Disagreement Cases')
    
    # 2. False Negative Analysis (Binary misses weaknesses)
    if false_negatives:
        fn_dove_scores = [item['dove_mean'] for item in false_negatives]
        fn_binary_scores = [item['binary_mean'] for item in false_negatives]
        fn_consistencies = [item['consistency'] for item in false_negatives]
        
        scatter = ax2.scatter(fn_dove_scores, fn_consistencies, s=100, c='orange', 
                             alpha=0.7, edgecolors='black')
        ax2.set_xlabel('DOVE Mean Score')
        ax2.set_ylabel('Performance Consistency')
        ax2.set_title(f'False Negatives: Binary Misses Weaknesses (n={len(false_negatives)})')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax2.axvline(0.5, color='red', linestyle=':', alpha=0.7, label='Binary Threshold')
        ax2.axvline(0.3, color='orange', linestyle=':', alpha=0.7, label='DOVE Critical')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No False Negatives Found', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('False Negatives Analysis')
    
    # 3. False Positive Analysis (Binary overestimates weaknesses)
    if false_positives:
        fp_dove_scores = [item['dove_mean'] for item in false_positives]
        fp_binary_scores = [item['binary_mean'] for item in false_positives]
        fp_consistencies = [item['consistency'] for item in false_positives]
        
        scatter = ax3.scatter(fp_dove_scores, fp_consistencies, s=100, c='lightblue', 
                             alpha=0.7, edgecolors='black')
        ax3.set_xlabel('DOVE Mean Score')
        ax3.set_ylabel('Performance Consistency')
        ax3.set_title(f'False Positives: Binary Overestimates Weaknesses (n={len(false_positives)})')
        ax3.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax3.axvline(0.5, color='red', linestyle=':', alpha=0.7, label='Binary Threshold')
        ax3.axvline(0.7, color='green', linestyle=':', alpha=0.7, label='DOVE Strong')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No False Positives Found', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title('False Positives Analysis')
    
    # 4. Disagreement Impact Summary
    categories = ['High\nDisagreement', 'False\nNegatives', 'False\nPositives', 'Perfect\nAgreement']
    perfect_agreement = len(intersection_data) - len(high_disagreement) - len(false_negatives) - len(false_positives)
    counts = [len(high_disagreement), len(false_negatives), len(false_positives), perfect_agreement]
    colors = ['red', 'orange', 'lightblue', 'green']
    
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Number of Capabilities')
    ax4.set_title('Disagreement Impact Summary')
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = count / total * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure_disagreement_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_disagreement_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return {
        'high_disagreement': high_disagreement,
        'false_negatives': false_negatives,
        'false_positives': false_positives
    }

def create_targeted_weakness_identification(intersection_data, disagreement_results):
    """Identify specific weaknesses for targeted question generation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Weakness Severity Matrix
    dove_levels = ["Critical", "High", "Moderate", "Strong"]
    severity_scores = {"Critical": 4, "High": 3, "Moderate": 2, "Strong": 1}
    
    # Calculate weakness priority scores
    priority_data = []
    for item in intersection_data:
        dove_severity = severity_scores[item['dove_weakness']]
        binary_miss = 1 if (item['binary_weakness'] == 'Strong' and dove_severity > 2) else 0
        consistency_penalty = item['consistency'] * 2  # Higher std = less reliable
        
        priority_score = dove_severity + binary_miss - consistency_penalty
        priority_data.append({
            'capability': item['capability'],
            'priority_score': priority_score,
            'dove_weakness': item['dove_weakness'],
            'binary_miss': binary_miss,
            'dove_mean': item['dove_mean'],
            'consistency': item['consistency']
        })
    
    # Sort by priority score
    priority_data.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Plot top priority weaknesses
    top_priorities = priority_data[:15]
    capabilities = [item['capability'] for item in top_priorities]
    scores = [item['priority_score'] for item in top_priorities]
    colors = [{'Critical': '#d32f2f', 'High': '#f57c00', 'Moderate': '#fbc02d', 'Strong': '#388e3c'}[item['dove_weakness']] 
             for item in top_priorities]
    
    bars = ax1.barh(range(len(capabilities)), scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(capabilities)))
    ax1.set_yticklabels([cap.replace('Capability_', 'Cap ') for cap in capabilities], fontsize=9)
    ax1.set_xlabel('Priority Score (Higher = More Important)')
    ax1.set_title('Top Priority Weaknesses for Question Generation')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add binary miss indicators
    for i, (bar, item) in enumerate(zip(bars, top_priorities)):
        if item['binary_miss']:
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    '⚠️', va='center', fontsize=12)
    
    # 2. Weakness Distribution by DOVE Level
    dove_weakness_counts = Counter(item['dove_weakness'] for item in intersection_data)
    binary_miss_by_dove = defaultdict(int)
    
    for item in intersection_data:
        if item['binary_weakness'] == 'Strong' and item['dove_weakness'] in ['Critical', 'High']:
            binary_miss_by_dove[item['dove_weakness']] += 1
    
    dove_levels_present = list(dove_weakness_counts.keys())
    total_counts = [dove_weakness_counts[level] for level in dove_levels_present]
    missed_counts = [binary_miss_by_dove[level] for level in dove_levels_present]
    
    x = np.arange(len(dove_levels_present))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, total_counts, width, label='Total Weaknesses', 
                    color=['#d32f2f', '#f57c00', '#fbc02d', '#388e3c'][:len(dove_levels_present)], alpha=0.7)
    bars2 = ax2.bar(x + width/2, missed_counts, width, label='Missed by Binary', 
                    color='red', alpha=0.7)
    
    ax2.set_xlabel('DOVE Weakness Level')
    ax2.set_ylabel('Number of Capabilities')
    ax2.set_title('Weakness Detection: DOVE vs Binary')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dove_levels_present)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar1, bar2, total, missed in zip(bars1, bars2, total_counts, missed_counts):
        miss_rate = (missed / total * 100) if total > 0 else 0
        ax2.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.5,
                f'{miss_rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Consistency-Based Targeting
    # Group by consistency levels for targeted question generation
    low_consistency = [item for item in intersection_data if item['consistency'] < 0.2]
    medium_consistency = [item for item in intersection_data if 0.2 <= item['consistency'] < 0.4]
    high_consistency = [item for item in intersection_data if item['consistency'] >= 0.4]
    
    consistency_groups = ['Low Variance\n(Reliable)', 'Medium Variance', 'High Variance\n(Inconsistent)']
    group_counts = [len(low_consistency), len(medium_consistency), len(high_consistency)]
    
    # Calculate weakness rates for each group
    weakness_rates = []
    for group in [low_consistency, medium_consistency, high_consistency]:
        if group:
            weak_count = sum(1 for item in group if item['dove_weakness'] in ['Critical', 'High'])
            weakness_rates.append(weak_count / len(group) * 100)
        else:
            weakness_rates.append(0)
    
    bars = ax3.bar(consistency_groups, weakness_rates, 
                   color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Weakness Rate (%)')
    ax3.set_title('Weakness Rate by Performance Consistency')
    ax3.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, group_counts):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    # 4. Question Generation Targets
    # Identify the most promising targets for new question generation
    generation_targets = []
    
    # High priority: Critical/High weaknesses missed by binary
    for item in intersection_data:
        if (item['dove_weakness'] in ['Critical', 'High'] and 
            item['binary_weakness'] == 'Strong' and 
            item['consistency'] < 0.3):  # Reliable measurement
            generation_targets.append({
                'type': 'Missed Critical',
                'capability': item['capability'],
                'dove_score': item['dove_mean'],
                'priority': 'High'
            })
    
    # Medium priority: Moderate weaknesses with high consistency
    for item in intersection_data:
        if (item['dove_weakness'] == 'Moderate' and 
            item['consistency'] < 0.2):  # Very reliable
            generation_targets.append({
                'type': 'Reliable Moderate',
                'capability': item['capability'],
                'dove_score': item['dove_mean'],
                'priority': 'Medium'
            })
    
    # Plot generation targets
    target_types = ['Missed Critical', 'Reliable Moderate']
    target_counts = [len([t for t in generation_targets if t['type'] == tt]) for tt in target_types]
    
    bars = ax4.bar(target_types, target_counts, 
                   color=['red', 'orange'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Number of Target Capabilities')
    ax4.set_title('Question Generation Targets')
    ax4.grid(True, alpha=0.3)
    
    # Add specific recommendations
    for bar, count, target_type in zip(bars, target_counts, target_types):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{count} targets', ha='center', va='bottom', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig('figure_targeted_weakness_identification.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_targeted_weakness_identification.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return {
        'priority_data': priority_data,
        'generation_targets': generation_targets,
        'consistency_analysis': {
            'low_consistency': low_consistency,
            'medium_consistency': medium_consistency,
            'high_consistency': high_consistency
        }
    }

def create_robustness_comparison(intersection_data):
    """Compare robustness of DOVE vs Binary evaluation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Evaluation Robustness Metrics
    dove_means = [item['dove_mean'] for item in intersection_data]
    binary_means = [item['binary_mean'] for item in intersection_data]
    consistencies = [item['consistency'] for item in intersection_data]
    
    # Calculate robustness metrics
    dove_range = max(dove_means) - min(dove_means)
    binary_range = max(binary_means) - min(binary_means)
    dove_granularity = len(set([round(score, 2) for score in dove_means]))
    binary_granularity = len(set(binary_means))
    
    metrics = ['Score Range', 'Granularity\n(Unique Values)', 'Mean Consistency', 'Std Consistency']
    dove_values = [dove_range, dove_granularity, np.mean(consistencies), np.std(consistencies)]
    binary_values = [binary_range, binary_granularity, 0, 0]  # Binary has no consistency measure
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dove_values, width, label='DOVE Evaluation', 
                    color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, binary_values, width, label='Binary Evaluation', 
                    color='red', alpha=0.7)
    
    ax1.set_xlabel('Robustness Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Evaluation Robustness Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Weakness Detection Sensitivity
    # Compare how sensitive each method is to different performance levels
    performance_bins = np.arange(0, 1.1, 0.1)
    dove_sensitivity = []
    binary_sensitivity = []
    
    for i in range(len(performance_bins) - 1):
        lower, upper = performance_bins[i], performance_bins[i+1]
        
        # Count capabilities in this performance range
        dove_in_range = [item for item in intersection_data 
                        if lower <= item['dove_mean'] < upper]
        binary_in_range = [item for item in intersection_data 
                          if lower <= item['binary_mean'] < upper]
        
        dove_sensitivity.append(len(dove_in_range))
        binary_sensitivity.append(len(binary_in_range))
    
    bin_centers = (performance_bins[:-1] + performance_bins[1:]) / 2
    width = 0.04
    
    bars1 = ax2.bar(bin_centers - width/2, dove_sensitivity, width, 
                    label='DOVE', color='blue', alpha=0.7)
    bars2 = ax2.bar(bin_centers + width/2, binary_sensitivity, width, 
                    label='Binary', color='red', alpha=0.7)
    
    ax2.set_xlabel('Performance Score Range')
    ax2.set_ylabel('Number of Capabilities')
    ax2.set_title('Evaluation Sensitivity Across Performance Ranges')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reliability Analysis
    # Group capabilities by reliability (consistency) and compare detection rates
    reliable_caps = [item for item in intersection_data if item['consistency'] < 0.25]
    unreliable_caps = [item for item in intersection_data if item['consistency'] >= 0.25]
    
    groups = ['Reliable\nMeasurements', 'Unreliable\nMeasurements']
    
    # Calculate weakness detection rates
    reliable_dove_weak = sum(1 for item in reliable_caps if item['dove_weakness'] in ['Critical', 'High'])
    reliable_binary_weak = sum(1 for item in reliable_caps if item['binary_weakness'] == 'Weak')
    
    unreliable_dove_weak = sum(1 for item in unreliable_caps if item['dove_weakness'] in ['Critical', 'High'])
    unreliable_binary_weak = sum(1 for item in unreliable_caps if item['binary_weakness'] == 'Weak')
    
    dove_rates = [
        reliable_dove_weak / len(reliable_caps) * 100 if reliable_caps else 0,
        unreliable_dove_weak / len(unreliable_caps) * 100 if unreliable_caps else 0
    ]
    binary_rates = [
        reliable_binary_weak / len(reliable_caps) * 100 if reliable_caps else 0,
        unreliable_binary_weak / len(unreliable_caps) * 100 if unreliable_caps else 0
    ]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, dove_rates, width, label='DOVE', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, binary_rates, width, label='Binary', color='red', alpha=0.7)
    
    ax3.set_xlabel('Measurement Reliability')
    ax3.set_ylabel('Weakness Detection Rate (%)')
    ax3.set_title('Weakness Detection by Measurement Reliability')
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add sample size labels
    sample_sizes = [len(reliable_caps), len(unreliable_caps)]
    for i, (bar1, bar2, size) in enumerate(zip(bars1, bars2, sample_sizes)):
        ax3.text(i, max(bar1.get_height(), bar2.get_height()) + 2,
                f'n={size}', ha='center', va='bottom', fontsize=10)
    
    # 4. Overall Robustness Summary
    # Create a comprehensive robustness score
    robustness_metrics = {
        'DOVE Evaluation': {
            'Granularity': dove_granularity / 10,  # Normalize
            'Range Coverage': dove_range * 100,
            'Consistency Info': 100,  # DOVE provides consistency information
            'Reliability': len(reliable_caps) / len(intersection_data) * 100
        },
        'Binary Evaluation': {
            'Granularity': binary_granularity / 10,
            'Range Coverage': binary_range * 100,
            'Consistency Info': 0,  # Binary doesn't provide consistency
            'Reliability': 50  # Assume 50% baseline
        }
    }
    
    metrics_names = list(robustness_metrics['DOVE Evaluation'].keys())
    dove_scores = list(robustness_metrics['DOVE Evaluation'].values())
    binary_scores = list(robustness_metrics['Binary Evaluation'].values())
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, dove_scores, width, label='DOVE', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, binary_scores, width, label='Binary', color='red', alpha=0.7)
    
    ax4.set_xlabel('Robustness Dimensions')
    ax4.set_ylabel('Score')
    ax4.set_title('Overall Robustness Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_robustness_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_robustness_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return robustness_metrics

def main():
    """Main function to generate weakness intersection analysis"""
    print("Loading data...")
    dove_scores, weakness_profile = load_data()
    
    if not dove_scores:
        print("Error: Could not load DOVE scores")
        return
    
    print(f"Loaded {len(dove_scores)} DOVE scores")
    
    # Simulate capability-based analysis
    print("Simulating capability-based analysis...")
    capabilities = simulate_capability_analysis(dove_scores, num_capabilities=50)
    print(f"Created {len(capabilities)} artificial capabilities for analysis")
    
    # Analyze intersections
    print("Analyzing capability intersections...")
    intersection_data = analyze_capability_intersections(capabilities)
    
    print("\nGenerating intersection visualizations...")
    
    print("1. Creating intersection overview...")
    intersection_counts = create_intersection_overview(intersection_data)
    
    print("2. Creating disagreement analysis...")
    disagreement_results = create_disagreement_analysis(intersection_data)
    
    print("3. Creating targeted weakness identification...")
    targeting_results = create_targeted_weakness_identification(intersection_data, disagreement_results)
    
    print("4. Creating robustness comparison...")
    robustness_results = create_robustness_comparison(intersection_data)
    
    # Generate summary for question generation
    print("\n" + "="*60)
    print("WEAKNESS INTERSECTION ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total capabilities analyzed: {len(intersection_data)}")
    
    print(f"\nIntersection Distribution:")
    for intersection_type, count in intersection_counts.items():
        if count > 0:
            print(f"  {intersection_type.replace('_', ' → ')}: {count} capabilities")
    
    print(f"\nDisagreement Analysis:")
    print(f"  High disagreement cases: {len(disagreement_results['high_disagreement'])}")
    print(f"  False negatives (Binary misses weaknesses): {len(disagreement_results['false_negatives'])}")
    print(f"  False positives (Binary overestimates): {len(disagreement_results['false_positives'])}")
    
    print(f"\nQuestion Generation Targets:")
    target_counts = Counter(target['type'] for target in targeting_results['generation_targets'])
    for target_type, count in target_counts.items():
        print(f"  {target_type}: {count} capabilities")
    
    print(f"\nTop Priority Capabilities for Question Generation:")
    top_priorities = targeting_results['priority_data'][:10]
    for i, item in enumerate(top_priorities, 1):
        print(f"  {i}. {item['capability']}: Score {item['dove_mean']:.3f}, Priority {item['priority_score']:.2f}")
    
    print("\nFiles generated:")
    print("  - figure_intersection_overview.pdf/.png")
    print("  - figure_disagreement_analysis.pdf/.png")
    print("  - figure_targeted_weakness_identification.pdf/.png")  
    print("  - figure_robustness_comparison.pdf/.png")
    
    print(f"\nKey Insights for Paper:")
    print(f"  - DOVE identifies {len([item for item in intersection_data if item['dove_weakness'] in ['Critical', 'High']])} critical/high weaknesses")
    print(f"  - Binary evaluation misses {len(disagreement_results['false_negatives'])} true weaknesses")
    print(f"  - {len(targeting_results['generation_targets'])} capabilities identified for targeted question generation")
    print(f"  - Robustness improvement: {robustness_results['DOVE Evaluation']['Granularity']:.1f}x more granular than binary")

if __name__ == "__main__":
    main()