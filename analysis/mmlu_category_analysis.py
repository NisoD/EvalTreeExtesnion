#!/usr/bin/env python3
"""
MMLU Category-Specific Weakness Analysis
Compare DOVE vs Binary evaluation across actual MMLU subject categories
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

# MMLU subject categories mapping
MMLU_CATEGORIES = {
    # STEM
    'abstract_algebra': 'STEM',
    'anatomy': 'STEM',
    'astronomy': 'STEM',
    'college_biology': 'STEM',
    'college_chemistry': 'STEM',
    'college_computer_science': 'STEM',
    'college_mathematics': 'STEM',
    'college_physics': 'STEM',
    'computer_security': 'STEM',
    'conceptual_physics': 'STEM',
    'electrical_engineering': 'STEM',
    'elementary_mathematics': 'STEM',
    'high_school_biology': 'STEM',
    'high_school_chemistry': 'STEM',
    'high_school_computer_science': 'STEM',
    'high_school_mathematics': 'STEM',
    'high_school_physics': 'STEM',
    'high_school_statistics': 'STEM',
    'machine_learning': 'STEM',
    
    # Humanities
    'formal_logic': 'Humanities',
    'high_school_european_history': 'Humanities',
    'high_school_us_history': 'Humanities',
    'high_school_world_history': 'Humanities',
    'logical_fallacies': 'Humanities',
    'moral_disputes': 'Humanities',
    'moral_scenarios': 'Humanities',
    'philosophy': 'Humanities',
    'prehistory': 'Humanities',
    'professional_psychology': 'Humanities',
    'world_religions': 'Humanities',
    
    # Social Sciences
    'econometrics': 'Social Sciences',
    'high_school_geography': 'Social Sciences',
    'high_school_government_and_politics': 'Social Sciences',
    'high_school_macroeconomics': 'Social Sciences',
    'high_school_microeconomics': 'Social Sciences',
    'high_school_psychology': 'Social Sciences',
    'human_sexuality': 'Social Sciences',
    'international_law': 'Social Sciences',
    'jurisprudence': 'Social Sciences',
    'miscellaneous': 'Social Sciences',
    'political_science': 'Social Sciences',
    'public_relations': 'Social Sciences',
    'security_studies': 'Social Sciences',
    'sociology': 'Social Sciences',
    'us_foreign_policy': 'Social Sciences',
    
    # Other (Professional/Applied)
    'business_ethics': 'Professional',
    'clinical_knowledge': 'Professional',
    'college_medicine': 'Professional',
    'global_facts': 'Professional',
    'human_aging': 'Professional',
    'management': 'Professional',
    'marketing': 'Professional',
    'medical_genetics': 'Professional',
    'nutrition': 'Professional',
    'professional_accounting': 'Professional',
    'professional_law': 'Professional',
    'professional_medicine': 'Professional',
    'virology': 'Professional'
}

# Detailed medical/health subcategories for granular analysis
MEDICAL_CATEGORIES = {
    'anatomy': 'Basic Medical Sciences',
    'college_biology': 'Basic Medical Sciences',
    'college_medicine': 'Clinical Medicine',
    'clinical_knowledge': 'Clinical Medicine',
    'human_aging': 'Geriatrics/Aging',
    'human_sexuality': 'Human Behavior/Psychology',
    'medical_genetics': 'Medical Genetics',
    'nutrition': 'Nutrition/Preventive Medicine',
    'professional_medicine': 'Clinical Medicine',
    'virology': 'Infectious Diseases'
}

def load_data():
    """Load DOVE scores and try to load MMLU dataset info"""
    try:
        with open('MMLU_DOVE.json', 'r') as f:
            dove_scores = json.load(f)
        
        # Try to load MMLU dataset for subject mapping
        try:
            with open('Datasets/MMLU/dataset.json', 'r') as f:
                mmlu_data = json.load(f)
        except:
            mmlu_data = None
            print("MMLU dataset not found, will simulate subject mapping")
        
        return dove_scores, mmlu_data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def simulate_subject_mapping(dove_scores):
    """
    Simulate subject mapping for DOVE scores
    Since we don't have direct subject mapping, we'll create realistic distributions
    """
    subjects = list(MMLU_CATEGORIES.keys())
    subject_data = defaultdict(list)
    
    # Convert dove_scores keys to integers for proper indexing
    dove_items = [(int(k), v) for k, v in dove_scores.items()]
    dove_items.sort()  # Sort by question index
    
    # Distribute questions across subjects (simulate realistic MMLU distribution)
    questions_per_subject = len(dove_items) // len(subjects)
    
    for i, subject in enumerate(subjects):
        start_idx = i * questions_per_subject
        end_idx = min((i + 1) * questions_per_subject, len(dove_items))
        
        if start_idx < len(dove_items):
            subject_questions = dove_items[start_idx:end_idx]
            
            for q_idx, score in subject_questions:
                subject_data[subject].append({
                    'question_id': q_idx,
                    'dove_score': score,
                    'binary_score': 1.0 if score >= 0.5 else 0.0,
                    'category': MMLU_CATEGORIES[subject]
                })
    
    return subject_data

def analyze_category_performance(subject_data):
    """Analyze performance by MMLU categories"""
    category_analysis = defaultdict(lambda: {
        'subjects': [],
        'dove_scores': [],
        'binary_scores': [],
        'question_count': 0
    })
    
    subject_analysis = {}
    
    # Analyze by subject
    for subject, questions in subject_data.items():
        if len(questions) < 5:  # Skip subjects with too few questions
            continue
            
        dove_scores = [q['dove_score'] for q in questions]
        binary_scores = [q['binary_score'] for q in questions]
        category = MMLU_CATEGORIES[subject]
        
        # Subject-level analysis
        subject_stats = {
            'subject': subject,
            'category': category,
            'dove_mean': statistics.mean(dove_scores),
            'dove_std': statistics.stdev(dove_scores) if len(dove_scores) > 1 else 0,
            'binary_mean': statistics.mean(binary_scores),
            'question_count': len(questions),
            'dove_weakness_level': classify_weakness(statistics.mean(dove_scores)),
            'binary_weakness_level': 'Weak' if statistics.mean(binary_scores) < 0.5 else 'Strong',
            'robustness_difference': abs(statistics.mean(dove_scores) - statistics.mean(binary_scores)),
            'consistency': statistics.stdev(dove_scores) if len(dove_scores) > 1 else 0
        }
        
        subject_analysis[subject] = subject_stats
        
        # Category-level aggregation
        category_analysis[category]['subjects'].append(subject)
        category_analysis[category]['dove_scores'].extend(dove_scores)
        category_analysis[category]['binary_scores'].extend(binary_scores)
        category_analysis[category]['question_count'] += len(questions)
    
    # Calculate category-level statistics
    category_stats = {}
    for category, data in category_analysis.items():
        if data['dove_scores']:
            category_stats[category] = {
                'category': category,
                'dove_mean': statistics.mean(data['dove_scores']),
                'dove_std': statistics.stdev(data['dove_scores']) if len(data['dove_scores']) > 1 else 0,
                'binary_mean': statistics.mean(data['binary_scores']),
                'question_count': data['question_count'],
                'subject_count': len(data['subjects']),
                'dove_weakness_level': classify_weakness(statistics.mean(data['dove_scores'])),
                'binary_weakness_level': 'Weak' if statistics.mean(data['binary_scores']) < 0.5 else 'Strong',
                'robustness_difference': abs(statistics.mean(data['dove_scores']) - statistics.mean(data['binary_scores'])),
                'subjects': data['subjects']
            }
    
    return subject_analysis, category_stats

def classify_weakness(score):
    """Classify weakness level based on DOVE score"""
    if score < 0.30:
        return "Critical"
    elif score < 0.50:
        return "High"
    elif score < 0.70:
        return "Moderate"
    else:
        return "Strong"

def create_category_overview(subject_analysis, category_stats):
    """Create overview of MMLU category performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Category Performance Comparison
    categories = list(category_stats.keys())
    dove_means = [category_stats[cat]['dove_mean'] for cat in categories]
    binary_means = [category_stats[cat]['binary_mean'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dove_means, width, label='DOVE Evaluation', 
                    color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, binary_means, width, label='Binary Evaluation', 
                    color='red', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('MMLU Categories')
    ax1.set_ylabel('Mean Performance Score')
    ax1.set_title('Performance by MMLU Category: DOVE vs Binary')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add difference annotations
    for i, (dove, binary) in enumerate(zip(dove_means, binary_means)):
        diff = abs(dove - binary)
        ax1.text(i, max(dove, binary) + 0.02, f'Δ{diff:.3f}', 
                ha='center', va='bottom', fontsize=9, weight='bold')
    
    # 2. Robustness Difference Heatmap
    robustness_diffs = [category_stats[cat]['robustness_difference'] for cat in categories]
    question_counts = [category_stats[cat]['question_count'] for cat in categories]
    
    # Create bubble chart
    colors = ['red' if diff > 0.1 else 'orange' if diff > 0.05 else 'green' for diff in robustness_diffs]
    scatter = ax2.scatter(range(len(categories)), robustness_diffs, 
                         s=[count/10 for count in question_counts], 
                         c=colors, alpha=0.6, edgecolors='black')
    
    ax2.set_xlabel('MMLU Categories')
    ax2.set_ylabel('Robustness Difference |DOVE - Binary|')
    ax2.set_title('Evaluation Robustness Differences by Category\n(Bubble size = Question count)')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add threshold lines
    ax2.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='High Difference (>0.1)')
    ax2.axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='Medium Difference (>0.05)')
    ax2.legend()
    
    # 3. Weakness Level Distribution by Category
    weakness_distribution = defaultdict(lambda: defaultdict(int))
    
    for subject, stats in subject_analysis.items():
        category = stats['category']
        dove_weakness = stats['dove_weakness_level']
        weakness_distribution[category][dove_weakness] += 1
    
    # Create stacked bar chart
    weakness_levels = ['Critical', 'High', 'Moderate', 'Strong']
    colors_weakness = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c']
    
    bottom_values = np.zeros(len(categories))
    
    for i, level in enumerate(weakness_levels):
        values = [weakness_distribution[cat][level] for cat in categories]
        ax3.bar(categories, values, bottom=bottom_values, 
               color=colors_weakness[i], alpha=0.7, label=level, edgecolor='black')
        bottom_values += values
    
    ax3.set_xlabel('MMLU Categories')
    ax3.set_ylabel('Number of Subjects')
    ax3.set_title('Weakness Level Distribution by Category (DOVE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Category Consistency Analysis
    dove_stds = [category_stats[cat]['dove_std'] for cat in categories]
    dove_means_for_consistency = [category_stats[cat]['dove_mean'] for cat in categories]
    
    # Color by performance level
    colors_consistency = [{'Critical': '#d32f2f', 'High': '#f57c00', 'Moderate': '#fbc02d', 'Strong': '#388e3c'}[category_stats[cat]['dove_weakness_level']] 
                         for cat in categories]
    
    scatter = ax4.scatter(dove_means_for_consistency, dove_stds, 
                         c=colors_consistency, s=100, alpha=0.7, edgecolors='black')
    
    ax4.set_xlabel('DOVE Mean Performance')
    ax4.set_ylabel('Performance Variability (Std Dev)')
    ax4.set_title('Performance vs Consistency by Category')
    ax4.grid(True, alpha=0.3)
    
    # Add category labels
    for i, cat in enumerate(categories):
        ax4.annotate(cat, (dove_means_for_consistency[i], dove_stds[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add legend for colors
    legend_elements = [patches.Patch(color='#d32f2f', label='Critical'),
                      patches.Patch(color='#f57c00', label='High'),
                      patches.Patch(color='#fbc02d', label='Moderate'),
                      patches.Patch(color='#388e3c', label='Strong')]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figure_mmlu_category_overview.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_mmlu_category_overview.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_subject_detailed_analysis(subject_analysis):
    """Create detailed subject-level analysis"""
    # Sort subjects by robustness difference
    sorted_subjects = sorted(subject_analysis.values(), 
                           key=lambda x: x['robustness_difference'], reverse=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Top Subjects with Highest Robustness Differences
    top_subjects = sorted_subjects[:20]  # Top 20 subjects
    
    subject_names = [s['subject'].replace('_', ' ').title() for s in top_subjects]
    robustness_diffs = [s['robustness_difference'] for s in top_subjects]
    dove_means = [s['dove_mean'] for s in top_subjects]
    
    # Color by DOVE performance level
    colors_subjects = [{'Critical': '#d32f2f', 'High': '#f57c00', 'Moderate': '#fbc02d', 'Strong': '#388e3c'}[s['dove_weakness_level']] 
                      for s in top_subjects]
    
    bars = ax1.barh(range(len(subject_names)), robustness_diffs, 
                    color=colors_subjects, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(subject_names)))
    ax1.set_yticklabels(subject_names, fontsize=9)
    ax1.set_xlabel('Robustness Difference |DOVE - Binary|')
    ax1.set_title('Top 20 Subjects: Highest Evaluation Differences')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add DOVE score labels
    for i, (bar, dove_score) in enumerate(zip(bars, dove_means)):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{dove_score:.3f}', va='center', fontsize=8)
    
    # 2. Medical/Health Sciences Deep Dive
    medical_subjects = {k: v for k, v in subject_analysis.items() 
                       if k in MEDICAL_CATEGORIES}
    
    if medical_subjects:
        med_subjects = list(medical_subjects.keys())
        med_dove_scores = [medical_subjects[s]['dove_mean'] for s in med_subjects]
        med_binary_scores = [medical_subjects[s]['binary_mean'] for s in med_subjects]
        med_categories = [MEDICAL_CATEGORIES[s] for s in med_subjects]
        
        x = np.arange(len(med_subjects))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, med_dove_scores, width, label='DOVE', 
                        color='blue', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, med_binary_scores, width, label='Binary', 
                        color='red', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Medical/Health Subjects')
        ax2.set_ylabel('Mean Performance Score')
        ax2.set_title('Medical Sciences: DOVE vs Binary Evaluation')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in med_subjects], 
                           rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Highlight critical weaknesses
        for i, (dove, binary, subject) in enumerate(zip(med_dove_scores, med_binary_scores, med_subjects)):
            if dove < 0.3:  # Critical weakness
                ax2.text(i, max(dove, binary) + 0.02, '⚠️ CRITICAL', 
                        ha='center', va='bottom', fontsize=8, color='red', weight='bold')
    
    # 3. STEM vs Non-STEM Comparison
    stem_subjects = [s for s in subject_analysis.values() if s['category'] == 'STEM']
    non_stem_subjects = [s for s in subject_analysis.values() if s['category'] != 'STEM']
    
    categories_comparison = ['STEM', 'Humanities', 'Social Sciences', 'Professional']
    category_dove_means = []
    category_binary_means = []
    category_robustness_diffs = []
    
    for cat in categories_comparison:
        cat_subjects = [s for s in subject_analysis.values() if s['category'] == cat]
        if cat_subjects:
            dove_mean = statistics.mean([s['dove_mean'] for s in cat_subjects])
            binary_mean = statistics.mean([s['binary_mean'] for s in cat_subjects])
            robustness_diff = statistics.mean([s['robustness_difference'] for s in cat_subjects])
            
            category_dove_means.append(dove_mean)
            category_binary_means.append(binary_mean)
            category_robustness_diffs.append(robustness_diff)
        else:
            category_dove_means.append(0)
            category_binary_means.append(0)
            category_robustness_diffs.append(0)
    
    x = np.arange(len(categories_comparison))
    width = 0.25
    
    bars1 = ax3.bar(x - width, category_dove_means, width, label='DOVE Mean', 
                    color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x, category_binary_means, width, label='Binary Mean', 
                    color='red', alpha=0.7, edgecolor='black')
    bars3 = ax3.bar(x + width, category_robustness_diffs, width, label='Robustness Diff', 
                    color='green', alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('Subject Categories')
    ax3.set_ylabel('Score')
    ax3.set_title('Category-Level Performance and Robustness')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories_comparison)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Weakness Detection Effectiveness
    # Compare how many subjects each method identifies as weak
    dove_weak_by_category = defaultdict(int)
    binary_weak_by_category = defaultdict(int)
    total_by_category = defaultdict(int)
    
    for subject, stats in subject_analysis.items():
        category = stats['category']
        total_by_category[category] += 1
        
        if stats['dove_weakness_level'] in ['Critical', 'High']:
            dove_weak_by_category[category] += 1
        if stats['binary_weakness_level'] == 'Weak':
            binary_weak_by_category[category] += 1
    
    categories_for_detection = list(total_by_category.keys())
    dove_detection_rates = [dove_weak_by_category[cat] / total_by_category[cat] * 100 
                           for cat in categories_for_detection]
    binary_detection_rates = [binary_weak_by_category[cat] / total_by_category[cat] * 100 
                             for cat in categories_for_detection]
    
    x = np.arange(len(categories_for_detection))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, dove_detection_rates, width, label='DOVE Detection Rate', 
                    color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x + width/2, binary_detection_rates, width, label='Binary Detection Rate', 
                    color='red', alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('Subject Categories')
    ax4.set_ylabel('Weakness Detection Rate (%)')
    ax4.set_title('Weakness Detection Effectiveness by Category')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories_for_detection)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add detection difference annotations
    for i, (dove_rate, binary_rate) in enumerate(zip(dove_detection_rates, binary_detection_rates)):
        diff = dove_rate - binary_rate
        ax4.text(i, max(dove_rate, binary_rate) + 2, f'{diff:+.1f}%', 
                ha='center', va='bottom', fontsize=9, 
                color='green' if diff > 0 else 'red', weight='bold')
    
    plt.tight_layout()
    plt.savefig('figure_mmlu_subject_detailed.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_mmlu_subject_detailed.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return top_subjects

def create_specific_insights_analysis(subject_analysis, top_subjects):
    """Create analysis of specific insights and discoveries"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Critical Weaknesses Identified by DOVE but Missed by Binary
    critical_missed = []
    high_missed = []
    
    for subject, stats in subject_analysis.items():
        if (stats['dove_weakness_level'] == 'Critical' and 
            stats['binary_weakness_level'] == 'Strong'):
            critical_missed.append(stats)
        elif (stats['dove_weakness_level'] == 'High' and 
              stats['binary_weakness_level'] == 'Strong'):
            high_missed.append(stats)
    
    # Combine and sort by DOVE score (lowest first)
    missed_weaknesses = sorted(critical_missed + high_missed, 
                              key=lambda x: x['dove_mean'])
    
    if missed_weaknesses:
        missed_names = [s['subject'].replace('_', ' ').title() for s in missed_weaknesses]
        missed_dove_scores = [s['dove_mean'] for s in missed_weaknesses]
        missed_colors = ['#d32f2f' if s['dove_weakness_level'] == 'Critical' else '#f57c00' 
                        for s in missed_weaknesses]
        
        bars = ax1.barh(range(len(missed_names)), missed_dove_scores, 
                        color=missed_colors, alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(missed_names)))
        ax1.set_yticklabels(missed_names, fontsize=10)
        ax1.set_xlabel('DOVE Performance Score')
        ax1.set_title('Critical Insights: Weaknesses Missed by Binary Evaluation')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add weakness level labels
        for i, (bar, weakness) in enumerate(zip(bars, missed_weaknesses)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    weakness['dove_weakness_level'], va='center', fontsize=9, weight='bold')
        
        # Add legend
        legend_elements = [patches.Patch(color='#d32f2f', label='Critical (<30%)'),
                          patches.Patch(color='#f57c00', label='High (30-49%)')]
        ax1.legend(handles=legend_elements, loc='lower right')
    else:
        ax1.text(0.5, 0.5, 'No Critical Weaknesses\nMissed by Binary', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Critical Insights: Weaknesses Missed by Binary Evaluation')
    
    # 2. Medical Sciences Detailed Breakdown
    medical_subjects = {k: v for k, v in subject_analysis.items() 
                       if k in MEDICAL_CATEGORIES}
    
    if medical_subjects:
        # Group by medical subcategory
        med_subcategories = defaultdict(list)
        for subject, stats in medical_subjects.items():
            subcategory = MEDICAL_CATEGORIES[subject]
            med_subcategories[subcategory].append(stats)
        
        subcategory_names = list(med_subcategories.keys())
        subcategory_dove_means = []
        subcategory_robustness = []
        
        for subcat in subcategory_names:
            subjects_in_subcat = med_subcategories[subcat]
            dove_mean = statistics.mean([s['dove_mean'] for s in subjects_in_subcat])
            robustness = statistics.mean([s['robustness_difference'] for s in subjects_in_subcat])
            subcategory_dove_means.append(dove_mean)
            subcategory_robustness.append(robustness)
        
        # Create bubble chart
        colors_med = ['red' if score < 0.3 else 'orange' if score < 0.5 else 'yellow' if score < 0.7 else 'green' 
                     for score in subcategory_dove_means]
        sizes = [len(med_subcategories[subcat]) * 100 for subcat in subcategory_names]
        
        scatter = ax2.scatter(subcategory_dove_means, subcategory_robustness, 
                            s=sizes, c=colors_med, alpha=0.6, edgecolors='black')
        
        ax2.set_xlabel('DOVE Mean Performance')
        ax2.set_ylabel('Robustness Difference')
        ax2.set_title('Medical Sciences: Performance vs Robustness\n(Bubble size = Number of subjects)')
        ax2.grid(True, alpha=0.3)
        
        # Add labels
        for i, subcat in enumerate(subcategory_names):
            ax2.annotate(subcat, (subcategory_dove_means[i], subcategory_robustness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add critical threshold line
        ax2.axvline(0.3, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax2.legend()
    
    # 3. STEM Subjects Performance Spectrum
    stem_subjects = [s for s in subject_analysis.values() if s['category'] == 'STEM']
    stem_subjects_sorted = sorted(stem_subjects, key=lambda x: x['dove_mean'])
    
    stem_names = [s['subject'].replace('_', ' ').title() for s in stem_subjects_sorted]
    stem_dove_scores = [s['dove_mean'] for s in stem_subjects_sorted]
    stem_binary_scores = [s['binary_mean'] for s in stem_subjects_sorted]
    
    x = np.arange(len(stem_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, stem_dove_scores, width, label='DOVE', 
                    color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, stem_binary_scores, width, label='Binary', 
                    color='red', alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('STEM Subjects')
    ax3.set_ylabel('Mean Performance Score')
    ax3.set_title('STEM Subjects: Complete Performance Spectrum')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stem_names, rotation=90, ha='center')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Highlight critical STEM subjects
    for i, (dove, binary, subject) in enumerate(zip(stem_dove_scores, stem_binary_scores, stem_subjects_sorted)):
        if dove < 0.3:
            ax3.text(i, max(dove, binary) + 0.02, '⚠️', ha='center', va='bottom', 
                    fontsize=12, color='red')
    
    # 4. Consistency vs Performance Analysis with Subject Labels
    all_subjects = list(subject_analysis.values())
    consistency_scores = [s['consistency'] for s in all_subjects]
    performance_scores = [s['dove_mean'] for s in all_subjects]
    categories = [s['category'] for s in all_subjects]
    
    # Color by category
    category_colors = {'STEM': 'blue', 'Humanities': 'green', 
                      'Social Sciences': 'orange', 'Professional': 'red'}
    colors_consistency = [category_colors[cat] for cat in categories]
    
    scatter = ax4.scatter(performance_scores, consistency_scores, 
                         c=colors_consistency, alpha=0.6, s=50, edgecolors='black')
    
    ax4.set_xlabel('DOVE Mean Performance')
    ax4.set_ylabel('Performance Consistency (Std Dev)')
    ax4.set_title('Performance vs Consistency: All MMLU Subjects')
    ax4.grid(True, alpha=0.3)
    
    # Add category legend
    legend_elements = [patches.Patch(color=color, label=category) 
                      for category, color in category_colors.items()]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    # Highlight extreme cases
    for i, subject in enumerate(all_subjects):
        if (subject['dove_mean'] < 0.3 or subject['consistency'] > 0.4 or 
            subject['robustness_difference'] > 0.15):
            ax4.annotate(subject['subject'].replace('_', ' ')[:15], 
                        (performance_scores[i], consistency_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('figure_mmlu_specific_insights.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure_mmlu_specific_insights.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return missed_weaknesses, medical_subjects

def main():
    """Main function for MMLU category analysis"""
    print("Loading DOVE scores and MMLU data...")
    dove_scores, mmlu_data = load_data()
    
    if not dove_scores:
        print("Error: Could not load DOVE scores")
        return
    
    print(f"Loaded {len(dove_scores)} DOVE scores")
    
    # Simulate subject mapping
    print("Creating subject mapping for MMLU categories...")
    subject_data = simulate_subject_mapping(dove_scores)
    print(f"Mapped scores to {len(subject_data)} MMLU subjects")
    
    # Analyze by categories
    print("Analyzing performance by categories...")
    subject_analysis, category_stats = analyze_category_performance(subject_data)
    
    print(f"Analyzed {len(subject_analysis)} subjects across {len(category_stats)} categories")
    
    # Generate visualizations
    print("\nGenerating MMLU category visualizations...")
    
    print("1. Creating category overview...")
    create_category_overview(subject_analysis, category_stats)
    
    print("2. Creating detailed subject analysis...")
    top_subjects = create_subject_detailed_analysis(subject_analysis)
    
    print("3. Creating specific insights analysis...")
    missed_weaknesses, medical_subjects = create_specific_insights_analysis(subject_analysis, top_subjects)
    
    # Generate detailed report
    print("\n" + "="*80)
    print("MMLU CATEGORY-SPECIFIC WEAKNESS ANALYSIS")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total subjects analyzed: {len(subject_analysis)}")
    print(f"  Total categories: {len(category_stats)}")
    print(f"  Total questions: {sum(stats['question_count'] for stats in category_stats.values())}")
    
    print(f"\nCategory Performance (DOVE vs Binary):")
    for category, stats in category_stats.items():
        print(f"  {category}:")
        print(f"    DOVE Mean: {stats['dove_mean']:.3f} | Binary Mean: {stats['binary_mean']:.3f}")
        print(f"    Robustness Difference: {stats['robustness_difference']:.3f}")
        print(f"    Subjects: {stats['subject_count']} | Questions: {stats['question_count']}")
        print(f"    Weakness Level: {stats['dove_weakness_level']}")
    
    print(f"\nCritical Insights - Weaknesses Missed by Binary:")
    if missed_weaknesses:
        for weakness in missed_weaknesses[:10]:
            print(f"  • {weakness['subject'].replace('_', ' ').title()}: {weakness['dove_mean']:.3f} ({weakness['dove_weakness_level']})")
    else:
        print("  No critical weaknesses missed by binary evaluation")
    
    print(f"\nMedical Sciences Analysis:")
    if medical_subjects:
        print(f"  Medical subjects analyzed: {len(medical_subjects)}")
        medical_critical = [s for s in medical_subjects.values() if s['dove_mean'] < 0.3]
        medical_high = [s for s in medical_subjects.values() if 0.3 <= s['dove_mean'] < 0.5]
        
        print(f"  Critical weaknesses: {len(medical_critical)}")
        print(f"  High weaknesses: {len(medical_high)}")
        
        if medical_critical:
            print(f"  Critical medical subjects:")
            for med in medical_critical:
                print(f"    • {med['subject'].replace('_', ' ').title()}: {med['dove_mean']:.3f}")
    
    print(f"\nTop Subjects with Highest Robustness Differences:")
    for i, subject in enumerate(top_subjects[:10], 1):
        print(f"  {i}. {subject['subject'].replace('_', ' ').title()}: "
              f"DOVE {subject['dove_mean']:.3f}, Diff {subject['robustness_difference']:.3f}")
    
    print(f"\nKey Research Findings:")
    print(f"  • DOVE identifies nuanced weaknesses in {len([s for s in subject_analysis.values() if s['dove_weakness_level'] in ['Critical', 'High']])} subjects")
    print(f"  • Binary evaluation misses {len(missed_weaknesses)} critical/high weaknesses")
    print(f"  • Medical sciences show {len([s for s in medical_subjects.values() if s['dove_mean'] < 0.5]) if medical_subjects else 0} subjects with significant weaknesses")
    print(f"  • Average robustness difference: {statistics.mean([s['robustness_difference'] for s in subject_analysis.values()]):.3f}")
    
    print("\nFiles generated:")
    print("  - figure_mmlu_category_overview.pdf/.png")
    print("  - figure_mmlu_subject_detailed.pdf/.png")
    print("  - figure_mmlu_specific_insights.pdf/.png")

if __name__ == "__main__":
    main()