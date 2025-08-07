#!/usr/bin/env python3
"""
Advanced Weakness Profile Insights

Discovers innovative patterns across multiple thresholds and explores
the relationship between accuracy, robustness, and capability hierarchies.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter
import statistics
from scipy import stats
import pandas as pd

class AdvancedWeaknessAnalyzer:
    """Advanced analyzer for multi-threshold weakness profiles."""
    
    def __init__(self, tree_path, model_name="Llama-3.1-8B-Instruct"):
        self.tree_path = tree_path
        self.model_name = model_name
        self.profiles = {}
        self.tree_data = None
        
        # Load tree data
        with open(tree_path, 'r') as f:
            self.tree_data = json.load(f)
    
    def load_profile(self, threshold, profile_path):
        """Load a weakness profile for a specific threshold."""
        with open(profile_path, 'r') as f:
            self.profiles[threshold] = json.load(f)
    
    def analyze_threshold_evolution(self):
        """Analyze how weaknesses evolve across thresholds."""
        print("=== THRESHOLD EVOLUTION ANALYSIS ===\n")
        
        evolution_data = []
        
        for threshold in sorted(self.profiles.keys()):
            profile = self.profiles[threshold]
            
            for i, weakness in enumerate(profile['weakness_nodes']):
                evolution_data.append({
                    'threshold': threshold,
                    'weakness_id': i,
                    'capability': weakness['capability'],
                    'size': weakness['size'],
                    'accuracy': weakness['accuracy'],
                    'avg_dove': statistics.mean(weakness['dove_scores']),
                    'subjects': weakness['subjects'],
                    'num_subjects': len(weakness['subjects'])
                })
        
        # Create evolution DataFrame
        df = pd.DataFrame(evolution_data)
        
        print("Threshold Evolution Summary:")
        threshold_summary = df.groupby('threshold').agg({
            'size': ['sum', 'mean', 'count'],
            'accuracy': 'mean',
            'avg_dove': 'mean',
            'num_subjects': 'mean'
        }).round(3)
        print(threshold_summary)
        
        return df
    
    def discover_capability_hierarchies(self):
        """Discover hierarchical relationships between capabilities."""
        print("\n=== CAPABILITY HIERARCHY DISCOVERY ===\n")
        
        # Extract all capabilities across thresholds
        all_capabilities = {}
        
        for threshold, profile in self.profiles.items():
            for weakness in profile['weakness_nodes']:
                cap = weakness['capability']
                if cap not in all_capabilities:
                    all_capabilities[cap] = {
                        'thresholds': [],
                        'sizes': [],
                        'accuracies': [],
                        'dove_scores': [],
                        'subjects_sets': []
                    }
                
                all_capabilities[cap]['thresholds'].append(threshold)
                all_capabilities[cap]['sizes'].append(weakness['size'])
                all_capabilities[cap]['accuracies'].append(weakness['accuracy'])
                all_capabilities[cap]['dove_scores'].append(statistics.mean(weakness['dove_scores']))
                all_capabilities[cap]['subjects_sets'].append(set(weakness['subjects']))
        
        # Analyze capability relationships
        print("Capability Stability Analysis:")
        stable_caps = []
        emerging_caps = []
        
        for cap, data in all_capabilities.items():
            if len(data['thresholds']) == len(self.profiles):
                stable_caps.append(cap)
                print(f"STABLE: {cap[:60]}...")
                print(f"  Appears at all thresholds: {data['thresholds']}")
                print(f"  Size evolution: {data['sizes']}")
                print(f"  Accuracy evolution: {[round(a, 3) for a in data['accuracies']]}")
            else:
                emerging_caps.append(cap)
                print(f"EMERGING: {cap[:60]}...")
                print(f"  Appears at thresholds: {data['thresholds']}")
                print(f"  Size: {data['sizes']}")
                print(f"  Accuracy: {[round(a, 3) for a in data['accuracies']]}")
        
        return all_capabilities, stable_caps, emerging_caps
    
    def analyze_dove_accuracy_decoupling(self):
        """Find interesting cases where DOVE and accuracy decouple."""
        print("\n=== DOVE-ACCURACY DECOUPLING ANALYSIS ===\n")
        
        all_questions = []
        
        # Extract all questions across all thresholds
        for threshold, profile in self.profiles.items():
            for weakness in profile['weakness_nodes']:
                for q in weakness['node_data']['subtrees'] if isinstance(weakness['node_data'].get('subtrees'), list) else []:
                    if isinstance(q.get('subtrees'), (int, type(None))) and 'dove_score' in q:
                        accuracy = None
                        if 'ranking' in q:
                            for model_data in q['ranking']:
                                if model_data[0] == self.model_name:
                                    accuracy = model_data[1]
                                    break
                        
                        if accuracy is not None:
                            all_questions.append({
                                'threshold': threshold,
                                'accuracy': accuracy,
                                'dove_score': q['dove_score'],
                                'subject': q.get('subject', 'unknown'),
                                'question': q.get('input', '')[:100] + '...',
                                'capability': weakness['capability']
                            })
        
        df = pd.DataFrame(all_questions)
        
        # Find decoupling patterns
        print("Decoupling Pattern Discovery:")
        
        # High DOVE, Low Accuracy (Robust but Hard)
        robust_hard = df[(df['dove_score'] > 0.6) & (df['accuracy'] < 0.5)]
        print(f"\nROBUST BUT HARD questions: {len(robust_hard)}")
        if len(robust_hard) > 0:
            print("Top subjects:", robust_hard['subject'].value_counts().head())
            print("Example:", robust_hard.iloc[0]['question'] if len(robust_hard) > 0 else "None")
        
        # Low DOVE, High Accuracy (Accurate but Brittle)  
        accurate_brittle = df[(df['dove_score'] < 0.3) & (df['accuracy'] > 0.5)]
        print(f"\nACCURATE BUT BRITTLE questions: {len(accurate_brittle)}")
        if len(accurate_brittle) > 0:
            print("Top subjects:", accurate_brittle['subject'].value_counts().head())
            print("Example:", accurate_brittle.iloc[0]['question'] if len(accurate_brittle) > 0 else "None")
        
        # Extreme decoupling cases
        df['decoupling_score'] = abs(df['dove_score'] - df['accuracy'])
        extreme_decoupling = df.nlargest(10, 'decoupling_score')
        
        print(f"\nEXTREME DECOUPLING cases (top 10):")
        for idx, row in extreme_decoupling.iterrows():
            print(f"  Acc: {row['accuracy']:.3f}, DOVE: {row['dove_score']:.3f}, "
                  f"Gap: {row['decoupling_score']:.3f}, Subject: {row['subject']}")
        
        return df, robust_hard, accurate_brittle, extreme_decoupling
    
    def analyze_subject_vulnerability_progression(self):
        """Analyze how subjects become vulnerable as thresholds increase."""
        print("\n=== SUBJECT VULNERABILITY PROGRESSION ===\n")
        
        subject_threshold_map = defaultdict(list)
        
        for threshold, profile in self.profiles.items():
            affected_subjects = set()
            for weakness in profile['weakness_nodes']:
                affected_subjects.update(weakness['subjects'])
            
            for subject in affected_subjects:
                subject_threshold_map[subject].append(threshold)
        
        # Categorize subjects by vulnerability pattern
        always_vulnerable = []
        emerging_vulnerable = []
        threshold_specific = defaultdict(list)
        
        all_thresholds = set(self.profiles.keys())
        
        for subject, thresholds in subject_threshold_map.items():
            threshold_set = set(thresholds)
            
            if threshold_set == all_thresholds:
                always_vulnerable.append(subject)
            elif len(threshold_set) == 1:
                threshold_specific[list(threshold_set)[0]].append(subject)
            else:
                emerging_vulnerable.append((subject, sorted(thresholds)))
        
        print("Subject Vulnerability Categories:")
        print(f"\nALWAYS VULNERABLE (appear at all thresholds): {len(always_vulnerable)}")
        for subject in always_vulnerable:
            print(f"  - {subject}")
        
        print(f"\nEMERGING VULNERABLE (appear at multiple thresholds): {len(emerging_vulnerable)}")
        for subject, thresholds in emerging_vulnerable:
            print(f"  - {subject}: {thresholds}")
        
        print(f"\nTHRESHOLD-SPECIFIC VULNERABILITIES:")
        for threshold in sorted(threshold_specific.keys()):
            subjects = threshold_specific[threshold]
            print(f"  Only at threshold {threshold}: {subjects}")
        
        return subject_threshold_map, always_vulnerable, emerging_vulnerable, threshold_specific
    
    def create_advanced_visualizations(self, output_prefix="advanced_analysis"):
        """Create advanced visualizations of the insights."""
        
        # 1. Threshold Evolution Heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Weakness Profile Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for heatmap
        threshold_data = []
        capability_names = []
        
        for threshold, profile in self.profiles.items():
            row_data = []
            for weakness in profile['weakness_nodes']:
                if weakness['capability'] not in capability_names:
                    capability_names.append(weakness['capability'][:40] + '...')
                row_data.append(weakness['accuracy'])
            threshold_data.append(row_data)
        
        # Pad rows to same length
        max_len = max(len(row) for row in threshold_data)
        for row in threshold_data:
            while len(row) < max_len:
                row.append(np.nan)
        
        # Create heatmap
        ax1 = axes[0, 0]
        heatmap_data = np.array(threshold_data)
        im = ax1.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax1.set_title('Accuracy Across Thresholds')
        ax1.set_ylabel('Threshold')
        ax1.set_xlabel('Weakness Index')
        ax1.set_yticks(range(len(self.profiles)))
        ax1.set_yticklabels([f"T={t}" for t in sorted(self.profiles.keys())])
        plt.colorbar(im, ax=ax1)
        
        # 2. Subject Vulnerability Network
        ax2 = axes[0, 1]
        
        # Count subject appearances across thresholds
        subject_counts = Counter()
        for threshold, profile in self.profiles.items():
            for weakness in profile['weakness_nodes']:
                for subject in weakness['subjects']:
                    subject_counts[subject] += 1
        
        subjects = list(subject_counts.keys())[:15]  # Top 15
        counts = [subject_counts[s] for s in subjects]
        
        ax2.barh(subjects, counts, color='skyblue')
        ax2.set_title('Subject Vulnerability Frequency')
        ax2.set_xlabel('Number of Threshold Appearances')
        
        # 3. DOVE vs Accuracy Scatter with Threshold Coloring
        ax3 = axes[1, 0]
        
        colors = ['red', 'blue', 'green']
        for i, (threshold, profile) in enumerate(sorted(self.profiles.items())):
            all_acc = []
            all_dove = []
            
            for weakness in profile['weakness_nodes']:
                # Extract individual questions
                def extract_questions(node):
                    questions = []
                    if isinstance(node.get('subtrees'), (int, type(None))) and 'dove_score' in node:
                        accuracy = None
                        if 'ranking' in node:
                            for model_data in node['ranking']:
                                if model_data[0] == self.model_name:
                                    accuracy = model_data[1]
                                    break
                        if accuracy is not None:
                            questions.append((accuracy, node['dove_score']))
                    elif isinstance(node.get('subtrees'), list):
                        for child in node['subtrees']:
                            questions.extend(extract_questions(child))
                    return questions
                
                questions = extract_questions(weakness['node_data'])
                for acc, dove in questions:
                    all_acc.append(acc)
                    all_dove.append(dove)
            
            ax3.scatter(all_acc, all_dove, alpha=0.6, c=colors[i], 
                       label=f'Threshold {threshold}', s=30)
        
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('DOVE Score')
        ax3.set_title('Weakness Questions by Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Capability Size Evolution
        ax4 = axes[1, 1]
        
        # Track how weakness sizes change
        capability_evolution = defaultdict(list)
        
        for threshold in sorted(self.profiles.keys()):
            profile = self.profiles[threshold]
            threshold_sizes = []
            for weakness in profile['weakness_nodes']:
                threshold_sizes.append(weakness['size'])
            capability_evolution['sizes'].append(threshold_sizes)
            capability_evolution['thresholds'].append(threshold)
        
        # Plot total questions affected
        total_questions = [sum(sizes) for sizes in capability_evolution['sizes']]
        ax4.plot(capability_evolution['thresholds'], total_questions, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Total Questions in Weaknesses')
        ax4.set_title('Weakness Scope Evolution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"figures/{output_prefix}_comprehensive.png", dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to figures/{output_prefix}_comprehensive.png")
        
        return fig

def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python advanced_weakness_insights.py <tree_path> <profile1_path> <profile2_path> [profile3_path]")
        sys.exit(1)
    
    tree_path = sys.argv[1]
    profile_paths = sys.argv[2:]
    
    # Initialize analyzer
    analyzer = AdvancedWeaknessAnalyzer(tree_path)
    
    # Load profiles (infer thresholds from filenames)
    for profile_path in profile_paths:
        if 't05' in profile_path:
            threshold = 0.5
        elif 't07' in profile_path:
            threshold = 0.7
        elif 't06' in profile_path:
            threshold = 0.6
        else:
            threshold = 0.6  # default
        
        analyzer.load_profile(threshold, profile_path)
    
    print(f"Loaded {len(analyzer.profiles)} threshold profiles: {sorted(analyzer.profiles.keys())}")
    
    # Run advanced analyses
    print("\n" + "="*80)
    
    # 1. Threshold Evolution
    evolution_df = analyzer.analyze_threshold_evolution()
    
    # 2. Capability Hierarchies  
    capabilities, stable, emerging = analyzer.discover_capability_hierarchies()
    
    # 3. DOVE-Accuracy Decoupling
    decoupling_df, robust_hard, accurate_brittle, extreme = analyzer.analyze_dove_accuracy_decoupling()
    
    # 4. Subject Vulnerability
    subject_map, always_vuln, emerging_vuln, threshold_specific = analyzer.analyze_subject_vulnerability_progression()
    
    # 5. Create visualizations
    analyzer.create_advanced_visualizations()
    
    print("\n" + "="*80)
    print("ADVANCED INSIGHTS SUMMARY:")
    print(f"- Found {len(stable)} stable capabilities across all thresholds")
    print(f"- Found {len(emerging)} emerging capabilities at higher thresholds") 
    print(f"- Identified {len(robust_hard)} robust-but-hard questions")
    print(f"- Identified {len(accurate_brittle)} accurate-but-brittle questions")
    print(f"- Found {len(always_vuln)} always-vulnerable subjects")
    print(f"- Discovered threshold-specific vulnerabilities: {len(threshold_specific)} categories")

if __name__ == "__main__":
    main() 