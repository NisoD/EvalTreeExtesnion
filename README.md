# DOVE-Based Weakness Profiling for Language Models

## 🎯 **Research Focus**

This repository contains the analysis and findings for **DOVE-based weakness profiling** of language models, with a focus on **false positive mitigation** in traditional accuracy evaluation.

---

## 📁 **Repository Structure**

```
EvalAgain/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── research_paper.md           # Draft research paper
├── summaryCursor.md            # Complete analysis summary
├── 
├── analysis/                   # Core analysis scripts
│   ├── complete_false_positive_analysis.py      # Main analysis
│   ├── false_positive_mitigation_analysis.py    # FP mitigation
│   ├── robustness_correlation_analysis.py       # Correlation study
│   ├── generate_paper_figures.py                # Paper figures
│   ├── hierarchical_dove_profiler.py            # Hierarchical analysis
│   ├── mmlu_category_analysis.py                # MMLU categories
│   └── weakness_intersection_analysis.py        # Intersection study
├── 
├── data/                       # Core datasets
│   ├── MMLU.json              # MMLU EvalTree structure
│   ├── MMLU_DOVE.json         # DOVE robustness scores
│   └── stage1_output_example.json  # Example output
├── 
├── figures/                    # Key visualizations
│   ├── complete_confusion_matrix.pdf/.png       # False positive analysis
│   ├── error_comparison.pdf/.png                # Error type comparison
│   ├── quadrant_analysis.pdf/.png               # Capability classification
│   └── false_positive_detailed.pdf/.png         # Detailed FP analysis
├── 
├── docs/                       # Documentation
│   ├── dove_hierarchical_analysis_summary.md    # Key findings
│   └── weakness_intersection_summary.md         # Intersection analysis
├── 
└── utils/                      # Utility functions
    ├── api_inference.py        # API utilities
    ├── common.py              # Common functions
    └── compute_elo.py         # ELO computation
```

---

## 🔍 **Key Research Findings**

### **1. False Positive Mitigation Analysis**

**Main Result**: DOVE provides comprehensive false positive mitigation by correcting both over-penalization (5.3%) and under-penalization (21.1%) errors in traditional accuracy evaluation.

- **False Positives**: 3 capabilities (5.3%) - Traditional over-penalizes robust areas
- **False Negatives**: 12 capabilities (21.1%) - Traditional misses vulnerable areas  
- **Key Insight**: Traditional accuracy is 4x more likely to miss vulnerabilities than create false alarms

### **2. Hierarchical Weakness Profiling**

Successfully mapped 5,670 DOVE robustness scores to MMLU's hierarchical capability structure:

- **Critical Weaknesses** (< 0.30): 1,432 capabilities (26.1%)
- **High Weaknesses** (0.30-0.49): 829 capabilities (15.1%)
- **Moderate Weaknesses** (0.50-0.69): 711 capabilities (12.9%)
- **Strong Areas** (≥ 0.70): 2,520 capabilities (45.9%)

### **3. Robustness-Accuracy Correlation**

Moderate positive correlation (r = 0.396) between individual question robustness and category proficiency, with significant performance disparities across academic categories.

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Run Core Analysis**
```bash
# Complete false positive/negative analysis
python analysis/complete_false_positive_analysis.py

# False positive mitigation analysis
python analysis/false_positive_mitigation_analysis.py

# Robustness correlation analysis  
python analysis/robustness_correlation_analysis.py

# Generate paper figures
python analysis/generate_paper_figures.py
```

### **View Results**
- **Figures**: Check `figures/` directory for PDF/PNG visualizations
- **Analysis Summary**: Read `summaryCursor.md` for complete findings
- **Research Paper**: See `research_paper.md` for draft paper

---

## 📊 **Generated Visualizations**

1. **Complete Confusion Matrix** - Shows all four classification categories (TP/TN/FP/FN)
2. **Error Comparison** - Compares false positive vs false negative rates
3. **Quadrant Analysis** - Visualizes capabilities in accuracy-robustness space
4. **False Positive Detailed** - Detailed analysis of over-penalized capabilities

---

## 🎯 **Research Contributions**

### **Methodological Innovations**
- **Comprehensive Error Analysis**: Examines both false positives and false negatives
- **Hierarchical Integration**: Maps DOVE scores to EvalTree capability structure
- **Continuous Score Assessment**: Captures partial understanding missed by binary evaluation

### **Practical Implications**
- **Enhanced Weakness Detection**: Reveals vulnerabilities missed by traditional accuracy
- **Reduced False Alarms**: Prevents over-penalization of robust capabilities
- **Targeted Improvement**: Identifies specific areas needing attention

---

## 📝 **Citation**

```bibtex
@article{dove_weakness_profiling_2024,
  title={DOVE-Based Weakness Profiling: False Positive Mitigation in Language Model Evaluation},
  author={[Your Name]},
  year={2024},
  note={In preparation}
}
```

---

## 📞 **Contact**

For questions about this research or collaboration opportunities, please open an issue or contact the author.

---

**Last Updated**: December 2024
