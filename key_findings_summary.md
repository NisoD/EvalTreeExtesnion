# Key Findings Summary for Paper

## 🎯 **Main Research Question**
Can DOVE robustness scores improve EvalTree weakness profiling compared to traditional accuracy?

---

## 📊 **Dataset & Coverage**
- **MMLU Total**: 14,042 questions across hierarchical capability tree
- **DOVE Coverage**: 5,670 questions (40.4% coverage)
- **Model**: Llama 3.1 8B evaluated on both accuracy and robustness

---

## 🔍 **Key Finding 1: False Positive/Negative Analysis**

### **Classification Results (57 capabilities analyzed)**
- **True Positives** (both methods agree: weak): 8 capabilities (14.0%)
- **True Negatives** (both methods agree: strong): 34 capabilities (59.6%)
- **False Positives** (accuracy says weak, robustness says strong): 3 capabilities (5.3%)
- **False Negatives** (accuracy says strong, robustness says weak): 12 capabilities (21.1%)

### **Key Insight**
Traditional accuracy misses vulnerabilities 4x more often than it creates false alarms.

---

## 🔍 **Key Finding 2: Largest Accuracy-Robustness Gaps**

### **Most Underestimated Capabilities by Traditional Accuracy**
1. **Quantitative reasoning**: Accuracy 82.9% → Robustness 34.2% (Gap: 48.7%)
2. **Mathematical problem-solving**: Accuracy 77.3% → Robustness 36.2% (Gap: 41.1%)
3. **Advanced scientific mathematical**: Accuracy 70.9% → Robustness 30.8% (Gap: 40.1%)
4. **Mathematical reasoning principles**: Accuracy 74.1% → Robustness 36.5% (Gap: 37.5%)
5. **Statistical and probabilistic reasoning**: Accuracy 73.9% → Robustness 37.7% (Gap: 36.2%)

### **Pattern**
Mathematical and quantitative domains show the largest accuracy-robustness gaps.

---

## 🔍 **Key Finding 3: EvalTree Weakness Profiling Results**

### **DOVE-Enhanced EvalTree Weakness Profiles**
We successfully integrated DOVE robustness scores with EvalTree's hierarchical structure:

- **Coverage**: 5,670 questions with confidence intervals
- **Hierarchical Levels**: Multi-level tree structure preserved
- **Weakness Extraction**: At threshold 0.5, found 27 weakness nodes affecting 1,000 questions

### **Most Critical Weaknesses Identified**
1. **root_subtree_6_subtree_0_subtree_5_subtree_1**: Size 21, Mean 0.223, Upper bound 0.298
2. **root_subtree_7_subtree_0_subtree_0_subtree_3_subtree_1_subtree_1**: Size 24, Mean 0.242, Upper bound 0.307
3. **root_subtree_9_subtree_0_subtree_1**: Size 54, Mean 0.292, Upper bound 0.342

### **Weakness Profile Scaling**
- **Threshold 0.3**: 1 weakness (21 questions affected)
- **Threshold 0.4**: 12 weaknesses (485 questions affected)
- **Threshold 0.5**: 27 weaknesses (1,000 questions affected)
- **Threshold 0.6**: 39 weaknesses (1,414 questions affected)
- **Threshold 0.7**: 52 weaknesses (1,979 questions affected)

---

## 🔍 **Key Finding 4: Hierarchical Weakness Distribution**

### **DOVE-Based Weakness Profile (5,492 capability nodes)**
- **Critical Weaknesses** (< 0.30): 1,432 capabilities (26.1%)
- **High Weaknesses** (0.30-0.49): 829 capabilities (15.1%)
- **Moderate Weaknesses** (0.50-0.69): 711 capabilities (12.9%)
- **Strong Areas** (≥ 0.70): 2,520 capabilities (45.9%)

### **Most Critical Weaknesses** (Score < 0.01)
1. Analyzing theoretical frameworks (0.000)
2. Analyzing behavioral traits (0.000)
3. Applying Wien's Displacement Law (0.000)
4. Statistical reasoning application (0.001)

---

## 🔍 **Key Finding 5: Correlation Analysis**

### **Overall Correlation**
- **Robustness-Accuracy Correlation**: r = 0.396 (moderate positive)
- **Agreement Rate**: 73.7% of capabilities classified consistently
- **Major Disagreements**: 26.3% of capabilities show significant differences

---

## 💡 **Main Contributions**

### **What Existed Before**
- **DOVE**: Robustness evaluation through prompt perturbations
- **EvalTree**: Hierarchical weakness profiling using binary accuracy

### **Our Novel Contribution**
- **First combination** of DOVE robustness scores with EvalTree hierarchical structure
- **Successful integration** preserving hierarchical relationships with partial coverage (40%)
- **Comprehensive analysis** showing traditional accuracy misses 21.1% of vulnerabilities
- **EvalTree-compatible weakness profiles** using continuous robustness scores

---

## 📝 **Paper Story Arc**

1. **Problem**: Traditional accuracy evaluation misses model vulnerabilities
2. **Solution**: Combine DOVE robustness with EvalTree hierarchy  
3. **Method**: Integrate 5,670 DOVE scores into MMLU's 14,042-question tree structure
4. **Finding**: Traditional accuracy underestimates weakness in 21.1% of capabilities
5. **Impact**: Robustness-aware EvalTree provides better weakness detection

---

## 🎯 **Key Messages for Paper**

1. **Integration Success**: First successful combination of DOVE + EvalTree
2. **Traditional accuracy is incomplete** - misses vulnerabilities 4x more than creating false alarms
3. **Mathematical domains are most affected** - largest accuracy-robustness gaps
4. **Partial coverage is sufficient** - 40% coverage provides meaningful EvalTree profiles
5. **Hierarchical structure is preserved** - DOVE scores integrate seamlessly with EvalTree
6. **Practical impact** - EvalTree-compatible weakness profiles with continuous scores

---

## 📊 **Paper Figures**

### **Figure 1: Complete Confusion Matrix**
Shows the four classification categories (TP/TN/FP/FN) comparing traditional accuracy vs DOVE robustness

### **Figure 2: EvalTree Weakness Profile Scaling**
Shows how weakness profiles scale with different thresholds (0.3-0.7)

### **Figure 3: Accuracy-Robustness Gap Analysis**
Highlights mathematical domains with largest gaps between traditional accuracy and robustness 