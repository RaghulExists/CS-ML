# Module 5: Instance-Based Learning - Complete Exam Study Guide

> **For 3rd Year BE Students | VTU/Karnataka Curriculum**
> **Topic:** k-NN, Locally Weighted Regression, Case-Based Reasoning, RBF Networks

---

## How This Guide is Organized

I have grouped the 20 questions into **5 logical sections**. Once you understand one question in a section, the others become easy because they use the **same core idea** with different details.

| Section | Topic | Questions |
|---------|-------|-----------|
| **A** | Foundations of Instance-Based Learning | Q1, Q4, Q6, Q10 |
| **B** | k-Nearest Neighbor (Theory) | Q7, Q8, Q9, Q12, Q13, Q14, Q15 |
| **C** | k-NN Numericals | Q17, Q20 |
| **D** | Locally Weighted Regression (LWR) | Q16, Q18, Q19 |
| **E** | RBF Networks & Case-Based Reasoning | Q2, Q3, Q5, Q11 |

---

# THE MASTER FORMULA SHEET (Memorize These!)

You only need these 5 formulas to solve this entire module:

### 1. Euclidean Distance (used everywhere)
$$d(x_q, x_i) = \sqrt{\sum_{r=1}^{n} (a_r(x_q) - a_r(x_i))^2}$$

For 2 attributes: $d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$

### 2. k-NN for Discrete (Classification) — Majority Vote
$$\hat{f}(x_q) = \arg\max_{v \in V} \sum_{i=1}^{k} \delta(v, f(x_i))$$
*(Pick the class that appears most among k nearest neighbors)*

### 3. k-NN for Continuous (Regression) — Average
$$\hat{f}(x_q) = \frac{1}{k} \sum_{i=1}^{k} f(x_i)$$
*(Average target value of k nearest neighbors)*

### 4. Distance-Weighted k-NN — Closer neighbors count more
$$w_i = \frac{1}{d(x_q, x_i)^2}$$

For discrete: $\hat{f}(x_q) = \arg\max_v \sum_{i=1}^{k} w_i \cdot \delta(v, f(x_i))$

For continuous: $\hat{f}(x_q) = \frac{\sum_{i=1}^{k} w_i \cdot f(x_i)}{\sum_{i=1}^{k} w_i}$

### 5. Gaussian Kernel (for LWR & RBF)
$$K(d) = e^{-\frac{d^2}{2\sigma^2}} \quad \text{or equivalently} \quad e^{-\frac{d^2}{2\tau^2}}$$

Where σ (or τ) is the bandwidth.

---

# SECTION A: FOUNDATIONS OF INSTANCE-BASED LEARNING

## Q1. Explain Instance-Based Learning in Detail with an Example. (5 Marks)

### Definition:

**Instance-Based Learning (IBL)** is a family of learning algorithms that, instead of building an explicit model during training, simply **store the training examples** and delay processing until a new instance must be classified.

It is also called **Memory-Based Learning** or **Lazy Learning**.

### Key Characteristics:

1. **No explicit model is constructed** during training.
2. **All work is done at query time** (when classifying a new instance).
3. **Local approximation** — each new query is approximated based on nearby training examples.
4. **Different target functions** can be approximated for different queries (no global model).

### How It Works:

| Phase | What Happens |
|-------|--------------|
| **Training Phase** | Just store all training examples (no learning happens) |
| **Testing Phase** | When a new query arrives, find similar stored instances and use them to predict |

### Examples of Instance-Based Learning Algorithms:

1. **k-Nearest Neighbor (k-NN)**
2. **Locally Weighted Regression (LWR)**
3. **Radial Basis Function (RBF) Networks**
4. **Case-Based Reasoning (CBR)**

### Concrete Example: Movie Recommendation

Suppose you've watched 100 movies and rated them. Now, you ask "Will I like this new movie X?"

**Instance-based approach:**
1. Find 5 movies in your history most similar to X (same genre, director, actors).
2. Average those 5 ratings.
3. Predict that as your rating for X.

There's no "model of your taste" — just a comparison with stored examples.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Collect training examples ⟨x_i, f(x_i)⟩ |
| 2 | Store them in memory (no processing) |
| 3 | When query x_q arrives, find similar instances using distance |
| 4 | Use those similar instances to predict f(x_q) |
| 5 | Discard the local approximation; next query starts fresh |

### Why It's Useful:

- Works well when target function is **complex** but **locally simple**.
- Easy to add new training data (just append).
- Can model very flexible boundaries.

---

## Q4. Explain the Key Difference between Lazy Learner and Eager Learner. (5 Marks)

### Definitions:

**Lazy Learner:** Defers all generalization until a query is made. Stores training data and processes it only when needed.

**Eager Learner:** Performs generalization (builds a model) during training, before any queries arrive.

### Comparison Table:

| Aspect | Lazy Learner | Eager Learner |
|--------|-------------|---------------|
| **When learning happens** | At query time | During training |
| **What is stored** | All training examples | A compact model (rules, weights) |
| **Training time** | Almost zero (just storage) | High (model construction) |
| **Query/Prediction time** | High (search through data) | Low (just apply model) |
| **Memory required** | High (stores all data) | Low (stores only model) |
| **Approximation** | Local (different for each query) | Global (one model fits all) |
| **Adapts to new data** | Easily (just add to memory) | Needs retraining |
| **Examples** | k-NN, LWR, CBR | Decision Trees, Neural Networks, Naive Bayes, SVM |

### Visual Difference:

```
EAGER LEARNER:
Training data → [Build Model] → Model
                                  ↓
              New query → Apply model → Prediction

LAZY LEARNER:
Training data → [Just Store]
                     ↓
   New query → Search through stored data → Prediction
```

### Example:

**Lazy (k-NN):** When you ask "Is this email spam?", it searches through all 10,000 emails it has stored, finds the 5 most similar, and votes.

**Eager (Naive Bayes):** During training, it computes P(spam), P(word | spam) for all words. When you ask, it just multiplies probabilities — no searching.

### Step-by-Step:

| Step | Lazy Learner | Eager Learner |
|------|--------------|---------------|
| 1 | Store training examples | Process training examples |
| 2 | Wait for query | Build model (rules/weights) |
| 3 | When query comes, find similar examples | Discard training data |
| 4 | Compute prediction locally | When query comes, apply model |

---

## Q6. Describe the Advantages and Disadvantages of Instance-Based Learning. (5 Marks)

### Advantages:

1. **Simple to Implement:** No complex model building — just store and search.

2. **Naturally Handles Complex Target Functions:** Can model very intricate decision boundaries because it uses local approximations.

3. **Easy to Add New Data:** Just append new examples to memory; no retraining needed.

4. **Different Approximations for Different Queries:** Each query gets its own local model — more flexible.

5. **No Information Loss:** Since all training data is stored, no information is discarded.

6. **Works with Any Type of Distance Metric:** Euclidean, Manhattan, Hamming, custom — flexibility.

### Disadvantages:

1. **High Computational Cost at Query Time:** Must search through ALL training examples for each new query → slow for large datasets.

2. **High Memory Requirement:** Stores ALL training examples → memory expensive.

3. **Sensitive to Irrelevant Attributes (Curse of Dimensionality):** Distance gets distorted when there are many irrelevant features. All points become "equally far".

4. **Sensitive to Noisy Data:** A single noisy neighbor can mislead the prediction.

5. **No Generalization Until Query:** Cannot give insights or rules about the data.

6. **Sensitive to Choice of Distance Metric:** Wrong metric = wrong predictions.

### Step-by-Step Summary:

| Aspect | Pros | Cons |
|--------|------|------|
| Training | Fast (just store) | Needs lots of memory |
| Querying | Flexible local approx. | Slow (search all data) |
| Data quality | No info loss | Sensitive to noise/irrelevant attrs |
| Adaptation | Easy to add data | Curse of dimensionality |

### How to Mitigate Disadvantages:

- **Use kd-trees** to speed up neighbor search.
- **Feature weighting** to handle irrelevant attributes.
- **Distance-weighted voting** to reduce noise impact.
- **Normalize attributes** so all have equal scale.

---

## Q10. Describe Remarks on Lazy and Eager Learning with Suitable Example. (6 Marks)

### Key Remarks (Important Theoretical Points):

#### Remark 1: Different Hypothesis Spaces

- **Eager learners** must commit to ONE global hypothesis that covers the entire instance space.
- **Lazy learners** can use a **richer hypothesis space** because they construct many local approximations — one for each query.

This means lazy learners can represent more complex target functions implicitly.

#### Remark 2: Computation Trade-off

- **Eager learner:** Spends time training (high) but predictions are fast (low).
- **Lazy learner:** Training is fast (just storage) but predictions are slow (search).

#### Remark 3: Inductive Bias

- **Eager learners:** Have a fixed inductive bias (e.g., decision trees prefer short trees).
- **Lazy learners:** The bias depends on the local region of the query point.

#### Remark 4: Approximation Quality

- **Eager** must approximate target globally — may be poor in some regions.
- **Lazy** approximates locally — usually better near the query.

### Concrete Example: Predicting House Prices

**Setup:** Training data has 1000 houses with features (area, bedrooms, location) and prices.

**Eager Approach (Linear Regression):**
- During training: Fit ONE line/equation: Price = w₁·area + w₂·bedrooms + ...
- Prediction: Just plug in features.
- Problem: A single global formula may not fit all neighborhoods well.

**Lazy Approach (k-NN):**
- During training: Just store the 1000 houses.
- Prediction: For new house, find 5 most similar houses, average their prices.
- Advantage: Captures local price patterns (luxury area vs budget area).

### Visualization:

```
Eager: Fits ONE model that may be slightly wrong everywhere
   *  *
      *      ─── (one straight line for everything)
  *      *
*           *

Lazy: Builds DIFFERENT models for different regions
   *  *  ─── (line near these points)
      *
  *      *  ─── (different line near these points)
*           *
```

### Step-by-Step Comparison:

| Step | Lazy Approach | Eager Approach |
|------|--------------|----------------|
| 1 | Store all 1000 houses | Compute weights for global formula |
| 2 | Receive query for new house | Receive query for new house |
| 3 | Find 5 nearest houses | Apply formula |
| 4 | Average their prices | Output prediction |
| 5 | Output prediction | Done |

### Conclusion:

**Lazy learners** are flexible but slow at query time.  
**Eager learners** are committed but fast at query time.  
The choice depends on whether you have more time for **training** or **prediction**.

---

# SECTION B: k-NEAREST NEIGHBOR (THEORY)

> **All k-NN questions are essentially asking the same algorithm with different framings.**
> 
> **Core Idea:** Find k nearest training points, then either VOTE (classification) or AVERAGE (regression).

## Q15. Describe k-NN Learning. Explain k-NN for Discrete-Valued Function with Pseudocode. (10 Marks)

### What is k-Nearest Neighbor (k-NN)?

**k-NN** is the most fundamental instance-based learning algorithm. To classify a new instance, it:
1. Finds the **k closest training examples** (by Euclidean distance).
2. Takes a **majority vote** of their class labels.

### Notation:

- $x_q$ = query (test) instance
- $x_i$ = i-th training instance
- $f(x_i)$ = class label of training instance
- V = set of possible class values
- δ(a, b) = 1 if a = b, else 0

### Distance Formula (Euclidean):

For instances with attributes $a_1, a_2, ..., a_n$:
$$d(x_q, x_i) = \sqrt{\sum_{r=1}^{n} (a_r(x_q) - a_r(x_i))^2}$$

### Pseudocode for Discrete-Valued (Classification):

```
ALGORITHM: k-NN for Classification
INPUT: 
    - Training set D = {(x_1, f(x_1)), ..., (x_m, f(x_m))}
    - Query instance x_q
    - Number of neighbors k

TRAINING PHASE:
    Store all training examples in D

CLASSIFICATION PHASE:
    1. For each training instance x_i in D:
           Compute d(x_q, x_i) using Euclidean distance
    
    2. Sort training instances by distance (ascending)
    
    3. Select the k nearest instances: x_1, x_2, ..., x_k
    
    4. Return the most common class:
           f_hat(x_q) = argmax_{v in V} sum_{i=1}^{k} delta(v, f(x_i))

OUTPUT: Predicted class label f_hat(x_q)
```

### Final Formula:

$$\hat{f}(x_q) = \arg\max_{v \in V} \sum_{i=1}^{k} \delta(v, f(x_i))$$

Where $\delta(a, b) = 1$ if a = b, else 0.

### Example (Mini):

Training: 
- A(1,1) = Red, B(2,2) = Red, C(5,5) = Blue, D(6,6) = Blue

Query: x_q = (1.5, 1.5), k = 3

Distances:
- d(A) = √0.5 ≈ 0.71
- d(B) = √0.5 ≈ 0.71
- d(C) = √24.5 ≈ 4.95
- d(D) = √40.5 ≈ 6.36

3 nearest: A (Red), B (Red), C (Blue)
Vote: Red=2, Blue=1 → **Predicted: Red**

### Step-by-Step Procedure:

| Step | Action |
|------|--------|
| 1 | Compute distance from query to ALL training points |
| 2 | Sort distances in ascending order |
| 3 | Pick top k closest points |
| 4 | Count class labels among these k points |
| 5 | Return class with most votes |

### Choosing k:

- **k too small (k=1):** Sensitive to noise (single bad neighbor misleads).
- **k too large:** Smooths out local patterns, loses detail.
- **Common choice:** k = √n, or use cross-validation.
- **Often:** Use odd k for binary classification (avoids ties).

---

## Q14. Explain k-NN for Continuous-Valued Target Functions with Pseudocode. (8 Marks)

### Difference from Classification:

For continuous targets (regression), instead of voting, we **average** the values of k nearest neighbors.

### Pseudocode:

```
ALGORITHM: k-NN for Regression
INPUT: 
    - Training set D = {(x_1, f(x_1)), ..., (x_m, f(x_m))}
    - Query x_q, value of k

TRAINING PHASE:
    Store all training examples

PREDICTION PHASE:
    1. For each training instance x_i:
           Compute d(x_q, x_i) using Euclidean distance
    
    2. Sort by distance and find k nearest: x_1, ..., x_k
    
    3. Compute the average:
           f_hat(x_q) = (1/k) * sum_{i=1}^{k} f(x_i)

OUTPUT: Predicted real value f_hat(x_q)
```

### Final Formula:

$$\hat{f}(x_q) = \frac{1}{k} \sum_{i=1}^{k} f(x_i)$$

### Example:

Training (predicting house prices):
- House A(area=1000, beds=2) = ₹50L
- House B(area=1200, beds=3) = ₹65L  
- House C(area=2000, beds=4) = ₹120L

Query: New house (area=1100, beds=2.5), k = 2

Distances (Euclidean on (area, beds)):
- d(A) ≈ 100.0
- d(B) ≈ 100.0
- d(C) ≈ 900.0

2 nearest: A (₹50L), B (₹65L)
Predicted price = (50 + 65) / 2 = **₹57.5L**

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Compute Euclidean distance to all training points |
| 2 | Sort and pick k nearest |
| 3 | Average the f(x_i) values |
| 4 | Return average as prediction |

### Difference Summary:

| Aspect | Classification | Regression |
|--------|----------------|------------|
| Output type | Discrete class | Real number |
| Combine neighbors | Majority vote | Average |
| Formula | argmax of count | Mean |

---

## Q9. Distance-Weighted k-NN for Discrete-Valued Function. (6 Marks)

### Motivation:

In standard k-NN, all k neighbors get **equal vote**. But intuitively, **closer neighbors should matter more** than far ones.

### Solution: Weight Each Vote by Inverse Distance

Each neighbor's vote is weighted by:
$$w_i = \frac{1}{d(x_q, x_i)^2}$$

(If d = 0 i.e., exact match, just return that class.)

### Modified Formula (Discrete):

$$\hat{f}(x_q) = \arg\max_{v \in V} \sum_{i=1}^{k} w_i \cdot \delta(v, f(x_i))$$

### Algorithm:

```
1. Compute d(x_q, x_i) for all training instances
2. Sort and pick k nearest
3. For each class v in V:
       score(v) = sum of (1/d_i^2) for neighbors with class v
4. Return v with maximum score
```

### Example:

Training:
- A(1,1) = Red, B(2,2) = Red, C(1.1, 1.1) = Blue (very close to query!)

Query: x_q = (1, 1), k = 3

Distances:
- d(A) = 0 → exact match, return Red
  
Let's modify: Query = (1.05, 1.05)
- d(A) = 0.071, w_A = 1/0.005 = 200
- d(B) = 1.34, w_B = 1/1.79 ≈ 0.56
- d(C) = 0.071, w_C = 1/0.005 = 200

Vote:
- Red: 200 + 0.56 = 200.56
- Blue: 200

Closer points dominate → **Red** wins.

### Advantages of Distance-Weighting:

1. **Less sensitive to choice of k** — distant points contribute almost nothing.
2. **Can use ALL training data** as neighbors (k = m). This is called **Shepard's method**.
3. **More robust to noise**.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Compute distances |
| 2 | Compute weights w_i = 1/d_i² |
| 3 | For each class, sum weights of neighbors of that class |
| 4 | Return class with maximum weighted sum |

---

## Q12. Distance-Weighted k-NN for Continuous-Valued Function. (6 Marks)

### Modified Formula (Continuous):

$$\hat{f}(x_q) = \frac{\sum_{i=1}^{k} w_i \cdot f(x_i)}{\sum_{i=1}^{k} w_i}$$

Where $w_i = \frac{1}{d(x_q, x_i)^2}$

This is a **weighted average** — closer points pull the prediction more.

### Algorithm:

```
1. Compute d(x_q, x_i) for all training instances
2. Sort by distance, pick k nearest
3. Compute w_i = 1 / d_i^2 for each
4. Compute weighted average:
       prediction = sum(w_i * f(x_i)) / sum(w_i)
```

### Example:

Predicting price for house:
- Training: A=₹50L (d=2), B=₹65L (d=1), C=₹100L (d=5)

Weights:
- w_A = 1/4 = 0.25
- w_B = 1/1 = 1.0
- w_C = 1/25 = 0.04

Weighted average:
$$\hat{f} = \frac{0.25 \times 50 + 1.0 \times 65 + 0.04 \times 100}{0.25 + 1.0 + 0.04} = \frac{12.5 + 65 + 4}{1.29} = \frac{81.5}{1.29} = ₹63.18\text{L}$$

Compare to plain average: (50+65+100)/3 = ₹71.67L. The closer house (B) pulled the prediction down.

### Difference: Discrete vs Continuous Distance-Weighted k-NN

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| Output | Class label | Real value |
| Combination | Weighted **vote** | Weighted **average** |
| Formula | argmax Σ wᵢ·δ(v, f(xᵢ)) | Σwᵢf(xᵢ) / Σwᵢ |

---

## Q7. Practical Issues in k-NN Implementation and Solutions. (5 Marks)

### Issue 1: Curse of Dimensionality

**Problem:** With many irrelevant attributes, the distance metric is dominated by noise. All points become "equally far", making k-NN useless.

**Example:** If you have 20 features but only 3 are relevant for classification, the 17 irrelevant features add noise to all distance calculations.

**Solutions:**
1. **Feature Selection:** Remove irrelevant attributes before applying k-NN.
2. **Feature Weighting:** Assign weights to attributes:
   $$d(x_i, x_j) = \sqrt{\sum_{r} z_r \cdot (a_r(x_i) - a_r(x_j))^2}$$
   where $z_r$ is learned weight (relevance) of attribute r.
3. **Dimensionality Reduction:** Use PCA or similar to reduce features.

### Issue 2: Efficient Nearest Neighbor Search

**Problem:** With m training examples and a query, naive search is O(m). For large datasets (millions of points), this is too slow.

**Solutions:**
1. **kd-trees:** Tree-based data structures that prune search space. Average query time: O(log m).
2. **Ball trees:** Better than kd-trees for high dimensions.
3. **Approximate Nearest Neighbor (ANN):** LSH (Locality-Sensitive Hashing) for very fast (but approximate) results.

### Other Common Issues:

| Issue | Solution |
|-------|----------|
| Different attribute scales | **Normalize** features (e.g., to [0,1] or z-score) |
| Choice of k | **Cross-validation** to find best k |
| Noisy data | Use **distance-weighted k-NN** or larger k |
| Memory cost | **Edited k-NN** — store only "useful" examples |

### Step-by-Step (How to Handle Issues):

| Issue | Detection | Fix |
|-------|-----------|-----|
| Curse of dimensionality | Many features, poor accuracy | Feature selection / weighting |
| Slow queries | Large training set | Use kd-tree |
| Different scales | Some features dominate | Normalize |
| Noisy labels | Single bad neighbor | Increase k or use weighted voting |

---

## Q13. Major Drawbacks of k-NN and How to Correct Them. (8 Marks)

This combines the answers from Q6 and Q7 into a more detailed treatment.

### Drawback 1: High Computational Cost at Prediction Time

**Issue:** Must compute distance from query to ALL m training points → O(m·n) time per query, where n = number of features.

**Correction:**
- Use **kd-trees** for low dimensions (≤ 10), giving O(log m) query.
- Use **ball trees** or **LSH** for high dimensions.
- Reduce m by **prototype selection** (keep only representative examples).

### Drawback 2: Curse of Dimensionality

**Issue:** Distances become meaningless in high-dimensional spaces — all points become roughly equidistant from the query.

**Correction:**
- Apply **feature selection** to keep only relevant attributes.
- Use **feature weighting**: $z_r$ for each attribute r.
- Apply **dimensionality reduction** (PCA, LDA).

### Drawback 3: Sensitivity to Irrelevant Attributes

**Issue:** Even a few irrelevant features can distort distance calculations.

**Correction:**
- **Stretching axes**: Adjust feature weights so important features have larger range.
- **Cross-validation** to learn optimal weights.

### Drawback 4: Sensitivity to Noisy Data

**Issue:** A single mislabeled neighbor can flip the prediction (especially for k=1).

**Correction:**
- Use **larger k** (e.g., k=5 or 7).
- Use **distance-weighted voting** (closer = more weight).
- Apply **noise filtering** before training.

### Drawback 5: Different Attribute Scales

**Issue:** Attribute with large range (e.g., income in lakhs) dominates over small-range attribute (e.g., age in years).

**Correction:**
- **Normalize** all features to [0, 1] using min-max scaling:
  $$x'_i = \frac{x_i - \min(x)}{\max(x) - \min(x)}$$
- Or **standardize** using z-score:
  $$x'_i = \frac{x_i - \mu}{\sigma}$$

### Drawback 6: Memory Cost

**Issue:** Stores all m training examples → expensive for large datasets.

**Correction:**
- Use **edited k-NN**: Remove redundant or correctly-classified-by-others points.
- Use **condensed k-NN**: Keep only points near decision boundaries.

### Summary Table:

| Drawback | Correction |
|----------|------------|
| Slow at query | kd-tree, ball-tree |
| Curse of dimensionality | Feature selection / PCA |
| Irrelevant attributes | Feature weighting |
| Noisy data | Larger k, weighted voting |
| Different scales | Normalization |
| Memory cost | Prototype selection |

### Step-by-Step Mitigation Procedure:

| Step | Action |
|------|--------|
| 1 | **Preprocess data**: Normalize features, handle missing values |
| 2 | **Feature engineering**: Remove irrelevant attributes |
| 3 | **Choose distance metric**: Euclidean, Manhattan, etc. |
| 4 | **Select k via cross-validation** |
| 5 | **Use kd-tree** for efficient querying |
| 6 | **Apply distance-weighted voting** for robustness |

---

## Q8. Define Terms in k-NN: Regression, Residual, Kernel Function. (6 Marks)

### i) Regression

**Definition:** **Regression** is the problem of approximating a **real-valued (continuous) target function** from training examples.

**In k-NN context:** Instead of predicting a class label (classification), regression predicts a numerical value by averaging the target values of nearest neighbors:
$$\hat{f}(x_q) = \frac{1}{k} \sum_{i=1}^{k} f(x_i)$$

**Examples:** Predicting house price, temperature, stock price, person's weight.

### ii) Residual

**Definition:** The **residual** is the **error** between the predicted value and the actual value for a given instance.

$$\text{Residual} = \hat{f}(x) - f(x)$$

Or equivalently: $\text{Residual} = \text{Predicted} - \text{Actual}$

**In k-NN regression:** If we predict 50 for a house but actual price is 55, residual = 50 - 55 = -5.

**Why important?**
- Sum of squared residuals is minimized in regression.
- Indicates how "wrong" the model is.

### iii) Kernel Function

**Definition:** A **kernel function** $K(d)$ is a function that **converts distance into a weight**. It decreases as distance increases — so closer points get higher weights.

**Common kernel:** Inverse distance kernel (used in distance-weighted k-NN):
$$K(d(x_q, x_i)) = \frac{1}{d(x_q, x_i)^2}$$

**Another common kernel:** Gaussian kernel (used in LWR and RBF):
$$K(d) = e^{-\frac{d^2}{2\sigma^2}}$$

**Purpose:** To smoothly weight contributions of training points based on how close they are to the query.

### Step-by-Step Summary:

| Term | Symbol/Formula | Meaning |
|------|----------------|---------|
| Regression | $\hat{f}(x_q) = \frac{1}{k} \sum f(x_i)$ | Predict continuous value |
| Residual | $\hat{f}(x) - f(x)$ | Error in prediction |
| Kernel | $K(d) = e^{-d²/2σ²}$ | Distance → weight conversion |

---

# SECTION C: k-NN NUMERICALS

> **Universal Procedure for ALL k-NN numericals:**
> 1. Compute Euclidean distance from query to every training point.
> 2. Sort by distance (ascending).
> 3. Pick the k nearest.
> 4. **Classification:** majority vote. **Regression:** average.

## Q17. NUMERICAL: Classify (Height=170, Weight=57) using k-NN (10 Marks)

### Given Training Data:

| Height (cm) | Weight (kg) | Class |
|-------------|-------------|-------|
| 167 | 51 | Underweight |
| 182 | 62 | Normal |
| 176 | 69 | Normal |
| 173 | 64 | Normal |
| 172 | 65 | Normal |
| 174 | 56 | Underweight |
| 169 | 58 | Normal |
| 173 | 57 | Normal |
| 170 | 55 | Normal |

**Query:** Height = 170, Weight = 57. Find class.

### Step 1: Compute Euclidean Distance to Each Point

**Formula:** $d = \sqrt{(h_i - 170)^2 + (w_i - 57)^2}$

| # | Height | Weight | Class | Calculation | Distance |
|---|--------|--------|-------|-------------|----------|
| 1 | 167 | 51 | Underweight | √(9 + 36) = √45 | **6.7082** |
| 2 | 182 | 62 | Normal | √(144 + 25) = √169 | **13.0000** |
| 3 | 176 | 69 | Normal | √(36 + 144) = √180 | **13.4164** |
| 4 | 173 | 64 | Normal | √(9 + 49) = √58 | **7.6158** |
| 5 | 172 | 65 | Normal | √(4 + 64) = √68 | **8.2462** |
| 6 | 174 | 56 | Underweight | √(16 + 1) = √17 | **4.1231** |
| 7 | 169 | 58 | Normal | √(1 + 1) = √2 | **1.4142** |
| 8 | 173 | 57 | Normal | √(9 + 0) = √9 | **3.0000** |
| 9 | 170 | 55 | Normal | √(0 + 4) = √4 | **2.0000** |

### Step 2: Sort by Distance (Ascending)

| Rank | Point # | Distance | Class |
|------|---------|----------|-------|
| 1 | 7 | 1.4142 | **Normal** |
| 2 | 9 | 2.0000 | **Normal** |
| 3 | 8 | 3.0000 | **Normal** |
| 4 | 6 | 4.1231 | Underweight |
| 5 | 1 | 6.7082 | Underweight |
| 6 | 4 | 7.6158 | Normal |
| 7 | 5 | 8.2462 | Normal |
| 8 | 2 | 13.0000 | Normal |
| 9 | 3 | 13.4164 | Normal |

### Step 3: Apply k-NN Voting

**For k = 3 (most common choice):**

3 nearest points are #7, #9, #8 — all have class **Normal**.

Vote count:
- Normal: 3
- Underweight: 0

$$\boxed{\hat{f}(170, 57) = \text{Normal}}$$

**For k = 5 (also common):**

5 nearest: #7 (Normal), #9 (Normal), #8 (Normal), #6 (Underweight), #1 (Underweight)

Vote count:
- Normal: 3
- Underweight: 2

$$\boxed{\hat{f}(170, 57) = \text{Normal}}$$

### Step-by-Step Summary:

| Step | Action | Result |
|------|--------|--------|
| 1 | Calculate Euclidean distance to all 9 points | 9 distances computed |
| 2 | Sort distances in ascending order | Point 7 closest (d=1.41) |
| 3 | For k=3, pick top 3 neighbors | All 3 are "Normal" |
| 4 | Majority vote among k neighbors | Normal wins |
| 5 | Return predicted class | **Normal** |

### Final Answer: **The query point (170, 57) is classified as "Normal"**.

---

## Q20. NUMERICAL: Classify Rani (Age=11, Gender=1) using k-NN (10 Marks)

### Given Training Data:

| Name | Age | Gender | Weight |
|------|-----|--------|--------|
| Anjali | 32 | 0 | 55.2 |
| Madhu | 40 | 0 | 48.5 |
| Sanjana | 16 | 1 | 40.0 |
| Anjana | 34 | 1 | 50.7 |
| Saket | 55 | 0 | 75.6 |
| Rahul | 40 | 0 | 48.5 |
| Pooja | 20 | 1 | 50.2 |
| Smith | 15 | 0 | 42.2 |
| Laxmi | 55 | 1 | 67.0 |

**Query:** Rani, Age = 11, Gender = 1. Predict her **Weight**.

> **Note:** Since "Weight" is a real number (continuous), this is a **k-NN regression** problem. We will **average** the weights of nearest neighbors.

### Step 1: Compute Euclidean Distance

**Formula:** $d = \sqrt{(\text{Age}_i - 11)^2 + (\text{Gender}_i - 1)^2}$

| Name | Age | Gender | Weight | Calculation | Distance |
|------|-----|--------|--------|-------------|----------|
| Anjali | 32 | 0 | 55.2 | √(441 + 1) = √442 | **21.0238** |
| Madhu | 40 | 0 | 48.5 | √(841 + 1) = √842 | **29.0172** |
| Sanjana | 16 | 1 | 40.0 | √(25 + 0) = √25 | **5.0000** |
| Anjana | 34 | 1 | 50.7 | √(529 + 0) = √529 | **23.0000** |
| Saket | 55 | 0 | 75.6 | √(1936 + 1) = √1937 | **44.0114** |
| Rahul | 40 | 0 | 48.5 | √(841 + 1) = √842 | **29.0172** |
| Pooja | 20 | 1 | 50.2 | √(81 + 0) = √81 | **9.0000** |
| Smith | 15 | 0 | 42.2 | √(16 + 1) = √17 | **4.1231** |
| Laxmi | 55 | 1 | 67.0 | √(1936 + 0) = √1936 | **44.0000** |

### Step 2: Sort by Distance

| Rank | Name | Distance | Weight |
|------|------|----------|--------|
| 1 | Smith | 4.1231 | 42.2 |
| 2 | Sanjana | 5.0000 | 40.0 |
| 3 | Pooja | 9.0000 | 50.2 |
| 4 | Anjali | 21.0238 | 55.2 |
| 5 | Anjana | 23.0000 | 50.7 |
| 6 | Madhu | 29.0172 | 48.5 |
| 7 | Rahul | 29.0172 | 48.5 |
| 8 | Laxmi | 44.0000 | 67.0 |
| 9 | Saket | 44.0114 | 75.6 |

### Step 3: Apply k-NN (Regression — Average)

**For k = 3:**

3 nearest: Smith (42.2), Sanjana (40.0), Pooja (50.2)

$$\hat{f}(\text{Rani}) = \frac{42.2 + 40.0 + 50.2}{3} = \frac{132.4}{3} = 44.13$$

$$\boxed{\text{Rani's predicted weight} \approx 44.13 \text{ kg}}$$

**For k = 5:**

5 nearest: Smith (42.2), Sanjana (40.0), Pooja (50.2), Anjali (55.2), Anjana (50.7)

$$\hat{f}(\text{Rani}) = \frac{42.2 + 40.0 + 50.2 + 55.2 + 50.7}{5} = \frac{238.3}{5} = 47.66$$

$$\boxed{\text{Rani's predicted weight (k=5)} \approx 47.66 \text{ kg}}$$

### Step-by-Step Summary:

| Step | Action | Result |
|------|--------|--------|
| 1 | Compute Euclidean distance from Rani to all 9 people | 9 distances |
| 2 | Sort by distance | Smith closest (d=4.12) |
| 3 | For k=3, pick top 3 neighbors | Smith, Sanjana, Pooja |
| 4 | Average their weights | (42.2+40.0+50.2)/3 |
| 5 | Output predicted weight | **44.13 kg** |

### Note on Question Interpretation:

Some textbooks may ask to predict if Rani is "Underweight/Normal/Overweight" instead of weight. In that case, the same procedure applies but you'd need a class column. With the given data, weight is the target, so we use **regression** (averaging).

---

# SECTION D: LOCALLY WEIGHTED REGRESSION

## Q16. Explain Locally Weighted Linear Regression with Equations. (10 Marks)

### Definition:

**Locally Weighted Regression (LWR)** is an instance-based method that:
- For each query point, fits a **local linear regression** model.
- Gives **higher weight** to training points closer to the query.
- Uses this local model to predict the query's value.

It's a **lazy learning** algorithm — a new model is fit for each query.

### Why "Locally Weighted"?

Standard linear regression fits ONE line for the whole dataset.  
LWR fits a **different line for each query**, weighted by distance.

### Setup:

- Training data: $(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)$
- Query point: $x_q$
- Want to predict: $\hat{y}(x_q)$

### Step 1: Assign Weights to Training Points

Use a **kernel function** that gives more weight to closer points. Most common is the **Gaussian kernel**:

$$w_i = \exp\left(-\frac{(x_i - x_q)^2}{2\tau^2}\right)$$

Where:
- τ (tau) is the **bandwidth** parameter (controls how "local" the model is).
- Smaller τ → only very close points matter.
- Larger τ → more points contribute.

### Step 2: Fit a Weighted Linear Model

We assume the local model is linear:
$$\hat{y} = \beta_0 + \beta_1 x$$

Find $\beta_0, \beta_1$ by minimizing **weighted squared error**:
$$E = \sum_{i=1}^{m} w_i (y_i - \beta_0 - \beta_1 x_i)^2$$

### Step 3: Solve for Coefficients (Normal Equations)

Setting derivatives to zero gives:
$$\sum w_i \cdot y_i = \beta_0 \sum w_i + \beta_1 \sum w_i \cdot x_i$$
$$\sum w_i \cdot x_i \cdot y_i = \beta_0 \sum w_i \cdot x_i + \beta_1 \sum w_i \cdot x_i^2$$

In matrix form:
$$\begin{bmatrix} \sum w_i & \sum w_i x_i \\ \sum w_i x_i & \sum w_i x_i^2 \end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} = \begin{bmatrix} \sum w_i y_i \\ \sum w_i x_i y_i \end{bmatrix}$$

Solve using Cramer's rule or matrix inversion.

### Step 4: Predict

$$\hat{y}(x_q) = \beta_0 + \beta_1 \cdot x_q$$

### Algorithm Pseudocode:

```
ALGORITHM: Locally Weighted Regression
INPUT: Training data, query x_q, bandwidth tau

1. For each training point (x_i, y_i):
       Compute w_i = exp(-(x_i - x_q)^2 / (2*tau^2))

2. Solve weighted normal equations to get beta_0, beta_1

3. Predict y_hat(x_q) = beta_0 + beta_1 * x_q

OUTPUT: y_hat(x_q)
```

### Advantages:

1. **Captures local patterns** — fits different curves in different regions.
2. **No global model assumption** — works for non-linear data.
3. **Smooth predictions** — kernel weighting gives smooth output.

### Disadvantages:

1. **Slow at query time** — must solve a regression for every query.
2. **Bandwidth selection is critical** — too small overfits, too large underfits.
3. **Memory intensive** — stores all training data.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | For query x_q, compute weights w_i using Gaussian kernel |
| 2 | Set up weighted least squares problem |
| 3 | Solve normal equations for β₀, β₁ |
| 4 | Predict ŷ = β₀ + β₁·x_q |

---

## Q18. NUMERICAL: LWR at X = 2.5 with Bandwidth = 1 (10 Marks)

### Given:

| X | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Y | 2 | 3 | 2 | 5 |

**Query:** $x_q = 2.5$, bandwidth $\tau = 1$.

### Step 1: Compute Gaussian Kernel Weights

**Formula:** $w_i = \exp\left(-\frac{(x_i - x_q)^2}{2\tau^2}\right) = \exp\left(-\frac{(x_i - 2.5)^2}{2}\right)$

| $x_i$ | $(x_i - 2.5)$ | $(x_i - 2.5)^2$ | $w_i = e^{-(x_i-2.5)^2 / 2}$ | Value |
|-------|---------------|-----------------|------------------------------|-------|
| 1 | -1.5 | 2.25 | $e^{-1.125}$ | **0.3247** |
| 2 | -0.5 | 0.25 | $e^{-0.125}$ | **0.8825** |
| 3 | +0.5 | 0.25 | $e^{-0.125}$ | **0.8825** |
| 4 | +1.5 | 2.25 | $e^{-1.125}$ | **0.3247** |

### Step 2: Compute Weighted Sums

| Quantity | Calculation | Value |
|----------|-------------|-------|
| $\sum w_i$ | 0.3247 + 0.8825 + 0.8825 + 0.3247 | **2.4143** |
| $\sum w_i x_i$ | 0.3247(1) + 0.8825(2) + 0.8825(3) + 0.3247(4) | **6.0357** |
| $\sum w_i x_i^2$ | 0.3247(1) + 0.8825(4) + 0.8825(9) + 0.3247(16) | **16.9916** |
| $\sum w_i y_i$ | 0.3247(2) + 0.8825(3) + 0.8825(2) + 0.3247(5) | **6.6851** |
| $\sum w_i x_i y_i$ | 0.3247(1)(2) + 0.8825(2)(3) + 0.8825(3)(2) + 0.3247(4)(5) | **17.7323** |

### Step 3: Solve Normal Equations

$$\begin{bmatrix} 2.4143 & 6.0357 \\ 6.0357 & 16.9916 \end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} = \begin{bmatrix} 6.6851 \\ 17.7323 \end{bmatrix}$$

**Compute determinant:**
$$\det = (2.4143)(16.9916) - (6.0357)^2 = 41.024 - 36.430 = 4.594$$

**Solve using Cramer's rule:**

$$\beta_0 = \frac{(16.9916)(6.6851) - (6.0357)(17.7323)}{4.594} = \frac{113.594 - 107.029}{4.594} = \frac{6.565}{4.594} \approx 1.429$$

$$\beta_1 = \frac{(2.4143)(17.7323) - (6.0357)(6.6851)}{4.594} = \frac{42.811 - 40.350}{4.594} = \frac{2.461}{4.594} \approx 0.536$$

### Step 4: Predict at $x_q = 2.5$

$$\hat{y}(2.5) = \beta_0 + \beta_1 \cdot 2.5 = 1.429 + 0.536 \times 2.5$$
$$= 1.429 + 1.340$$
$$= \boxed{2.769}$$

### Final Answer: **The predicted value at X = 2.5 is approximately 2.77**

### Step-by-Step Summary:

| Step | Action | Result |
|------|--------|--------|
| 1 | Compute Gaussian weights for each training point | w = [0.32, 0.88, 0.88, 0.32] |
| 2 | Compute weighted sums (Σw, Σwx, Σwx², Σwy, Σwxy) | 5 sums |
| 3 | Solve 2×2 normal equations for β₀, β₁ | β₀=1.429, β₁=0.536 |
| 4 | Plug x_q=2.5 into ŷ = β₀ + β₁·x_q | **2.77** |

### Key Insight:

The points x=2 and x=3 are closest to x_q=2.5 → they have highest weights (0.88) and dominate the fit. The points x=1 and x=4 have lower weights (0.32) and contribute less.

---

## Q19. Similarities and Differences between k-NN and LWR. (10 Marks)

### Similarities:

| # | Property | Both k-NN and LWR |
|---|----------|-------------------|
| 1 | **Lazy Learners** | Both store training data; defer processing till query |
| 2 | **Instance-Based** | Both use stored examples to make predictions |
| 3 | **Local Approximation** | Both make predictions based on nearby training points |
| 4 | **Distance-Based** | Both use distance metric (typically Euclidean) |
| 5 | **No Global Model** | Neither builds a single global hypothesis |
| 6 | **Non-Parametric** | Neither assumes a fixed-form global function |
| 7 | **Memory Intensive** | Both store all training data |
| 8 | **Easy to Add Data** | Just append new examples, no retraining |
| 9 | **Used for Regression** | Both can predict continuous values |
| 10 | **Affected by Curse of Dimensionality** | Both struggle with many irrelevant features |

### Differences:

| Aspect | k-NN | LWR |
|--------|------|-----|
| **Approach** | Picks k nearest, votes/averages | Fits a regression model locally |
| **Output Computation** | Discrete: majority vote. Continuous: average. | Solves weighted linear regression |
| **Number of Neighbors** | Uses exactly k points | Uses all points, weighted by distance |
| **Local Model** | None — just statistic of neighbors | Linear (or polynomial) function |
| **Smoothness** | Predictions can be jumpy | Predictions are smooth |
| **Weighting** | Equal (in basic) or 1/d² (in weighted) | Gaussian kernel weighting |
| **Computation** | Find k-min, then average/vote | Solve normal equations |
| **Captures Trends** | No (just neighbor statistics) | Yes (local trends via slope) |
| **Bandwidth/k** | Uses parameter k | Uses bandwidth τ |
| **Memory of "Direction"** | No — only proximity matters | Yes — captures local slope |

### Example Comparison:

**Data:** (1,2), (2,3), (3,2), (4,5)  
**Query:** x_q = 2.5

**k-NN (k=2):** Average of y for x=2 (y=3) and x=3 (y=2) → ŷ = 2.5

**LWR (τ=1):** Fits a line through these points weighted by distance → ŷ ≈ 2.77

LWR captures the upward trend; k-NN gives the simple average.

### When to Use Which?

- **k-NN:** Simple problems, classification, when you don't need smooth predictions.
- **LWR:** When you need smooth predictions, regression with trends, finer accuracy.

### Step-by-Step:

| Step | k-NN | LWR |
|------|------|-----|
| 1 | Compute distances | Compute distances |
| 2 | Sort and pick k nearest | Compute Gaussian weights for ALL |
| 3 | Vote (discrete) or Average (continuous) | Solve weighted regression |
| 4 | Output | Output |

---

# SECTION E: RBF NETWORKS & CASE-BASED REASONING

## Q5. Describe Radial Basis Function (RBF) with the Equation. (5 Marks)

### Definition:

A **Radial Basis Function (RBF)** is a real-valued function whose value depends **only on the distance** from a center point.

It's called "radial" because the function value is the same for all points at the same radial distance from the center.

### General Form:

$$\phi(x) = K(||x - c||)$$

Where:
- $x$ = input point
- $c$ = center of the basis function
- $||x - c||$ = distance from x to center
- $K$ = some monotonically decreasing function

### Most Common: Gaussian RBF

$$\phi(x) = \exp\left(-\frac{||x - c||^2}{2\sigma^2}\right)$$

Where:
- $c$ = center
- $\sigma$ = width (controls spread)
- $||x - c||$ = Euclidean distance from x to c

### Properties:

1. **Maximum at center:** $\phi(c) = 1$
2. **Decreases with distance:** Far points get values near 0.
3. **Symmetric:** Function value depends only on distance, not direction.
4. **Smooth:** Differentiable everywhere.

### Visualization:

```
   φ(x)
    |
  1 ┤    ╱╲
    |   ╱  ╲
    |  ╱    ╲
  0 ┤━━━━━━━━━━ x
       c (center)
```

### Example:

If $c = 5$, $\sigma = 1$:
- $\phi(5) = e^0 = 1$ (at center)
- $\phi(6) = e^{-0.5} ≈ 0.607$
- $\phi(8) = e^{-4.5} ≈ 0.011$ (far away)

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Identify a center point c |
| 2 | Compute distance ||x - c|| from input to center |
| 3 | Apply kernel: φ(x) = exp(-d²/2σ²) |
| 4 | Output is high if x close to c, near 0 if far |

### Where Used:

- **RBF Networks** (covered in Q2)
- **Locally Weighted Regression** (kernel for weights)
- **Support Vector Machines** (RBF kernel)

---

## Q2. Explain Radial Basis Function Network with Diagram. (5 Marks)

### Definition:

A **Radial Basis Function (RBF) Network** is a type of artificial neural network with **3 layers**:
1. **Input layer**
2. **Hidden layer** (with RBF activation functions)
3. **Output layer** (linear combination)

It's used for **function approximation** and **classification**.

### Structure:

```
   Input Layer       Hidden Layer (RBF)        Output Layer
   
   x_1 ──────┐       φ_1 = exp(-||x-c_1||²/2σ²)
              \      
   x_2 ───────┼────► φ_2 = exp(-||x-c_2||²/2σ²) ──┐
              /                                     \
   x_3 ──────┘       φ_k = exp(-||x-c_k||²/2σ²)    ──► Σ w_i·φ_i  ──► f(x)
                     
                     (Each unit centered at c_i)        (Linear output)
```

### Mathematical Formulation:

The output of an RBF network is:
$$f(x) = w_0 + \sum_{u=1}^{k} w_u \cdot K_u(d(x_u, x))$$

Where:
- $K_u$ = kernel function for u-th hidden unit
- $x_u$ = center of u-th unit
- $w_u$ = weight from u-th unit to output
- $w_0$ = bias

### Common Kernel (Gaussian):

$$K_u(d(x_u, x)) = \exp\left(-\frac{||x - x_u||^2}{2\sigma_u^2}\right)$$

### Components:

1. **Input Layer:** Just passes input features.

2. **Hidden Layer (RBF Layer):**
   - Each node represents a "prototype" with center $c_u$ and width $\sigma_u$.
   - Computes how close x is to its center using kernel.

3. **Output Layer:**
   - Takes weighted sum of hidden activations.
   - Linear combination → produces final output.

### Training (Two-Stage):

**Stage 1: Set Centers and Widths**
- Use clustering (e.g., k-means) on training data.
- Or set centers at random points.

**Stage 2: Learn Output Weights**
- Solve a linear system: minimizing squared error.
- Fast because only output layer is "trained".

### Advantages:

1. **Fast training** (compared to MLP/backpropagation).
2. **Universal approximation** — can approximate any function.
3. **Local response** — good for problems with local structure.

### Disadvantages:

1. Number of hidden units (k) must be chosen.
2. Centers and widths affect performance.
3. Doesn't extrapolate well outside training region.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Choose centers c_u (often via clustering) |
| 2 | Set widths σ_u |
| 3 | For input x, compute hidden activations φ_u(x) |
| 4 | Compute output as weighted sum: f(x) = Σ w_u · φ_u(x) |
| 5 | Train weights w by linear regression |

### Connection to Other Methods:

RBF networks combine:
- **Eager learning:** Centers and weights are computed during training.
- **Local approximation:** Each RBF unit handles its own region.

---

## Q3. Illustrate Case-Based Reasoning (CBR) with a Suitable Example. (5 Marks)

### Definition:

**Case-Based Reasoning (CBR)** is an instance-based method that:
- Stores past **cases** (problems and their solutions).
- Solves new problems by **adapting solutions** of similar past cases.

It's like how humans solve problems: "I've seen something like this before, let me adapt that solution."

### How CBR Differs from k-NN:

| Aspect | k-NN | CBR |
|--------|------|-----|
| Representation | Numerical feature vectors | Rich symbolic descriptions |
| Distance | Euclidean | Custom similarity measures |
| Combination | Vote / Average | **Adapt** old solution to new case |
| Knowledge use | Just data | Domain knowledge + cases |

### CBR Process (4 R's):

1. **Retrieve** — Find similar past cases.
2. **Reuse** — Adapt their solutions to new problem.
3. **Revise** — Test the adapted solution; fix if wrong.
4. **Retain** — Store the new case for future use.

### Example: Help Desk System

**Stored cases:**
- Case 1: "Computer won't start, no power light" → Solution: Check power cable.
- Case 2: "Computer won't start, power light on but no display" → Solution: Check monitor connection.
- Case 3: "Slow internet on Chrome" → Solution: Clear browser cache.

**New problem:** "Computer doesn't start, no power light comes on"

**CBR Steps:**
1. **Retrieve:** Most similar case = Case 1.
2. **Reuse:** Adapt solution → "Check the power cable".
3. **Revise:** User checks cable. Issue solved!
4. **Retain:** Add this case to database for future.

### Another Example: Legal Reasoning

A lawyer uses past similar cases to argue current ones. Each new case enriches their experience for future cases.

### Advantages:

1. Mimics **human reasoning**.
2. Handles **complex, structured problems**.
3. Continuously **learns** from new experiences.
4. Provides **explanations** (cite past similar case).

### Disadvantages:

1. Hard to design **similarity measure** for complex cases.
2. **Adaptation** of solutions is non-trivial.
3. Case base may grow large.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | New problem arrives |
| 2 | Retrieve similar past cases from case base |
| 3 | Reuse: adapt their solution to the new problem |
| 4 | Revise: test the adapted solution, fix if needed |
| 5 | Retain: store the new case for future use |

---

## Q11. Explain CADET System Using T-Junction Pipe Example. (6 Marks)

### What is CADET?

**CADET (Case-based Design Tool)** is a Case-Based Reasoning system used to **design mechanical devices** (like water valves, pipes, etc.) by retrieving and adapting designs from similar past cases.

### How CADET Works:

CADET stores cases as:
- **Function:** What the device should do (specified using qualitative relationships).
- **Structure:** The actual design (parts and how they're connected).

When a new design problem arrives:
1. **Retrieve** cases with similar functions.
2. **Reuse** by combining/adapting their structures.
3. **Output** a candidate design.

### T-Junction Pipe Example:

**Problem:** Design a water faucet that mixes hot and cold water to produce warm water of desired temperature and flow rate.

**Function specification:**
- Inputs: Hot water flow $Q_h$ at temperature $T_h$, Cold water flow $Q_c$ at temperature $T_c$
- Output: Mixed water at desired flow $Q_m$ and temperature $T_m$
- Requirements (qualitative):
  - More hot water → higher output temperature
  - More cold water → lower output temperature
  - Sum of flows = output flow

**CADET retrieves a similar known design — the T-junction pipe:**

```
        Hot water (Q_h, T_h)
              |
              v
        ─────────────
        |    │      | ──► Mixed water (Q_m, T_m)
        ─────────────
              ^
              |
        Cold water (Q_c, T_c)
```

### How CADET Reasons:

In the T-junction case:
- $Q_m = Q_h + Q_c$ ✓ (matches "sum of flows")
- $T_m$ = weighted average ✓ (more hot → higher temp)

CADET sees these qualitative relations match the new problem and **suggests using a T-junction** as the basic structure for the faucet.

### Then Adaptation Occurs:

CADET adds:
- Valves to control $Q_h, Q_c$
- A handle to operate valves
- Producing the final faucet design.

### Why This Example Matters:

It shows CBR working in **engineering design**:
1. Reuses **proven designs**.
2. Reasons at a **functional level** (not just numerical similarity).
3. Combines multiple cases if needed.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | User provides function specification (qualitative) |
| 2 | CADET searches case base for cases with similar function |
| 3 | Retrieves T-junction (or similar) as a candidate |
| 4 | Adapts the structure (adds valves, handle) |
| 5 | Outputs the new device design |

### Limitations of CADET:

1. Cannot always combine cases automatically.
2. Limited to qualitative reasoning.
3. Case base must be carefully built by experts.

### Why It's Important:

CADET demonstrates that CBR can be applied beyond simple classification — to **creative design tasks**.

---

# UNIVERSAL EXAM CHEATSHEET (Last-Minute Revision)

## Common Procedure for ALL k-NN Numericals (Q17, Q20):

```
Step 1: Compute Euclidean distance from query to all training points
        d = sqrt(sum of (a_q - a_i)^2 for each attribute)

Step 2: Sort distances in ascending order

Step 3: Pick top k nearest neighbors

Step 4a: For Classification → Majority Vote
Step 4b: For Regression → Average target values
```

## Procedure for LWR Numerical (Q18):

```
Step 1: Compute weights using Gaussian kernel
        w_i = exp(-(x_i - x_q)^2 / (2*tau^2))

Step 2: Compute weighted sums
        Σw, Σwx, Σwx², Σwy, Σwxy

Step 3: Solve normal equations for β₀, β₁

Step 4: Predict ŷ = β₀ + β₁ · x_q
```

## Quick Comparison: All Numerical Answers

| Question | Type | Method | Answer |
|----------|------|--------|--------|
| Q17 | k-NN Classification | k=3, majority vote | **Normal** |
| Q18 | LWR Regression | Gaussian kernel, τ=1 | **2.77** |
| Q20 | k-NN Regression | k=3, average | **44.13 kg** |

---

# IMPORTANT POINTS TO REMEMBER FOR EXAM

## 1. Always show:
- Formula being used
- All distance calculations (especially squares and square roots)
- Sorted list with rank
- Final boxed answer

## 2. Common mistakes to avoid:
- ❌ Forgetting to take square root in Euclidean distance
- ❌ Not normalizing features when scales differ
- ❌ Using vote for regression (should average)
- ❌ Using average for classification (should vote)
- ❌ For LWR: forgetting to weight in normal equations

## 3. Tips for Theory Questions:
- **Always include a diagram** for RBF Network and BBN-style questions.
- **Give an example** even if not asked.
- **List advantages/disadvantages** to fill answer.
- **End with step-by-step procedure** in a table.

## 4. Decision-making in numericals:
- **Classification** (label/class) → **Majority Vote**
- **Regression** (number) → **Average**
- **Distance-weighted** → use $w_i = 1/d_i^2$

---

# FINAL TIPS FOR THE EXAM

1. **Memorize ONLY the master formulas** — they cover the entire module.

2. **Practice the calculation pattern**:
   - Distance → Sort → Vote/Average

3. **Show ALL steps** — examiners give marks for method.

4. **For theoretical questions**:
   - Definition → Formula → Example → Diagram → Pros/Cons → Step-by-step

5. **For numerical questions**:
   - State given data → Identify formula → Substitute → Calculate → Box answer

6. **Time management**: 
   - 5-mark questions: 8-10 minutes
   - 6-mark questions: 12 minutes
   - 8-mark questions: 15 minutes
   - 10-mark questions: 18-20 minutes

---

## All The Best for Your Exam! 🎯

Remember: **Instance-Based Learning is just three steps:**
1. **Find similar examples** (distance)
2. **Combine them** (vote or average or fit local model)
3. **Output prediction**

That's the entire module! Once you understand these 3 steps, all 20 questions become variations of the same idea.
