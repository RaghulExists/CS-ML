# Module 3: Decision Tree Learning - Complete Exam Study Guide

> **For 3rd Year BE Students | VTU/Karnataka Curriculum**
> **Topic:** Decision Trees, Entropy, Information Gain, ID3, Pruning

---

## How This Guide is Organized

I have grouped the 20 questions into **5 logical sections**. Once you understand one question in a section, the others become easy because they use the **same core concepts** with different applications.

| Section | Topic | Questions |
|---------|-------|-----------|
| **A** | Decision Tree Foundations | Q1, Q11, Q12, Q16, Q17 |
| **B** | Boolean Function Trees | Q2, Q9 |
| **C** | Entropy & Information Gain (Numericals + Theory) | Q10, Q13, Q18, Q20 |
| **D** | Issues in Decision Tree Learning | Q4, Q5, Q6, Q7, Q8, Q14, Q15 |
| **E** | Bias and Theoretical Concepts | Q3, Q19 |

---

# THE MASTER FORMULA SHEET (Memorize These!)

You only need these 4 formulas to solve the entire module:

### 1. Entropy (THE most important formula)
$$\text{Entropy}(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-)$$

For multiple classes:
$$\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $p_+$ = proportion of positive examples
- $p_-$ = proportion of negative examples
- Entropy = 0 (pure), Entropy = 1 (max impurity for binary)

### 2. Information Gain
$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$

In words: "How much does splitting on attribute A reduce uncertainty?"

### 3. Split Information (for Gain Ratio)
$$\text{SplitInfo}(S, A) = -\sum_{i=1}^{c} \frac{|S_i|}{|S|} \log_2 \frac{|S_i|}{|S|}$$

### 4. Gain Ratio
$$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$

---


# SECTION A: DECISION TREE FOUNDATIONS

## Q12. Define Decision Tree. Describe How a Decision Tree is Represented Using an Example. (6 Marks)

### Definition:

A **Decision Tree** is a tree-structured classifier where:
- **Internal nodes** represent tests on an attribute.
- **Branches** represent the outcome of the test.
- **Leaf nodes** represent the class labels (decisions).

It classifies an instance by **sorting it down the tree** from root to leaf.

### Components of a Decision Tree:

1. **Root Node:** The topmost node; first attribute to test.
2. **Internal Nodes:** Each represents a test on one attribute.
3. **Branches:** Outgoing edges representing each possible value of the attribute.
4. **Leaf Nodes:** Each represents a class label (final decision).

### Example: PlayTennis Decision Tree

```
              Outlook
         ┌──────┼──────┐
       Sunny  Overcast  Rain
         │      │        │
       Humidity  Yes    Wind
         │              │
       ┌─┴─┐          ┌─┴─┐
      High Normal   Strong Weak
       │     │        │     │
       No   Yes       No   Yes
```

### How to Classify Using This Tree:

**Test instance:** ⟨Outlook=Sunny, Temperature=Hot, Humidity=High, Wind=Weak⟩

**Steps:**
1. Start at root: Test Outlook → Sunny → go left
2. Next node: Test Humidity → High → go left
3. Reach leaf: **No** (don't play tennis)

### Equivalent Rule Representation:

The tree can be written as IF-THEN rules:
- IF Outlook=Sunny AND Humidity=High THEN PlayTennis=No
- IF Outlook=Sunny AND Humidity=Normal THEN PlayTennis=Yes
- IF Outlook=Overcast THEN PlayTennis=Yes
- IF Outlook=Rain AND Wind=Strong THEN PlayTennis=No
- IF Outlook=Rain AND Wind=Weak THEN PlayTennis=Yes

### Step-by-Step Procedure:

| Step | Action |
|------|--------|
| 1 | Start at the root node |
| 2 | Test the attribute at the current node |
| 3 | Follow the branch matching the attribute value |
| 4 | If you reach an internal node, repeat from step 2 |
| 5 | If you reach a leaf node, output its class label |

### Why Decision Trees are Useful:

- **Interpretable** (can be read as rules).
- **Handle both categorical and numerical data**.
- **No assumptions about data distribution**.
- **Fast classification** at query time.

---


## Q1. Describe Appropriate Problems for Decision Tree Learning. (5 Marks)

Decision tree learning is best suited for problems with these characteristics:

### 1. Instances Represented by Attribute-Value Pairs
- Each instance is described by a fixed set of attributes (e.g., Temperature, Humidity).
- Each attribute takes a small number of values (e.g., Hot, Mild, Cool).
- **Example:** Patient records with attributes like Age, Symptoms, Test Results.

### 2. Target Function Has Discrete Output Values
- The decision tree assigns each instance to a class (Yes/No, A/B/C).
- **Example:** Classify a loan applicant as Approve/Reject.
- (Trees can handle continuous outputs too, but discrete is most natural.)

### 3. Disjunctive Descriptions May Be Required
- The target concept might require expressions like "(A AND B) OR (C AND D)".
- Decision trees naturally represent disjunctions (different paths from root to leaves).

### 4. Training Data May Contain Errors (Robust to Noise)
- Decision tree algorithms (especially with pruning) are **robust to noisy/erroneous data**.
- Both errors in attribute values and class labels can be handled.

### 5. Training Data May Have Missing Attribute Values
- Decision trees can be modified to handle missing values (e.g., assign most common value, or fractional weights).

### Examples of Suitable Problems:

| Domain | Problem |
|--------|---------|
| **Medical Diagnosis** | Diagnose disease from symptoms |
| **Credit Approval** | Decide whether to approve loan |
| **Weather** | Predict whether to play tennis |
| **Fault Diagnosis** | Identify equipment failure cause |
| **Customer Behavior** | Predict if customer will buy |

### Step-by-Step Checklist:

| Criterion | Match? |
|-----------|--------|
| Instances are attribute-value pairs | ✓ |
| Target output is discrete | ✓ (preferred) |
| Need to express disjunctions | ✓ |
| Data may have noise | ✓ (handled by pruning) |
| Data may have missing values | ✓ (with modification) |

If your problem matches these → **Decision Trees are a good choice!**

---

## Q11. List All Issues Faced in Decision Tree Learning. (5 Marks)

Decision tree learning faces several practical issues:

### 1. Determining How Deeply to Grow the Tree
- **Issue:** A very deep tree may **overfit** the training data.
- **Solution:** Pruning (Reduced Error Pruning, Rule Post-Pruning).

### 2. Handling Continuous-Valued Attributes
- **Issue:** ID3 originally only handles discrete attributes.
- **Solution:** Define threshold-based splits (e.g., Temperature > 75).

### 3. Choosing an Appropriate Attribute Selection Measure
- **Issue:** Information Gain is biased toward attributes with many values.
- **Solution:** Use **Gain Ratio** instead.

### 4. Handling Training Data with Missing Attribute Values
- **Issue:** Real data often has missing values.
- **Solution:** Assign most common value, or use fractional examples.

### 5. Handling Attributes with Different Costs
- **Issue:** Some attributes are expensive to measure (e.g., medical tests).
- **Solution:** Modify gain to penalize costly attributes.

### 6. Improving Computational Efficiency
- **Issue:** Building the tree is expensive for large datasets.
- **Solution:** Various optimizations (sampling, parallelization).

### 7. Overfitting (THE BIGGEST ISSUE)
- **Issue:** Tree memorizes training data, performs poorly on new data.
- **Solution:** Pre-pruning (stop early) or Post-pruning (build full, then prune).

### Summary Table:

| # | Issue | Solution |
|---|-------|----------|
| 1 | Tree depth (overfitting) | Pruning |
| 2 | Continuous attributes | Threshold splits |
| 3 | Attribute selection bias | Gain Ratio |
| 4 | Missing values | Most common value / fractions |
| 5 | Different attribute costs | Cost-sensitive gain |
| 6 | Computational cost | Optimization techniques |
| 7 | Overfitting | Pruning, validation set |

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Identify which issues apply to your dataset |
| 2 | Apply appropriate preprocessing (handle missing values, normalize) |
| 3 | Choose appropriate attribute selection measure |
| 4 | Build the tree |
| 5 | Apply pruning to avoid overfitting |
| 6 | Validate on test data |

---


## Q16. Discuss Capabilities and Limitations of Decision Tree Learning. (8 Marks)

### Capabilities (Advantages):

#### 1. Easy to Understand and Interpret
- The tree can be visualized and read like rules.
- Non-experts can understand why a decision was made.

#### 2. Requires Little Data Preparation
- No need to normalize or scale data.
- Works with both numerical and categorical data.

#### 3. Handles Both Numerical and Categorical Data
- Numerical: split using thresholds (e.g., Age > 30).
- Categorical: split into branches per value.

#### 4. Performs Well on Large Datasets
- Tree-building algorithms (ID3, C4.5) are reasonably efficient.

#### 5. Robust to Outliers
- Outliers don't significantly affect the tree (unlike linear models).

#### 6. Can Capture Non-Linear Relationships
- Trees naturally split data into regions, capturing non-linear patterns.

#### 7. Can Be Converted to Rules
- Each path from root to leaf is one rule, which is human-readable.

### Limitations (Disadvantages):

#### 1. Overfitting
- Trees can grow too complex and memorize training data.
- **Fix:** Pruning (pre/post-pruning).

#### 2. Instability
- Small changes in data can lead to very different trees.
- **Fix:** Use ensemble methods (Random Forest).

#### 3. Bias Toward Attributes with Many Values
- Information Gain favors attributes with more values.
- **Fix:** Use Gain Ratio instead.

#### 4. Greedy Algorithm
- ID3 makes locally optimal choices; not globally optimal.
- May not produce the best tree.

#### 5. Difficulty with XOR-Type Functions
- Trees struggle with functions like A XOR B.
- Each split must be on one attribute at a time.

#### 6. Continuous Output is Awkward
- Originally designed for discrete outputs (classification).
- Regression trees are more complex.

#### 7. Class Imbalance
- If one class dominates, tree may be biased toward it.

### Summary Table:

| Capability | Limitation |
|------------|-----------|
| Easy to understand | Prone to overfitting |
| No data preparation | Unstable (small changes → big tree changes) |
| Handles mixed data types | Bias to multi-valued attributes |
| Captures non-linearity | Greedy (not globally optimal) |
| Converts to rules | Struggles with XOR |
| Robust to outliers | Better for discrete output |

### When to Use Decision Trees:

✅ When you need an **interpretable model**.  
✅ When data is a **mix of categorical and numerical**.  
✅ When you don't want to do **heavy preprocessing**.

### When NOT to Use:

❌ For **complex non-linear** patterns (use neural networks).  
❌ When **stability** is critical.  
❌ For **continuous targets** with high precision needs.

---

## Q17. Describe the ID3 Algorithm for Decision Tree Learning. (10 Marks)

### What is ID3?

**ID3 (Iterative Dichotomiser 3)** is the foundational algorithm for building decision trees. Developed by Ross Quinlan in 1986.

### Core Idea:

**At each step, choose the attribute that best splits the data — i.e., the one with the highest Information Gain.**

### Pseudocode:

```
ID3(Examples, Target_attribute, Attributes)
    Create a Root node for the tree
    
    If all Examples are positive, return single-node tree Root with label = Yes
    If all Examples are negative, return single-node tree Root with label = No
    
    If Attributes is empty:
        return single-node tree Root with label = most common value of Target_attribute in Examples
    
    Otherwise:
        A <- the attribute from Attributes that best classifies Examples
        (best = highest Information Gain)
        
        The decision attribute for Root <- A
        
        For each possible value v_i of A:
            Add a new tree branch below Root, corresponding to test A = v_i
            Let Examples_v_i be the subset of Examples with A = v_i
            
            If Examples_v_i is empty:
                Add a leaf node with label = most common value of Target_attribute in Examples
            Else:
                Add subtree ID3(Examples_v_i, Target_attribute, Attributes - {A})
    
    Return Root
```

### Key Terms:

- **Examples:** Training data (set of instances).
- **Target_attribute:** The class label we're predicting (e.g., PlayTennis).
- **Attributes:** Features available to split on.

### Step-by-Step Procedure:

| Step | Action |
|------|--------|
| 1 | Compute entropy of current dataset |
| 2 | For each attribute, compute information gain |
| 3 | Pick the attribute with **highest gain** as the splitting attribute |
| 4 | Create branches for each value of that attribute |
| 5 | Recursively repeat for each branch's subset |
| 6 | Stop when:<br>(a) All examples in subset have same class → leaf node<br>(b) No attributes left → leaf with majority class<br>(c) Subset is empty → leaf with majority of parent |

### Example: PlayTennis (Brief Walk-through)

**Dataset:** 14 examples, 9 Yes / 5 No

**Step 1:** Entropy(S) = 0.94

**Step 2:** Compute gain for each attribute:
- Gain(S, Outlook) = 0.247 ← Highest!
- Gain(S, Humidity) = 0.151
- Gain(S, Wind) = 0.048
- Gain(S, Temperature) = 0.029

**Step 3:** Choose **Outlook** as root.

**Step 4:** Split into branches:
- Sunny → Recurse on subset (5 examples)
- Overcast → All Yes → Leaf "Yes"
- Rain → Recurse on subset (5 examples)

**Step 5:** Continue until all leaves are pure.

### Properties of ID3:

1. **Top-down, greedy search** through hypothesis space.
2. **No backtracking** — once a choice is made, it's not revised.
3. **Statistically based** — uses entropy and information gain.
4. **Inductive bias:** Prefers shorter trees with high-gain attributes near root.

### Limitations of ID3:

1. Only handles **discrete attributes** (extension C4.5 handles continuous).
2. Doesn't handle **missing values**.
3. **Overfits** without pruning.
4. **Biased** toward attributes with many values.

---


# SECTION B: BOOLEAN FUNCTION DECISION TREES

## Q2. Build Decision Trees for Boolean Functions. (5 Marks)

### General Approach for Boolean Functions:

For a Boolean function, we build a tree where:
- Each internal node tests one Boolean variable.
- Branches are labeled "true" or "false" (or "1" or "0").
- Leaf nodes are the function's output.

### Part (a): A ∨ (B ∧ C)

**Truth Table:**

| A | B | C | A ∨ (B ∧ C) |
|---|---|---|-------------|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 |

**Decision Tree:**

```
            A
          ┌─┴─┐
         T     F
         │     │
         1     B
              ┌┴┐
              T  F
              │  │
              C  0
             ┌┴┐
             T  F
             │  │
             1  0
```

**Logic:**
- If A = True → output 1 (regardless of B, C).
- If A = False, B = False → output 0.
- If A = False, B = True, C = True → output 1.
- If A = False, B = True, C = False → output 0.

### Part (b): (A ∧ B) ∨ (C ∧ D)

**Decision Tree:**

```
                A
              ┌─┴─┐
             T     F
             │     │
             B     C
            ┌┴┐   ┌┴┐
            T F   T F
            │ │   │ │
            1 C   D 0
              ┌┴┐ ┌┴┐
              T F T F
              │ │ │ │
              1 0 1 0
```

**Logic:**
- If A=T, B=T → 1 (because A∧B is true)
- If A=T, B=F → check C∧D
- If A=F → A∧B is false; check C∧D

### Step-by-Step Procedure for Boolean Functions:

| Step | Action |
|------|--------|
| 1 | Write the truth table for the function |
| 2 | Choose any variable as root (often the most "important") |
| 3 | For each branch (True/False), simplify the function |
| 4 | Recursively build subtree for each simplified function |
| 5 | When function reduces to constant (0 or 1), make leaf |

### Tips:

- Try to **minimize tree depth** by choosing variables wisely.
- Variables that appear in more terms are usually better roots.
- **Always test:** trace the tree for each truth table row.

---

## Q9. Construct Decision Trees for: (a) A ∧ ¬B, (b) A XOR B (5 Marks)

### Part (a): A ∧ ¬B

**Truth Table:**

| A | B | ¬B | A ∧ ¬B |
|---|---|----|--------|
| 0 | 0 | 1  | 0 |
| 0 | 1 | 0  | 0 |
| 1 | 0 | 1  | 1 |
| 1 | 1 | 0  | 0 |

**Decision Tree:**

```
            A
          ┌─┴─┐
         T     F
         │     │
         B     0
       ┌─┴─┐
      T     F
      │     │
      0     1
```

**Logic:**
- If A = False → output 0 (since A∧... = 0).
- If A = True, B = True → output 0 (since ¬B = 0).
- If A = True, B = False → output 1 (since A=1, ¬B=1).

### Part (b): A XOR B

**Truth Table:**

| A | B | A XOR B |
|---|---|---------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Decision Tree:**

```
            A
          ┌─┴─┐
         T     F
         │     │
         B     B
       ┌─┴─┐ ┌─┴─┐
      T     F T     F
      │     │ │     │
      0     1 1     0
```

**Logic:**
- A=T, B=T → 0 (same → false)
- A=T, B=F → 1 (different → true)
- A=F, B=T → 1 (different → true)
- A=F, B=F → 0 (same → false)

### Important Note on XOR:

XOR is **notoriously hard** for decision trees because:
- Knowing only A doesn't help — the result depends on **both A and B together**.
- The tree must always test both attributes.
- This shows a **limitation** of decision trees for non-linearly separable functions.

### Step-by-Step Construction:

| Step | Action |
|------|--------|
| 1 | Write truth table |
| 2 | Pick first attribute (A) for root |
| 3 | For each value (T/F), check remaining function |
| 4 | If function depends on more variables, add more nodes |
| 5 | If function is determined, make a leaf |

---


# SECTION C: ENTROPY & INFORMATION GAIN (Theory + Numericals)

> **Master these formulas — they're used in EVERY entropy/gain question:**
> 
> **Entropy:** $E(S) = -p_+\log_2(p_+) - p_-\log_2(p_-)$
> 
> **Information Gain:** $\text{Gain}(S, A) = E(S) - \sum_v \frac{|S_v|}{|S|} E(S_v)$

## Q13. Explain the Concept of Entropy with Mathematical Equations. (6 Marks)

### Definition:

**Entropy** is a measure of **impurity** or **uncertainty** in a dataset.

- High entropy = data is mixed (uncertain about class).
- Low entropy = data is pure (one class dominates).

### Mathematical Definition:

For a binary classification (positive/negative):
$$E(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-)$$

For c classes:
$$E(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $p_+$ = proportion of positive examples in S
- $p_-$ = proportion of negative examples in S
- $p_i$ = proportion of class i

### Key Properties:

1. **Entropy = 0 when set is pure** (all same class).
   - Example: 10 Yes, 0 No → Entropy = 0
2. **Entropy = 1 when set is balanced** (50-50 binary).
   - Example: 5 Yes, 5 No → Entropy = 1
3. **Entropy ranges between 0 and log₂(c)** for c classes.

### Why log₂?

- Information theory measures uncertainty in **bits**.
- log₂ gives entropy in bits.
- 1 bit = 1 binary decision needed to determine class.

### Visualization:

```
   Entropy
     |
   1 ┤      ╱╲
     |    ╱    ╲
     |  ╱        ╲
   0 ┤────────────── p_+
     0    0.5    1
```

Maximum at p = 0.5 (most uncertain), zero at extremes (most certain).

### Examples:

**Example 1:** 14 examples, 9 Yes, 5 No
$$E(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14}$$
$$= -0.643 \times (-0.637) - 0.357 \times (-1.485)$$
$$= 0.410 + 0.530$$
$$= 0.940$$

**Example 2:** 4 Yes, 0 No (pure)
$$E(S) = 0$$

**Example 3:** 5 Yes, 5 No (max impure)
$$E(S) = -0.5\log_2(0.5) - 0.5\log_2(0.5) = 0.5 + 0.5 = 1$$

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Count number of each class |
| 2 | Compute proportions p₊ and p₋ |
| 3 | Compute log₂ of each proportion |
| 4 | Apply entropy formula: -p₊log₂(p₊) - p₋log₂(p₋) |
| 5 | Result is entropy in bits |

### Use in Decision Trees:

Entropy is used to:
- Measure how "mixed" a dataset is.
- Compute **Information Gain** (next topic).
- Decide which attribute to split on.

---

## Q10. Explain Information Gain with Mathematical Equations. (5 Marks)

### Definition:

**Information Gain** measures **how much an attribute reduces uncertainty (entropy)** when used to split the data.

- High gain = attribute is good for splitting.
- Low gain = attribute doesn't help much.

### Mathematical Definition:

$$\text{Gain}(S, A) = E(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} E(S_v)$$

Where:
- E(S) = entropy of full dataset before split
- $S_v$ = subset of S where attribute A has value v
- $|S_v|/|S|$ = proportion of examples with A = v
- $E(S_v)$ = entropy of subset $S_v$

### Intuition:

**Gain = (Original Entropy) - (Average Entropy after split)**

The attribute that **reduces entropy the most** is the best to split on.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Compute E(S) — entropy before split |
| 2 | For each value v of attribute A: compute E(Sv) |
| 3 | Take weighted sum: Σ (|Sv|/|S|) × E(Sv) |
| 4 | Subtract: Gain = E(S) - weighted sum |
| 5 | Repeat for each attribute; pick highest gain |

### Example:

Dataset: 14 examples, 9 Yes, 5 No → E(S) = 0.94

**Splitting on Outlook:**
- Sunny: 5 examples (2Y, 3N) → E(Sunny) = 0.971
- Overcast: 4 examples (4Y, 0N) → E(Overcast) = 0
- Rain: 5 examples (3Y, 2N) → E(Rain) = 0.971

**Weighted entropy:**
$$\frac{5}{14}(0.971) + \frac{4}{14}(0) + \frac{5}{14}(0.971) = 0.347 + 0 + 0.347 = 0.694$$

**Gain:**
$$\text{Gain}(S, \text{Outlook}) = 0.940 - 0.694 = 0.246$$

### Why Maximize Information Gain?

- High gain means we **learn a lot** about the class by knowing the attribute's value.
- It corresponds to a **purer** split.
- ID3 picks the highest-gain attribute at each node.

### Limitations:

- **Bias toward attributes with many values** (each value has fewer examples → spuriously high gain).
- **Solution:** Use **Gain Ratio** (covered in Q5).

---


## Q18. NUMERICAL: PlayTennis Entropy and Information Gain (10 Marks)

### Given Training Data:

| Day | Outlook | Temperature | Humidity | Wind | PlayTennis |
|-----|---------|-------------|----------|------|------------|
| D1 | Sunny | Hot | High | Weak | No |
| D2 | Sunny | Hot | High | Strong | No |
| D3 | Overcast | Hot | High | Weak | Yes |
| D4 | Rain | Mild | High | Weak | Yes |
| D5 | Rain | Cool | Normal | Weak | Yes |
| D6 | Rain | Cool | Normal | Strong | No |
| D7 | Overcast | Cool | Normal | Strong | Yes |
| D8 | Sunny | Mild | High | Weak | No |
| D9 | Sunny | Cool | Normal | Weak | Yes |
| D10 | Rain | Mild | Normal | Weak | Yes |
| D11 | Sunny | Mild | Normal | Strong | Yes |
| D12 | Overcast | Mild | High | Strong | Yes |
| D13 | Overcast | Hot | Normal | Weak | Yes |
| D14 | Rain | Mild | High | Strong | No |

### Part (i): Calculate Entropy of S

**Count classes:**
- PlayTennis = Yes: D3, D4, D5, D7, D9, D10, D11, D12, D13 → **9 examples**
- PlayTennis = No: D1, D2, D6, D8, D14 → **5 examples**
- Total = 14

**Proportions:**
- $p_+ = 9/14 = 0.6429$
- $p_- = 5/14 = 0.3571$

**Apply Entropy Formula:**
$$E(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-)$$
$$= -\frac{9}{14}\log_2\left(\frac{9}{14}\right) - \frac{5}{14}\log_2\left(\frac{5}{14}\right)$$
$$= -0.6429 \times (-0.6374) - 0.3571 \times (-1.4854)$$
$$= 0.4098 + 0.5305$$
$$= \boxed{0.940}$$

### Part (ii-a): Information Gain for Outlook

**Step 1: Split data by Outlook**

| Outlook | Days | Yes | No | Total |
|---------|------|-----|-----|-------|
| Sunny | D1, D2, D8, D9, D11 | 2 (D9, D11) | 3 (D1, D2, D8) | 5 |
| Overcast | D3, D7, D12, D13 | 4 | 0 | 4 |
| Rain | D4, D5, D6, D10, D14 | 3 (D4, D5, D10) | 2 (D6, D14) | 5 |

**Step 2: Compute entropy of each subset**

**E(Sunny):** 2 Yes, 3 No
$$E(\text{Sunny}) = -\frac{2}{5}\log_2(0.4) - \frac{3}{5}\log_2(0.6)$$
$$= 0.5288 + 0.4422 = 0.971$$

**E(Overcast):** 4 Yes, 0 No (pure)
$$E(\text{Overcast}) = 0$$

**E(Rain):** 3 Yes, 2 No
$$E(\text{Rain}) = -\frac{3}{5}\log_2(0.6) - \frac{2}{5}\log_2(0.4)$$
$$= 0.4422 + 0.5288 = 0.971$$

**Step 3: Compute weighted sum**
$$\text{Weighted} = \frac{5}{14}(0.971) + \frac{4}{14}(0) + \frac{5}{14}(0.971)$$
$$= 0.347 + 0 + 0.347 = 0.694$$

**Step 4: Information Gain**
$$\text{Gain}(S, \text{Outlook}) = E(S) - \text{Weighted}$$
$$= 0.940 - 0.694 = \boxed{0.246}$$

### Part (ii-b): Information Gain for Humidity

**Step 1: Split data by Humidity**

| Humidity | Days | Yes | No | Total |
|----------|------|-----|-----|-------|
| High | D1, D2, D3, D4, D8, D12, D14 | 3 (D3, D4, D12) | 4 (D1, D2, D8, D14) | 7 |
| Normal | D5, D6, D7, D9, D10, D11, D13 | 6 | 1 (D6) | 7 |

**Step 2: Compute entropy of each subset**

**E(High):** 3 Yes, 4 No
$$E(\text{High}) = -\frac{3}{7}\log_2(3/7) - \frac{4}{7}\log_2(4/7)$$
$$= -0.429 \times (-1.222) - 0.571 \times (-0.807)$$
$$= 0.524 + 0.461 = 0.985$$

**E(Normal):** 6 Yes, 1 No
$$E(\text{Normal}) = -\frac{6}{7}\log_2(6/7) - \frac{1}{7}\log_2(1/7)$$
$$= -0.857 \times (-0.222) - 0.143 \times (-2.807)$$
$$= 0.190 + 0.401 = 0.592$$

**Step 3: Compute weighted sum**
$$\text{Weighted} = \frac{7}{14}(0.985) + \frac{7}{14}(0.592)$$
$$= 0.493 + 0.296 = 0.789$$

**Step 4: Information Gain**
$$\text{Gain}(S, \text{Humidity}) = 0.940 - 0.789 = \boxed{0.151}$$

### Final Answer Summary:

| Quantity | Value |
|----------|-------|
| Entropy(S) | **0.940** |
| Gain(S, Outlook) | **0.246** |
| Gain(S, Humidity) | **0.151** |

**Conclusion:** Outlook has higher information gain than Humidity, so it's a better attribute for splitting at this stage.

### Step-by-Step Recap:

| Step | Action |
|------|--------|
| 1 | Count Yes/No in S | 9 Yes, 5 No |
| 2 | Compute E(S) | 0.940 |
| 3 | Split by attribute, count Yes/No in each branch |
| 4 | Compute entropy of each branch |
| 5 | Take weighted average of branch entropies |
| 6 | Gain = E(S) - weighted average |

---


## Q20. NUMERICAL: Discount Offered Problem (10 Marks)

### Given:

A dataset of 10 customers:
- 6 purchased, 4 did not
- **Discount = Yes:** 4 purchased, 1 did not (total 5)
- **Discount = No:** 2 purchased, 3 did not (total 5)

**Required:**
1. Compute entropy of dataset
2. Compute entropy after splitting on "Discount Offered"
3. Compute information gain
4. Determine if it's a good attribute for splitting

### Step 1: Entropy of Original Dataset (S)

**Total:** 10 examples (6 purchased, 4 not)

**Proportions:**
- $p_{\text{purchased}} = 6/10 = 0.6$
- $p_{\text{not}} = 4/10 = 0.4$

**Apply Entropy Formula:**
$$E(S) = -0.6 \log_2(0.6) - 0.4 \log_2(0.4)$$
$$= -0.6 \times (-0.737) - 0.4 \times (-1.322)$$
$$= 0.442 + 0.529$$
$$= \boxed{0.971}$$

### Step 2: Entropy of Each Subset After Split

**Subset 1: Discount = Yes (4 purchased, 1 not, total 5)**

$$E(\text{Discount=Yes}) = -\frac{4}{5}\log_2(0.8) - \frac{1}{5}\log_2(0.2)$$
$$= -0.8 \times (-0.322) - 0.2 \times (-2.322)$$
$$= 0.258 + 0.464$$
$$= 0.722$$

**Subset 2: Discount = No (2 purchased, 3 not, total 5)**

$$E(\text{Discount=No}) = -\frac{2}{5}\log_2(0.4) - \frac{3}{5}\log_2(0.6)$$
$$= -0.4 \times (-1.322) - 0.6 \times (-0.737)$$
$$= 0.529 + 0.442$$
$$= 0.971$$

### Step 3: Weighted Entropy After Split

$$E_{\text{after}} = \frac{|S_{\text{Yes}}|}{|S|} E(S_{\text{Yes}}) + \frac{|S_{\text{No}}|}{|S|} E(S_{\text{No}})$$

$$= \frac{5}{10}(0.722) + \frac{5}{10}(0.971)$$
$$= 0.361 + 0.486$$
$$= \boxed{0.847}$$

### Step 4: Information Gain

$$\text{Gain}(S, \text{Discount}) = E(S) - E_{\text{after}}$$
$$= 0.971 - 0.847$$
$$= \boxed{0.124}$$

### Step 5: Is "Discount Offered" a Good Attribute?

**Analysis:**
- Information Gain = **0.124** (positive but small)
- This means splitting on Discount **does** reduce uncertainty, but not by much.
- Compare with other attributes (if available); pick the highest gain.

**Verdict:**
- The gain is **modest**, suggesting Discount has some predictive value but isn't strong.
- If no better attributes exist, it can be used; otherwise, prefer attributes with higher gain.
- **It IS an appropriate attribute for splitting** (positive gain), but not a particularly strong one.

### Final Answer Summary:

| Quantity | Value |
|----------|-------|
| Entropy(S) | **0.971** |
| Entropy(Discount=Yes) | **0.722** |
| Entropy(Discount=No) | **0.971** |
| Weighted entropy after split | **0.847** |
| Information Gain | **0.124** |

### Step-by-Step Recap:

| Step | Action | Result |
|------|--------|--------|
| 1 | Count yes/no in dataset → compute E(S) | 0.971 |
| 2 | Split on attribute, count in each branch | 4/1 and 2/3 |
| 3 | Compute entropy of each branch | 0.722, 0.971 |
| 4 | Compute weighted average | 0.847 |
| 5 | Gain = E(S) - weighted average | **0.124** |

---


# SECTION D: ISSUES IN DECISION TREE LEARNING

## Q4. Explain Reduced Error Pruning with a Suitable Example. (5 Marks)

### What is Pruning?

**Pruning** is the technique of **removing parts of a decision tree** that don't significantly improve accuracy on validation data. It helps prevent **overfitting**.

### What is Reduced Error Pruning?

**Reduced Error Pruning (REP)** is a simple post-pruning technique:
1. **Build the full decision tree** on training data.
2. **Split data** into training and validation sets.
3. For each non-leaf node:
   - Try **removing it** (replacing it with the most common class in its subtree).
   - Check accuracy on validation set.
4. **Keep the change** only if accuracy doesn't decrease.
5. **Repeat** until no more pruning improves accuracy.

### Algorithm:

```
1. Train full decision tree T on training data
2. While accuracy on validation set keeps improving:
       For each internal node n in T:
           a. Temporarily replace n with a leaf labeled by majority class
           b. Compute new accuracy on validation set
       Choose the pruning that gives best accuracy improvement
       Apply that pruning to T
3. Return pruned tree T
```

### Example:

**Original tree (overfitted):**
```
            Outlook
         /     |     \
       Sunny  Over   Rain
        |     cast    |
      Humidity Yes   Wind
       /  \         /  \
     High Norm   Strong Weak
      |    |      |     |
      No  Temp    No   Yes
          / \
        Hot  Cool
        |     |
       Yes   No  ← These extra splits may overfit!
```

After **REP**, the deepest splits (Temperature) might be removed if they don't help validation accuracy:

```
            Outlook
         /     |     \
       Sunny  Over   Rain
        |     cast    |
      Humidity Yes   Wind
       /  \         /  \
     High Norm   Strong Weak
      |    |      |     |
      No  Yes     No   Yes
```

### Advantages of REP:

1. **Simple to implement.**
2. **Effective at reducing overfitting.**
3. **Produces smaller, more interpretable trees.**

### Disadvantages:

1. **Requires a validation set** (reduces training data).
2. **May over-prune** if validation set is small.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Build full tree on training data |
| 2 | Reserve some examples as validation set |
| 3 | Try replacing each subtree with majority leaf |
| 4 | Keep the replacement only if validation accuracy increases (or stays same) |
| 5 | Repeat until no improvement possible |

---

## Q5. Describe Alternative Measure: Gain Ratio. (5 Marks)

### Problem with Information Gain:

Information Gain has a **bias toward attributes with many values**.

**Example:** If we have an attribute "Date" (365 values), each value has very few examples → each subset is pure → high gain. But "Date" is useless for prediction!

### Solution: Gain Ratio

**Gain Ratio** normalizes Information Gain by penalizing attributes with many values.

### Formulas:

**Step 1: Split Information**
$$\text{SplitInfo}(S, A) = -\sum_{i=1}^{c} \frac{|S_i|}{|S|} \log_2 \frac{|S_i|}{|S|}$$

This measures the **breadth** of the split (how evenly examples are distributed across branches).

**Step 2: Gain Ratio**
$$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$

### Intuition:

- **Many balanced branches** → high SplitInfo → reduces Gain Ratio.
- **Few well-chosen branches** → low SplitInfo → maintains Gain Ratio.

This penalizes "useless" attributes that just split data into many small groups.

### Example:

**Outlook split:** Sunny (5), Overcast (4), Rain (5) — fairly balanced.

$$\text{SplitInfo}(S, \text{Outlook}) = -\frac{5}{14}\log_2\frac{5}{14} - \frac{4}{14}\log_2\frac{4}{14} - \frac{5}{14}\log_2\frac{5}{14}$$
$$\approx -0.357(-1.485) - 0.286(-1.807) - 0.357(-1.485)$$
$$\approx 0.530 + 0.517 + 0.530 = 1.577$$

**Gain Ratio:**
$$\text{GainRatio}(S, \text{Outlook}) = \frac{0.246}{1.577} = 0.156$$

### When to Use Gain Ratio:

- When you have attributes with many values (e.g., date, ID).
- C4.5 algorithm uses Gain Ratio (improvement over ID3).

### Caveat:

If SplitInfo is very small (one branch dominates), Gain Ratio can be **artificially inflated**. C4.5 handles this by using Gain Ratio only when Information Gain is above the average.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Compute Information Gain |
| 2 | Compute SplitInfo (entropy of branch sizes) |
| 3 | Divide: Gain / SplitInfo |
| 4 | Pick attribute with highest Gain Ratio |

---


## Q6. Handling Training Examples with Missing Attribute Values. (5 Marks)

### The Problem:

Real-world data often has **missing values** for some attributes. We can't just discard these examples (would lose info). How do we handle them?

### Approach 1: Assign Most Common Value

For attribute A with a missing value:
- Look at all other examples in the same branch.
- Use the **most common value of A** among them.

**Example:** If "Wind" is missing for an example, but among Yes-class examples in the same branch, "Strong" is the most common, fill in "Strong".

### Approach 2: Most Common Value Among Same-Class Examples

More refined version of Approach 1:
- Find the **most common value of A** among examples with the **same class** as the one with the missing value.

### Approach 3: Probability-Based (Fractional Examples)

Instead of picking ONE value, **assign fractional weights**:
- If attribute A can be Strong (60%) or Weak (40%) among similar examples...
- Send 0.6 of this example down "Strong" branch.
- Send 0.4 down "Weak" branch.

This is what C4.5 algorithm does.

### Approach 4: At Classification Time

When classifying a new example with missing values:
- Compute the probability of each class along all possible paths.
- Output the most probable class.

### Comparison:

| Approach | Pros | Cons |
|----------|------|------|
| 1. Most common | Simple | Loses info about specific case |
| 2. Same-class | Better than 1 | Still a single guess |
| 3. Fractional | Most accurate | More complex |
| 4. Probability at test time | Handles missing in queries | Slow inference |

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Identify examples with missing values |
| 2 | For each missing value, choose handling strategy |
| 3 | Apply strategy: replace value or use fractional weights |
| 4 | Continue normal tree building |

### Used By:

- **ID3:** Discards examples with missing values (basic).
- **C4.5:** Uses fractional examples (more sophisticated).

---

## Q7. Handling Attributes with Different Costs. (5 Marks)

### The Problem:

In some domains (especially medical), different attributes have different **costs** to obtain:
- Blood pressure: cheap, fast.
- MRI scan: expensive, slow.

We want trees that **prefer cheap attributes** when possible.

### Solution: Modify Information Gain to Penalize Cost

Two approaches:

#### Approach 1: Tan and Schlimmer Formula
$$\text{Gain}_{\text{cost}}(S, A) = \frac{\text{Gain}^2(S, A)}{\text{Cost}(A)}$$

Squaring the gain emphasizes accuracy; division by cost penalizes expensive attributes.

#### Approach 2: Nunez Formula
$$\text{Gain}_{\text{cost}}(S, A) = \frac{2^{\text{Gain}(S, A)} - 1}{(\text{Cost}(A) + 1)^w}$$

Where w ∈ [0, 1] adjusts cost importance.

### Example:

Attribute X has Gain = 0.5, Cost = 1
Attribute Y has Gain = 0.6, Cost = 10

**Without cost:** Y is preferred (higher gain).

**With cost (Tan formula):**
- $\text{Gain}_{\text{cost}}(X) = 0.5^2 / 1 = 0.25$
- $\text{Gain}_{\text{cost}}(Y) = 0.6^2 / 10 = 0.036$

**With cost:** X is preferred (cheaper despite slightly lower gain).

### Why Cost Matters:

- **Medical:** Avoid unnecessary expensive tests.
- **Industrial:** Reduce sensor costs.
- **Real-time systems:** Faster decisions.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Compute information gain for each attribute |
| 2 | Look up cost of each attribute |
| 3 | Apply cost-modified gain formula |
| 4 | Pick attribute with highest cost-adjusted gain |

### Trade-off:

This makes trees **slightly less accurate** but **more practical** in cost-sensitive domains.

---

## Q8. Handling Continuous-Valued Attributes. (5 Marks)

### The Problem:

ID3 originally handles only **discrete attributes** (e.g., Sunny/Overcast/Rain). What if we have a continuous attribute like **Temperature** (any real number)?

### Solution: Threshold-Based Splits

Convert continuous attribute into a **binary attribute** using a threshold:
- $A < \text{threshold}$ → True
- $A \geq \text{threshold}$ → False

### How to Find the Best Threshold?

**Step 1:** Sort the examples by the value of the continuous attribute.

**Step 2:** Identify candidate thresholds — the **midpoints** between consecutive values where the class changes.

**Example:**

| Temperature | PlayTennis |
|-------------|------------|
| 40 | No |
| 48 | No |
| 60 | Yes |
| 72 | Yes |
| 80 | Yes |
| 90 | No |

**Class changes** between:
- 48 (No) and 60 (Yes) → midpoint = 54
- 80 (Yes) and 90 (No) → midpoint = 85

**Candidate thresholds:** 54, 85

**Step 3:** For each candidate threshold, compute **information gain** as if it were a binary attribute.

**Step 4:** Pick the threshold with highest gain.

### Example Calculation:

**Threshold = 54:**
- Temp < 54: 2 examples (both No) → Entropy = 0
- Temp ≥ 54: 4 examples (3 Yes, 1 No) → Entropy = 0.811
- Weighted entropy = (2/6)(0) + (4/6)(0.811) = 0.541

**Threshold = 85:**
- Temp < 85: 5 examples (3 Yes, 2 No) → Entropy = 0.971
- Temp ≥ 85: 1 example (No) → Entropy = 0
- Weighted entropy = (5/6)(0.971) + (1/6)(0) = 0.809

**Choose threshold = 54** (lower weighted entropy = higher gain).

### Multiple Splits:

For more refined trees, you can split into 3+ ranges instead of binary. But binary is more common (simpler).

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Sort examples by continuous attribute |
| 2 | Find midpoints where class changes |
| 3 | For each midpoint, compute gain assuming binary split |
| 4 | Pick midpoint with highest gain as threshold |
| 5 | Use this threshold to split data |

### Used By:

- **C4.5** algorithm extends ID3 to handle continuous attributes this way.

---


## Q14. Three Main Advantages of Converting Decision Tree to Rules Before Pruning. (6 Marks)

### Background:

After building a decision tree, we can convert it into a set of **IF-THEN rules** (one rule per root-to-leaf path). Pruning these rules instead of pruning the tree directly has advantages.

### Advantages:

#### 1. Allows Distinguishing Between Different Contexts of an Attribute

Each rule represents a unique path. Pruning one rule doesn't affect others.

**Example:**
- Rule 1: IF Outlook=Sunny AND Humidity=High THEN No
- Rule 2: IF Outlook=Rain AND Wind=Strong THEN No

If we discover that "Humidity" is irrelevant in Rule 1, we can prune just that condition WITHOUT affecting Rule 2 (which doesn't use Humidity).

In tree pruning, removing the Humidity node would affect ALL rules going through it.

#### 2. Removes the Distinction Between Tests Near vs. Far From Root

In a tree, a test near the root affects ALL descendant rules. After conversion to rules:
- Each rule is independent.
- We can prune any condition (regardless of position).

**Example:** If "Outlook=Sunny" is the root test, removing it means all paths through it are affected. As rules, each rule with "Outlook=Sunny" can be pruned independently.

#### 3. Improves Readability

Rules are easier to understand than trees:
- People naturally read IF-THEN statements.
- Rules can be ordered by importance.
- Each rule stands alone.

### Process: Rule Post-Pruning (C4.5)

```
1. Build full tree
2. Convert tree to rules (one rule per leaf)
3. For each rule:
       For each condition in the rule:
           Try removing the condition
           Estimate accuracy with and without
           Remove if accuracy doesn't decrease
4. Sort rules by estimated accuracy
5. Use rules in order for classification
```

### Example:

**Original Rule:**  
IF (Outlook=Sunny) AND (Humidity=High) AND (Wind=Strong) THEN PlayTennis=No

**Try removing "Wind=Strong":**  
IF (Outlook=Sunny) AND (Humidity=High) THEN PlayTennis=No

If accuracy on validation doesn't decrease, **remove "Wind=Strong"**.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Build full decision tree |
| 2 | Convert to IF-THEN rules (one per leaf) |
| 3 | For each rule, test removing each condition |
| 4 | Keep removal if accuracy is maintained |
| 5 | Sort and use the simplified rules |

### Why Tree Pruning is Inferior:

- Tree pruning forces us to remove **whole subtrees** at once.
- Rule pruning lets us **fine-tune individual conditions**.
- Result: rule-based pruning is **more flexible and precise**.

---

## Q15. Describe Rule Post-Pruning to Avoid Overfitting. (6 Marks)

### Definition:

**Rule Post-Pruning** is a technique that:
1. Builds the full decision tree.
2. Converts it to rules.
3. Prunes individual conditions from rules to improve generalization.

It's used by the **C4.5 algorithm** and is often more effective than tree-based pruning.

### Algorithm:

```
1. Build the decision tree from training data
   (allow it to overfit; that's OK for now)

2. Convert each path from root to leaf into an IF-THEN rule
   - Each test along the path becomes a condition
   - The leaf label becomes the consequent

3. For each rule:
   For each condition (precondition) in the rule:
       a. Tentatively remove the condition
       b. Estimate the rule's accuracy on validation set
       c. If accuracy improves or stays the same, remove the condition

4. Sort rules by their estimated accuracy

5. Use the rules in this sorted order to classify new instances
   (Default to majority class if no rule matches)
```

### Example:

**Initial Tree (overfitted):**

```
            Outlook
         /     |     \
       Sunny  Over   Rain
        |     cast    |
      Humidity Yes   Wind
       /  \         /  \
     High Norm   Strong Weak
      |    |      |     |
      No  Yes     No   Yes
```

**Convert to Rules:**
1. IF (Outlook=Sunny) AND (Humidity=High) THEN No
2. IF (Outlook=Sunny) AND (Humidity=Normal) THEN Yes
3. IF (Outlook=Overcast) THEN Yes
4. IF (Outlook=Rain) AND (Wind=Strong) THEN No
5. IF (Outlook=Rain) AND (Wind=Weak) THEN Yes

**Pruning Step:** For each rule, test removing each condition:

**Rule 1:** Try removing "Outlook=Sunny" → IF (Humidity=High) THEN No  
- Check: does this still classify correctly? If yes, keep simpler rule.

**Rule 4:** Try removing "Wind=Strong" → IF (Outlook=Rain) THEN No  
- May not work; some Rain examples are Yes.

**After pruning:** Some rules may become simpler, others stay the same.

### Advantages:

1. **More flexibility** than tree pruning (each rule is independent).
2. **Improves readability** of the model.
3. **Reduces overfitting** effectively.

### Disadvantages:

1. **Computationally expensive** (test removing each condition).
2. **Order of rules matters** for classification.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Build full decision tree |
| 2 | Convert to IF-THEN rules |
| 3 | For each rule, try removing each condition |
| 4 | Keep removal if accuracy doesn't decrease |
| 5 | Sort rules by accuracy |
| 6 | Use sorted rules for prediction |

### Difference from Reduced Error Pruning:

| Aspect | REP | Rule Post-Pruning |
|--------|-----|-------------------|
| Operates on | Tree subtrees | Individual conditions |
| Granularity | Coarse | Fine |
| Flexibility | Limited | High |
| Used by | Simple algorithms | C4.5 |

---


# SECTION E: BIAS AND THEORETICAL CONCEPTS

## Q3. Explain Restriction Biases and Preference Biases in Machine Learning. (5 Marks)

### Definition: Inductive Bias

**Inductive bias** is the set of assumptions a learning algorithm makes to generalize from training data to unseen examples.

There are **two types** of inductive bias:

### 1. Restriction Bias (Language Bias / Hypothesis Bias)

The learner is **restricted to a limited set of hypotheses** — it cannot consider hypotheses outside this set.

**Example:** 
- Linear regression only considers **straight-line** hypotheses.
- It can't represent curves like y = x² unless extended.

**Effect:** The hypothesis space H is restricted to a subset of all possible functions.

### 2. Preference Bias (Search Bias)

The learner can consider ALL hypotheses, but **prefers some over others** based on some criterion.

**Example:**
- ID3 considers all decision trees but **prefers shorter trees** (occam's razor).
- All trees are reachable, but ID3's greedy search favors compact ones.

**Effect:** Within the full hypothesis space, certain hypotheses are favored.

### Comparison:

| Aspect | Restriction Bias | Preference Bias |
|--------|------------------|-----------------|
| **Hypothesis space** | Limited to a subset | Full space available |
| **Mechanism** | Cannot represent some hypotheses | Prefers certain hypotheses |
| **Example algorithm** | Linear regression, perceptron | ID3, neural networks (with regularization) |
| **Sometimes called** | Language bias | Search bias |
| **Risk** | May miss the true hypothesis | May find suboptimal |

### Example Comparison:

**Restriction:** Decision tree with only 2 attributes — cannot represent functions of 3+ attributes.

**Preference:** ID3 builds full trees but prefers attributes with high information gain.

### Why Bias Matters:

- **Without bias, no learning is possible** — algorithms must make some assumptions to generalize.
- **Different biases** suit different problems.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Identify learning algorithm |
| 2 | Determine which hypotheses it can represent (restriction) |
| 3 | Determine which it prefers (preference) |
| 4 | Match bias to problem requirements |

### Which is Better?

- **Preference bias** is generally **preferred** because it's more flexible.
- **Restriction bias** can completely miss the right answer if true function is outside the restricted space.

### Examples in Module 3:

- **ID3:** Has **preference bias** for shorter trees and high-gain attributes near the root.
- **Candidate-Elimination:** Has **restriction bias** (only considers conjunctions of constraints).

---

## Q19. Inductive Bias in Decision Tree Learning with Suitable Example. (10 Marks)

### Definition Recap:

**Inductive bias** = assumptions a learning algorithm uses to generalize beyond the training data.

### Inductive Bias of ID3 (Decision Tree Learning):

ID3's inductive bias can be approximated as:

> "**Shorter trees are preferred over longer trees.**  
> **Trees that place high-information-gain attributes close to the root are preferred over those that don't.**"

This is a **preference bias** (not restriction bias) because:
- ID3 can build trees of any size.
- It just prefers certain trees due to its greedy algorithm.

### Why Does ID3 Prefer Shorter Trees?

This relates to **Occam's Razor**:
> "Among competing hypotheses, the simplest one is most likely to be correct."

ID3 uses information gain greedily, which tends to:
1. Place the most informative attributes near the root.
2. Reach pure leaves quickly.
3. Result in shallower (shorter) trees.

### Example:

**Suppose we have 4 attributes: A, B, C, D and want to learn:**  
target = A AND B (only A and B matter)

**Two trees can represent this:**

**Tree 1 (short):**
```
        A
      ┌─┴─┐
     T     F
     │     │
     B     0
   ┌─┴─┐
  T     F
  │     │
  1     0
```

**Tree 2 (long, with irrelevant attributes):**
```
        A
      ┌─┴─┐
     T     F
     │     │
     B     C
   ┌─┴─┐ ┌─┴─┐
  T     F T     F
  │     │ │     │
  C     0 D     0
 ┌┴┐    ┌┴┐
 T F    T F
 │ │    │ │
 1 0    0 0
```

Both are consistent with training data (assuming C, D didn't change anything). But Tree 1 is **simpler** and **generalizes better**.

ID3's bias **prefers Tree 1**.

### Restriction vs Preference Bias in Decision Trees:

| Algorithm | Type of Bias | Description |
|-----------|--------------|-------------|
| **ID3** | Preference | Can represent any function; prefers short trees |
| **Decision tree limited to depth 3** | Restriction | Cannot represent deeper functions |
| **Candidate-Elimination** | Restriction | Only conjunctions of constraints |

### Why ID3's Bias is Useful:

1. **Avoids overfitting:** Shorter trees generalize better.
2. **Computationally efficient:** Greedy search is fast.
3. **Interpretable:** Smaller trees are easier to understand.

### Connection to Occam's Razor:

Occam's Razor says: prefer simpler explanations. ID3's bias **mathematically captures** this by preferring trees with high-gain attributes (which lead to faster purity, hence shorter trees).

### Limitations:

1. **Greedy algorithm** doesn't always find the SHORTEST tree (just a short one).
2. **Bias can be wrong** if the true function actually needs a complex tree.

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Recognize that bias is necessary for generalization |
| 2 | Identify ID3's bias: prefer short trees, high-gain attributes near root |
| 3 | Note this is preference (not restriction) bias |
| 4 | Connect to Occam's Razor |
| 5 | Use this bias to justify why ID3 generalizes well |

### Final Insight:

ID3's bias means it's looking for **simple, generalizable patterns** rather than complex memorization of training data. This is why decision trees often work well in practice.

---

# UNIVERSAL EXAM CHEATSHEET (Last-Minute Revision)

## Common Procedure for ALL Entropy/Gain Questions:

```
Step 1: Count classes in dataset
        Compute Entropy(S) = -p+ log2(p+) - p- log2(p-)

Step 2: For each attribute A:
        a. Split data by A's values
        b. Compute Entropy of each subset
        c. Compute weighted avg: Σ (|Sv|/|S|) × Entropy(Sv)
        d. Information Gain = Entropy(S) - weighted avg

Step 3: Pick attribute with highest gain (this is the root)
```

## Quick Comparison: All Numerical Answers

| Question | Type | Method | Answer |
|----------|------|--------|--------|
| Q18 (i) | Entropy | -9/14*log(9/14) - 5/14*log(5/14) | **0.940** |
| Q18 (ii-a) | Gain Outlook | E(S) - weighted | **0.246** |
| Q18 (ii-b) | Gain Humidity | E(S) - weighted | **0.151** |
| Q20 | Discount problem | Full calculation | E=0.971, Gain=**0.124** |

---

# IMPORTANT POINTS TO REMEMBER FOR EXAM

## 1. Always show:
- Class counts in each subset
- Entropy formula with substituted values
- Step-by-step calculation
- Final boxed answer

## 2. Common mistakes to avoid:
- ❌ Using natural log (ln) instead of log₂
- ❌ Forgetting the negative sign in entropy
- ❌ Computing log of 0 (handle pure subsets specially: entropy = 0)
- ❌ Forgetting to weight by |Sv|/|S|

## 3. Tips for Theory Questions:
- **Always include a diagram** for tree-based questions
- **Give an example** even if not asked
- **List advantages/disadvantages** to fill answer
- **End with step-by-step procedure** in a table

## 4. Decision-making in numericals:
- For entropy of pure subset (all same class): **Entropy = 0**
- For balanced binary subset (50/50): **Entropy = 1**
- For information gain: pick attribute with **highest gain**

---

# FINAL TIPS FOR THE EXAM

1. **Memorize ONLY the master formulas** — Entropy and Information Gain cover everything.

2. **Practice the calculation pattern**:
   - Count → Entropy → Weighted avg → Gain

3. **Show ALL steps** — examiners give marks for method.

4. **For theoretical questions**:
   - Definition → Formula → Example → Pros/Cons → Step-by-step

5. **For numerical questions**:
   - State given data → Identify formula → Substitute → Calculate → Box answer

6. **Time management**: 
   - 5-mark questions: 8-10 minutes
   - 6-mark questions: 12 minutes
   - 8-mark questions: 15 minutes
   - 10-mark questions: 18-20 minutes

---

## All The Best for Your Exam! 🎯

Remember: **Decision Tree Learning is just three steps:**
1. **Compute entropy** (how mixed is the data?)
2. **Compute information gain** (which attribute reduces mixing the most?)
3. **Split on best attribute** (recursively until pure)

That's the entire module! Once you understand these 3 steps, all 20 questions become variations of the same idea.
