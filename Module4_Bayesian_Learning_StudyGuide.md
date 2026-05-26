# Module 4: Bayesian Learning - Complete Exam Study Guide

> **For 3rd Year BE Students | VTU/Karnataka Curriculum**
> **Topic:** Bayesian Learning, Naive Bayes, Bayesian Networks, EM Algorithm

---

## How This Guide is Organized

I have grouped the 20 questions into **5 logical sections**. Once you understand one question in a section, the others become easy because they use the **same core formula** with different numbers.

| Section | Topic | Questions |
|---------|-------|-----------|
| **A** | Bayes Theorem & Theoretical Foundations | Q1, Q2, Q9, Q10, Q12, Q13, Q14 |
| **B** | Bayesian Belief Networks | Q3, Q4, Q5, Q6, Q7 |
| **C** | Naive Bayes Classifier (Numerical) | Q16, Q18, Q20 |
| **D** | EM Algorithm | Q8, Q17 |
| **E** | MAP / Posterior Probability Numericals | Q11, Q15, Q19 |

---

# THE MASTER FORMULA SHEET (Memorize These!)

You only need to remember these few formulas to crack this entire module:

### 1. Bayes Theorem (THE most important formula)
$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

Where:
- **P(h|D)** = Posterior probability (probability of hypothesis h given data D)
- **P(D|h)** = Likelihood (probability of seeing data D if h is true)
- **P(h)** = Prior probability (our initial belief in h)
- **P(D)** = Evidence (total probability of data)

### 2. Total Probability (used to find P(D))
$$P(D) = \sum_{i} P(D|h_i) \cdot P(h_i)$$

### 3. MAP Hypothesis (Maximum A Posteriori)
$$h_{MAP} = \arg\max_{h \in H} P(D|h) \cdot P(h)$$
*(We drop P(D) because it's same for all hypotheses)*

### 4. ML Hypothesis (Maximum Likelihood)
$$h_{ML} = \arg\max_{h \in H} P(D|h)$$
*(Used when all priors P(h) are equal)*

### 5. Naive Bayes Classifier
$$v_{NB} = \arg\max_{v_j \in V} P(v_j) \prod_{i} P(a_i | v_j)$$

### 6. Joint Probability in Bayesian Network
$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | Parents(x_i))$$

### 7. MDL (Minimum Description Length)
$$h_{MDL} = \arg\min_{h \in H} [L_{C_1}(h) + L_{C_2}(D|h)]$$

---

# SECTION A: BAYES THEOREM & THEORETICAL FOUNDATIONS

## Q1. Explain Bayes Theorem. Give the relevance of Bayesian learning methods. (5 Marks)

### Answer:

**Bayes Theorem** is a fundamental theorem in probability that describes how to **update our belief** about a hypothesis when we observe new evidence/data.

**Statement:**
$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

**Where:**
- **P(h)** = Prior probability of hypothesis h (initial belief before seeing data)
- **P(D)** = Prior probability of training data D (evidence)
- **P(h|D)** = Posterior probability of h given D (updated belief after seeing data)
- **P(D|h)** = Likelihood of D given h (probability of observing D if h is true)

**Intuition:** "Posterior = (Likelihood × Prior) / Evidence"

### Relevance of Bayesian Learning Methods:

1. **Probabilistic Predictions:** Each training example can incrementally increase or decrease the estimated probability of a hypothesis being correct.

2. **Combines Prior Knowledge:** Allows combining prior knowledge with observed data (P(h) and P(D|h) work together).

3. **Handles Uncertainty:** Provides probabilistic outputs rather than just true/false, useful in real-world noisy data.

4. **Provides Standard for Comparison:** Even when computationally hard, Bayesian methods give the **gold standard** to evaluate other learning algorithms.

5. **Multiple Hypotheses:** Can accommodate hypotheses that make probabilistic predictions (e.g., "patient has 93% chance of recovery").

6. **Foundation for Algorithms:** Naive Bayes, Bayesian Networks, EM algorithm — all derive from Bayes theorem.

### Step-by-Step Explanation:

| Step | Description |
|------|-------------|
| Step 1 | Start with **prior belief** P(h) about hypothesis h |
| Step 2 | Collect **data D** from observations |
| Step 3 | Compute **likelihood** P(D|h) — how well h explains D |
| Step 4 | Apply Bayes formula to get **posterior** P(h|D) |
| Step 5 | Use posterior to make decisions or update beliefs |

---

## Q2. Explain the features and practical difficulties of Bayes Theorem. (5 Marks)

### Features of Bayesian Learning:

1. **Incremental Learning:** Each new training example incrementally updates the probability estimate.

2. **Prior Knowledge Integration:** Combines prior probabilities P(h) with observed data P(D|h).

3. **Probabilistic Hypotheses:** Hypotheses make probabilistic predictions (not just yes/no).

4. **Multiple Hypothesis Combination:** New instances can be classified by combining predictions from multiple hypotheses (weighted by posteriors).

5. **Optimal Decision Making:** Provides the theoretically optimal classifier (Bayes Optimal Classifier).

6. **Robust to Noise:** Naturally handles noisy data because of probabilistic framework.

### Practical Difficulties:

1. **Requires Initial Knowledge of Many Probabilities:** Need to know P(h) and P(D|h) for ALL hypotheses, which often is not available.

2. **High Computational Cost:** Computing the Bayes optimal hypothesis requires evaluating ALL hypotheses in the hypothesis space H — computationally expensive when H is large.

3. **Estimating Priors is Hard:** When prior knowledge is unavailable, we must estimate priors from background knowledge, previous data, or assumptions about underlying distributions.

4. **Curse of Dimensionality:** As number of features grows, estimating P(D|h) becomes infeasible due to data sparsity.

5. **Conditional Probability Tables Explode:** In Bayesian networks, CPT size grows exponentially with number of parents.

### Step-by-Step:
- **Step 1:** Identify what makes Bayesian learning attractive (the features above)
- **Step 2:** Identify what makes it hard in practice (the difficulties)
- **Step 3:** Note that Naive Bayes and Bayesian Networks are practical solutions to overcome these difficulties

---

## Q9. Illustrate the role of Brute Force MAP Learning Algorithm. (10 Marks)

### Answer:

The **Brute Force MAP Learning Algorithm** is a straightforward method to find the most probable hypothesis using Bayes theorem by examining EVERY hypothesis in the hypothesis space H.

### Algorithm:

**Input:** Hypothesis space H, training data D

**Steps:**

1. For each hypothesis h in H, calculate the posterior probability:
$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

2. Output the hypothesis $h_{MAP}$ with the highest posterior probability:
$$h_{MAP} = \arg\max_{h \in H} P(h|D)$$

Since P(D) is constant for all h, we simplify to:
$$h_{MAP} = \arg\max_{h \in H} P(D|h) \cdot P(h)$$

### Role / Significance:

1. **Provides a Standard:** Even though impractical for large H, it gives the **theoretically optimal** answer that other algorithms can be compared against.

2. **Connects Bayesian and Concept Learning:** Shows that under specific assumptions:
   - Uniform prior P(h) over all hypotheses
   - Noise-free deterministic data
   
   The MAP hypothesis equals any consistent hypothesis (Version Space).

3. **Foundation for Approximate Methods:** Forms the basis for practical algorithms like Naive Bayes that approximate MAP.

4. **Useful Theoretical Tool:** Shows that algorithms like Find-S and Candidate-Elimination output MAP hypothesis under certain assumptions.

### Example Illustration:

Suppose we have 3 hypotheses h1, h2, h3 with:
- P(h1) = 0.4, P(D|h1) = 0.2
- P(h2) = 0.3, P(D|h2) = 0.6
- P(h3) = 0.3, P(D|h3) = 0.1

Compute P(h|D) ∝ P(D|h) × P(h):
- h1: 0.2 × 0.4 = 0.08
- h2: 0.6 × 0.3 = **0.18** ← Maximum
- h3: 0.1 × 0.3 = 0.03

**h_MAP = h2**

### Step-by-Step Procedure:

| Step | Action |
|------|--------|
| Step 1 | List all hypotheses h ∈ H |
| Step 2 | Assign/compute prior P(h) for each hypothesis |
| Step 3 | Compute likelihood P(D|h) for each h |
| Step 4 | Multiply: P(D|h) × P(h) for each h |
| Step 5 | Pick the hypothesis with maximum value → h_MAP |

### Limitations:
- Requires evaluating ALL hypotheses (often infeasible)
- Requires knowing all priors and likelihoods
- Practical only for small hypothesis spaces

---

## Q10. Define MAP and ML hypothesis. Derive the relation for h_MAP and h_ML. (10 Marks)

### Definitions:

**Maximum A Posteriori (MAP) Hypothesis:**
The hypothesis that has the **maximum posterior probability** given observed data D.

$$h_{MAP} = \arg\max_{h \in H} P(h|D)$$

**Maximum Likelihood (ML) Hypothesis:**
The hypothesis that **maximizes the likelihood** of the observed data D, assuming all hypotheses are equally probable a priori.

$$h_{ML} = \arg\max_{h \in H} P(D|h)$$

### Derivation of h_MAP using Bayes Theorem:

**Starting from Bayes theorem:**
$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

**Step 1:** Apply argmax to find best h:
$$h_{MAP} = \arg\max_{h \in H} P(h|D) = \arg\max_{h \in H} \frac{P(D|h) \cdot P(h)}{P(D)}$$

**Step 2:** Note that P(D) is independent of h (it's the same for all hypotheses), so we can drop it:
$$h_{MAP} = \arg\max_{h \in H} P(D|h) \cdot P(h)$$

### Derivation of h_ML from h_MAP:

**Step 3:** If we assume **all hypotheses are equally probable a priori**:
$$P(h_i) = P(h_j) \text{ for all } h_i, h_j \in H$$

**Step 4:** Then P(h) is constant and can also be dropped:
$$h_{ML} = \arg\max_{h \in H} P(D|h)$$

### Relation between h_MAP and h_ML:

$$\boxed{h_{ML} = h_{MAP} \text{ when } P(h_i) = P(h_j) \text{ for all hypotheses}}$$

**In words:** ML is a special case of MAP where all hypotheses are equally likely a priori.

### Step-by-Step Summary:

| Step | Formula | Reason |
|------|---------|--------|
| 1 | $P(h\|D) = \frac{P(D\|h) P(h)}{P(D)}$ | Bayes theorem |
| 2 | $h_{MAP} = \arg\max P(h\|D)$ | Definition of MAP |
| 3 | $h_{MAP} = \arg\max P(D\|h) P(h)$ | Drop P(D) (constant) |
| 4 | $h_{ML} = \arg\max P(D\|h)$ | Drop P(h) (uniform prior) |

---

## Q12. Describe Maximum Likelihood Hypotheses for Predicting Probabilities. (10 Marks)

### Answer:

In some learning problems, we don't want a hypothesis that outputs a class label. Instead, we want one that **predicts the probability** of an outcome (e.g., "What's the probability the patient survives surgery?").

### Setup:

- Training data: D = {(x₁, d₁), (x₂, d₂), ..., (xₘ, dₘ)}
- Where x_i is the instance, d_i ∈ {0, 1} is the observed outcome
- Goal: Find hypothesis h(x) that outputs P(outcome = 1 | x)

### Deriving the ML Hypothesis:

**Step 1:** Probability of observing single training example (x_i, d_i) given h:
$$P(d_i | x_i, h) = h(x_i)^{d_i} \cdot (1 - h(x_i))^{1-d_i}$$

This is because:
- If d_i = 1 → P = h(x_i)
- If d_i = 0 → P = 1 - h(x_i)

**Step 2:** Likelihood of all data (assuming independence):
$$P(D|h) = \prod_{i=1}^{m} h(x_i)^{d_i} (1-h(x_i))^{1-d_i}$$

**Step 3:** Apply ML principle:
$$h_{ML} = \arg\max_{h} \prod_{i=1}^{m} h(x_i)^{d_i} (1-h(x_i))^{1-d_i}$$

**Step 4:** Take log (logarithm is monotonic, so it doesn't change argmax):
$$h_{ML} = \arg\max_{h} \sum_{i=1}^{m} \left[ d_i \ln h(x_i) + (1-d_i) \ln(1-h(x_i)) \right]$$

**Step 5:** This is the **Cross-Entropy Loss** (negative). To minimize loss:
$$h_{ML} = \arg\min_{h} - \sum_{i=1}^{m} \left[ d_i \ln h(x_i) + (1-d_i) \ln(1-h(x_i)) \right]$$

### Interpretation:

- This is the **cross-entropy** function used in training neural networks for binary classification!
- Maximizing likelihood = Minimizing cross-entropy
- Used in **logistic regression** and **neural network training**

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Express probability of one training example using Bernoulli form |
| 2 | Multiply across all examples (independence) |
| 3 | Apply ML criterion (argmax) |
| 4 | Take logarithm to convert product → sum |
| 5 | Result: Cross-entropy minimization |

---

## Q13. Prove how ML can minimize squared error between actual and predicted output. (10 Marks)

### Answer:

We want to prove: **Under the assumption that target values have Gaussian noise, the Maximum Likelihood hypothesis is the one that minimizes the sum of squared errors.**

### Setup:

- Training instances: $\langle x_i, d_i \rangle$ where i = 1, 2, ..., m
- Each $d_i = f(x_i) + e_i$ where:
  - f(x_i) is the true target value
  - $e_i$ is random noise drawn from Normal distribution $N(0, \sigma^2)$
- We want hypothesis h(x) that approximates f(x)

### Derivation:

**Step 1:** Since $e_i \sim N(0, \sigma^2)$, we have:
$$d_i \sim N(h(x_i), \sigma^2)$$

(Because if h is correct, d_i is centered around h(x_i) with noise variance σ²)

**Step 2:** Probability of observing d_i given h:
$$p(d_i | h) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(d_i - h(x_i))^2}{2\sigma^2}\right)$$

**Step 3:** Likelihood of all data (assuming independent samples):
$$p(D|h) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(d_i - h(x_i))^2}{2\sigma^2}\right)$$

**Step 4:** Apply ML:
$$h_{ML} = \arg\max_{h} \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(d_i - h(x_i))^2}{2\sigma^2}\right)$$

**Step 5:** Take natural log (doesn't change argmax):
$$h_{ML} = \arg\max_{h} \sum_{i=1}^{m} \left[ \ln \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(d_i - h(x_i))^2}{2\sigma^2} \right]$$

**Step 6:** Drop constants (the $\ln$ term and $\frac{1}{2\sigma^2}$ are constants w.r.t. h):
$$h_{ML} = \arg\max_{h} \sum_{i=1}^{m} -(d_i - h(x_i))^2$$

**Step 7:** Maximizing negative = Minimizing positive:
$$\boxed{h_{ML} = \arg\min_{h} \sum_{i=1}^{m} (d_i - h(x_i))^2}$$

### Conclusion:

**The Maximum Likelihood hypothesis is exactly the one that MINIMIZES THE SUM OF SQUARED ERRORS** between observed values d_i and predicted values h(x_i).

This justifies using **least squares** in regression — it's not arbitrary, it's the ML solution under Gaussian noise!

### Step-by-Step Summary:

| Step | Action |
|------|--------|
| 1 | Assume noise is Gaussian (Normal) with mean 0 |
| 2 | Express d_i as Normal distribution around h(x_i) |
| 3 | Write likelihood as product of Gaussian densities |
| 4 | Apply ML (argmax) |
| 5 | Take log to simplify product |
| 6 | Drop constants independent of h |
| 7 | Convert max(-error²) → min(error²) |

---

## Q14. Describe Minimum Description Length (MDL) Principle. Obtain h_MDL equation. (10 Marks)

### Answer:

The **Minimum Description Length (MDL)** principle states: *Choose the hypothesis that minimizes the total length needed to encode (1) the hypothesis itself and (2) the data when using the hypothesis.*

It's based on **Occam's Razor**: "The simplest explanation is usually correct."

### Derivation from MAP:

**Step 1:** Start with MAP hypothesis:
$$h_{MAP} = \arg\max_{h} P(D|h) \cdot P(h)$$

**Step 2:** Take log₂ (doesn't change argmax):
$$h_{MAP} = \arg\max_{h} [\log_2 P(D|h) + \log_2 P(h)]$$

**Step 3:** Negate (max → min):
$$h_{MAP} = \arg\min_{h} [-\log_2 P(D|h) - \log_2 P(h)]$$

**Step 4:** Apply **Information Theory result**: The optimal code length to encode an event with probability p is $-\log_2 p$ bits (Shannon's theorem).

So:
- $-\log_2 P(h)$ = length of optimal encoding of hypothesis h, denoted $L_{C_1}(h)$
- $-\log_2 P(D|h)$ = length of optimal encoding of data D given h, denoted $L_{C_2}(D|h)$

**Step 5:** Substitute:
$$\boxed{h_{MDL} = \arg\min_{h \in H} [L_{C_1}(h) + L_{C_2}(D|h)]}$$

### Interpretation of Two Terms:

1. **$L_{C_1}(h)$** = Length of hypothesis description
   - Penalizes complex hypotheses
   - A simple decision tree has small length
   - A complex tree has larger length

2. **$L_{C_2}(D|h)$** = Length of data encoded using h
   - If h perfectly explains data → length is small
   - If h has errors → need to encode the errors → larger length

**Trade-off:** MDL chooses a hypothesis that balances **simplicity** (small $L_{C_1}$) with **fit to data** (small $L_{C_2}$).

### Connection to MAP:

$$h_{MDL} = h_{MAP}$$

When the encoding scheme uses optimal codes (Shannon optimal codes), MDL selects the same hypothesis as MAP.

### Step-by-Step:

| Step | Description |
|------|-------------|
| 1 | Start with h_MAP formula |
| 2 | Take logarithm |
| 3 | Negate to convert max into min |
| 4 | Use Shannon's theorem: optimal code length = -log₂(P) |
| 5 | Express as sum of two description lengths |

### Practical Use:
- Decision tree pruning (avoid overfitting)
- Model selection (choosing between simple and complex models)
- Avoiding overfitting in neural networks

---


# SECTION B: BAYESIAN BELIEF NETWORKS

> **All questions in this section use this ONE formula:**
> $$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | Parents(x_i))$$

## Q3. Explain Bayesian Belief Networks with an example. (5 Marks)

### Definition:

A **Bayesian Belief Network (BBN)** is a directed acyclic graph (DAG) that represents a set of random variables and their conditional dependencies using a graph structure.

It is a graphical model used to represent **joint probability distributions** in a compact way.

### Components of a BBN:

1. **Nodes:** Each node represents a random variable (e.g., Cancer, Smoker, X-Ray).

2. **Edges (Arrows):** Directed edges represent **causal/conditional dependencies** between variables. An arrow from A → B means A directly influences B.

3. **Conditional Probability Table (CPT):** Each node has a CPT that gives P(node | parents).

4. **Acyclic:** No cycles allowed (you can't go from a node back to itself).

### Example: "Burglary Alarm" Network

**Network Structure:**
```
   Burglary (B)         Earthquake (E)
        \                  /
         \                /
          v              v
            Alarm (A)
            /        \
           v          v
       JohnCalls   MaryCalls
       (J)          (M)
```

**Variables:**
- B = Burglary occurred
- E = Earthquake occurred
- A = Alarm rings
- J = John calls
- M = Mary calls

**Conditional Probability Tables:**

| P(B = T) | P(E = T) |
|----------|----------|
| 0.001    | 0.002    |

| B | E | P(A = T \| B, E) |
|---|---|------------------|
| T | T | 0.95 |
| T | F | 0.94 |
| F | T | 0.29 |
| F | F | 0.001 |

| A | P(J = T \| A) | P(M = T \| A) |
|---|---------------|---------------|
| T | 0.90 | 0.70 |
| F | 0.05 | 0.01 |

### Joint Probability Calculation:

$$P(B, E, A, J, M) = P(B) \cdot P(E) \cdot P(A|B,E) \cdot P(J|A) \cdot P(M|A)$$

### Why use BBN?

- **Compact representation:** Instead of storing all 2^n probabilities for n variables, we store small CPTs.
- **Captures domain knowledge** through graph structure.
- **Supports inference** (predict any variable given evidence).

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Identify random variables in the problem |
| 2 | Draw nodes for each variable |
| 3 | Add directed edges based on causal relationships |
| 4 | Specify CPT for each node |
| 5 | Use chain rule to compute joint probabilities |

---

## Q4. Explain Conditional Independence assumptions used by Bayesian Networks. (5 Marks)

### Definition of Conditional Independence:

Variable X is **conditionally independent** of Y given Z if:
$$P(X | Y, Z) = P(X | Z)$$

In words: Once we know Z, knowing Y gives us no extra information about X.

### Key Assumption in Bayesian Networks:

**Each node is conditionally independent of its non-descendants, given its parents.**

Mathematically:
$$P(X_i | \text{Parents}(X_i), \text{Non-descendants}(X_i)) = P(X_i | \text{Parents}(X_i))$$

### How This Simplifies Joint Probability:

**Without Bayesian network (chain rule):**
$$P(x_1, x_2, ..., x_n) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1,x_2) \cdot ... \cdot P(x_n|x_1,...,x_{n-1})$$

This needs $2^n - 1$ probabilities for n binary variables — exponential!

**With Bayesian network:**
$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | \text{Parents}(x_i))$$

This needs only the CPTs — much fewer probabilities!

### Example:

In Burglary network: 
- Given Alarm, JohnCalls is independent of Burglary, Earthquake, MaryCalls.
- $P(J | A, B, E, M) = P(J | A)$

This is why we can write:
$$P(B,E,A,J,M) = P(B)P(E)P(A|B,E)P(J|A)P(M|A)$$

instead of full chain rule.

### Step-by-Step:

| Step | Description |
|------|-------------|
| 1 | Identify each variable in network |
| 2 | Find its parents (direct ancestors) |
| 3 | Apply conditional independence: P(node \| ancestors) = P(node \| parents) |
| 4 | Multiply CPTs to get joint probability |

### Benefits:
- **Reduces parameters needed** drastically
- **Reflects natural causal structure**
- **Enables efficient inference**

---

## Q5. Explain the concepts of inference and learning in Bayesian Belief Networks. (5 Marks)

### 1. Inference in BBN:

**Definition:** Computing the probability of some unknown variables given the values of observed variables (evidence).

**Goal:** Calculate $P(\text{Query} | \text{Evidence})$.

**Types of Inference:**

a) **Causal (Top-down):** Predict effect from cause
   - Example: Given "Smoker = Yes", what is P(Cancer = Yes)?

b) **Diagnostic (Bottom-up):** Infer cause from effect
   - Example: Given "X-Ray = Positive", what is P(Cancer = Yes)?

c) **Mixed:** Combination of both

**Inference Methods:**
- **Exact Inference:** Variable elimination, junction tree (NP-hard in general)
- **Approximate Inference:** Sampling methods (Gibbs sampling, MCMC)

**Example:** In burglary network, given JohnCalls=True, find P(Burglary=True).

Using Bayes:
$$P(B|J) = \frac{P(J|B) P(B)}{P(J)}$$

### 2. Learning in BBN:

**Definition:** Determining the network structure and/or CPT values from training data.

**Two Types:**

a) **Parameter Learning** (CPT values):
   - **Network structure given:** Estimate CPT entries from data
   - If all variables are observable: Use frequency counts (like Naive Bayes)
   - If hidden variables: Use **EM Algorithm**

b) **Structure Learning** (Network graph):
   - Network structure NOT given
   - Algorithms search for best DAG that explains data
   - Uses scoring functions like BIC, MDL

### Example:

Suppose we have data for variables {Smoker, Cancer, X-Ray}. We can:
- **Learn parameters:** Count occurrences to fill CPTs:
  - P(Smoker=Yes) = (# of smokers) / (total people)
  - P(Cancer=Yes | Smoker=Yes) = (# of smoker cancer patients) / (# of smokers)
- **Learn structure:** Determine if "Smoker → Cancer → X-Ray" or some other structure best fits data.

### Step-by-Step:

| Step | Inference | Learning |
|------|-----------|----------|
| 1 | Identify query and evidence | Collect training data |
| 2 | Use Bayes theorem / sum rules | Decide what to learn (params/structure) |
| 3 | Apply variable elimination | Use frequency counts or EM |
| 4 | Output P(Query \| Evidence) | Output learned CPTs / DAG |

---

## Q6. Explain Conditional Independence in BBN with an example. (5 Marks)

### Definition:

Two variables X and Y are **conditionally independent given Z** if:
$$P(X, Y | Z) = P(X|Z) \cdot P(Y|Z)$$

Equivalently:
$$P(X | Y, Z) = P(X | Z)$$

### Role in Bayesian Networks:

In a BBN, each node is conditionally independent of:
- Its non-descendants
- Given its parents

### Example: "Sprinkler" Network

```
         Cloudy (C)
         /         \
        v           v
   Sprinkler (S)   Rain (R)
        \          /
         \        /
          v      v
          WetGrass (W)
```

**Conditional Independencies in this network:**

1. **S and R are conditionally independent given C:**
   $$P(S, R | C) = P(S|C) \cdot P(R|C)$$
   Once we know if it's cloudy, the sprinkler and rain are independent.

2. **W is conditionally independent of C given S and R:**
   $$P(W | C, S, R) = P(W | S, R)$$
   Once we know about sprinkler and rain, cloudy doesn't matter for wet grass.

### Why It Matters:

Without conditional independence, joint probability needs:
$$P(C,S,R,W) = P(C) P(S|C) P(R|S,C) P(W|R,S,C)$$

With conditional independence:
$$P(C,S,R,W) = P(C) P(S|C) P(R|C) P(W|S,R)$$

Notice **R doesn't depend on S** (independent given C), and **W doesn't depend on C** (independent given S, R). This dramatically reduces the parameters needed!

### Step-by-Step:

| Step | Action |
|------|--------|
| 1 | Identify all variables in BBN |
| 2 | For each pair, check if there's a path between them |
| 3 | Apply rule: independent of non-descendants given parents |
| 4 | Use this to factorize joint probability |

---

## Q7. NUMERICAL: Cloudy → Rain → WetGrass Bayesian Network (8 Marks)

### Given:

**Network:** Cloudy (C) → Rain (R) → WetGrass (W)

**Probabilities:**
- P(C = true) = 0.6
- P(R = true | C = true) = 0.7
- P(R = true | C = false) = 0.2
- P(W = true | R = true) = 0.9
- P(W = true | R = false) = 0.1

### Part (i): Compute P(C=true, R=true, W=true)

**Formula:** Joint probability in BBN
$$P(C, R, W) = P(C) \cdot P(R|C) \cdot P(W|R)$$

**Substitute values:**
$$P(C=T, R=T, W=T) = P(C=T) \cdot P(R=T|C=T) \cdot P(W=T|R=T)$$
$$= 0.6 \times 0.7 \times 0.9$$
$$= 0.378$$

**Answer:** $\boxed{P(C=T, R=T, W=T) = 0.378}$

### Part (ii): Compute P(W = true)

We need to **marginalize** over all values of C and R.

**Formula (Marginalization):**
$$P(W=T) = \sum_{C} \sum_{R} P(C, R, W=T)$$

**Step 1: Find P(R = true)** using marginalization over C:
$$P(R=T) = P(R=T | C=T) \cdot P(C=T) + P(R=T | C=F) \cdot P(C=F)$$
$$= (0.7 \times 0.6) + (0.2 \times 0.4)$$
$$= 0.42 + 0.08$$
$$= 0.50$$

So P(R=T) = 0.50, and P(R=F) = 0.50.

**Step 2: Find P(W = true)** using marginalization over R:
$$P(W=T) = P(W=T | R=T) \cdot P(R=T) + P(W=T | R=F) \cdot P(R=F)$$
$$= (0.9 \times 0.50) + (0.1 \times 0.50)$$
$$= 0.45 + 0.05$$
$$= 0.50$$

**Answer:** $\boxed{P(W = true) = 0.50}$

### Step-by-Step Recap:

| Step | Action | Result |
|------|--------|--------|
| 1 | Apply chain rule for joint: P(C)·P(R\|C)·P(W\|R) | 0.6 × 0.7 × 0.9 = 0.378 |
| 2 | For P(W), first compute P(R=T) by summing over C | 0.42 + 0.08 = 0.50 |
| 3 | Then compute P(W=T) by summing over R | 0.45 + 0.05 = 0.50 |

### Key Trick to Remember:
- For **joint probability**: Just multiply along the chain
- For **marginal probability**: Sum over unobserved variables (Total Probability)

---


# SECTION C: NAIVE BAYES CLASSIFIER

> **All Naive Bayes problems use this ONE formula:**
> $$v_{NB} = \arg\max_{v_j \in V} P(v_j) \prod_{i} P(a_i | v_j)$$

## Q16. Explain Naive Bayes Classifier with Example. (10 Marks)

### Definition:

The **Naive Bayes Classifier** is a probabilistic classifier based on **Bayes theorem** with the "naive" assumption that all features (attributes) are **conditionally independent** given the class.

### Why "Naive"?

Because it makes the strong (and often unrealistic) assumption that all attributes are independent of each other given the target class. Despite this simplification, it works very well in practice!

### Derivation:

Given an instance with attributes $\langle a_1, a_2, ..., a_n \rangle$, we want to find the most probable class $v_j \in V$:

**Step 1:** Apply MAP:
$$v_{MAP} = \arg\max_{v_j \in V} P(v_j | a_1, a_2, ..., a_n)$$

**Step 2:** Apply Bayes theorem:
$$v_{MAP} = \arg\max_{v_j \in V} \frac{P(a_1, a_2, ..., a_n | v_j) \cdot P(v_j)}{P(a_1, a_2, ..., a_n)}$$

**Step 3:** Drop denominator (constant for all v_j):
$$v_{MAP} = \arg\max_{v_j \in V} P(a_1, a_2, ..., a_n | v_j) \cdot P(v_j)$$

**Step 4:** Apply the **naive (conditional independence) assumption:**
$$P(a_1, a_2, ..., a_n | v_j) = \prod_i P(a_i | v_j)$$

**Step 5:** Final Naive Bayes formula:
$$\boxed{v_{NB} = \arg\max_{v_j \in V} P(v_j) \prod_{i} P(a_i | v_j)}$$

### Example: Spam Classification

Suppose we want to classify an email as Spam or Not-Spam based on words present.

**Training Data Statistics:**
- P(Spam) = 0.4, P(Not-Spam) = 0.6
- P("offer" | Spam) = 0.8, P("offer" | Not-Spam) = 0.1
- P("meeting" | Spam) = 0.05, P("meeting" | Not-Spam) = 0.5

**Test Email:** Contains "offer" and "meeting"

**Compute for Spam:**
$$P(Spam) \cdot P(offer|Spam) \cdot P(meeting|Spam) = 0.4 \times 0.8 \times 0.05 = 0.016$$

**Compute for Not-Spam:**
$$P(NotSpam) \cdot P(offer|NotSpam) \cdot P(meeting|NotSpam) = 0.6 \times 0.1 \times 0.5 = 0.030$$

Since 0.030 > 0.016, **classify as Not-Spam**.

### Algorithm Steps:

| Step | Action |
|------|--------|
| 1 | From training data, compute P(v_j) for each class |
| 2 | For each class v_j and each attribute value a_i, compute P(a_i \| v_j) |
| 3 | For new instance, multiply P(v_j) × ∏ P(a_i \| v_j) for each class |
| 4 | Pick the class with highest product |

### Advantages:
- Simple and fast
- Works well even with small training data
- Effective for text classification

### Disadvantages:
- Assumes attribute independence (often unrealistic)
- Zero probability problem (handled by **Laplace smoothing**)

### Laplace Smoothing (m-estimate):
If P(a_i | v_j) = 0, the entire product becomes 0. Fix:
$$P(a_i | v_j) = \frac{n_c + mp}{n + m}$$

Where:
- $n_c$ = number of training examples with class v_j and attribute a_i
- $n$ = total examples with class v_j
- $p$ = prior estimate (usually 1/k for k attribute values)
- $m$ = equivalent sample size

---

## Q18. NUMERICAL: PlayTennis Naive Bayes Problem (10 Marks)

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

**Classify:** ⟨Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong⟩

### Step 1: Calculate Prior Probabilities P(v_j)

Total examples = 14
- PlayTennis = Yes: D3, D4, D5, D7, D9, D10, D11, D12, D13 → **9 examples**
- PlayTennis = No: D1, D2, D6, D8, D14 → **5 examples**

$$P(\text{Yes}) = \frac{9}{14} = 0.643$$
$$P(\text{No}) = \frac{5}{14} = 0.357$$

### Step 2: Calculate Conditional Probabilities P(a_i | v_j)

**For PlayTennis = Yes (9 examples):**

| Attribute | Count | P(a_i \| Yes) |
|-----------|-------|---------------|
| Outlook = Sunny | D9, D11 → 2 | 2/9 = 0.222 |
| Temperature = Cool | D5, D7, D9 → 3 | 3/9 = 0.333 |
| Humidity = High | D3, D4, D12 → 3 | 3/9 = 0.333 |
| Wind = Strong | D7, D11, D12 → 3 | 3/9 = 0.333 |

**For PlayTennis = No (5 examples):**

| Attribute | Count | P(a_i \| No) |
|-----------|-------|--------------|
| Outlook = Sunny | D1, D2, D8 → 3 | 3/5 = 0.600 |
| Temperature = Cool | D6 → 1 | 1/5 = 0.200 |
| Humidity = High | D1, D2, D8, D14 → 4 | 4/5 = 0.800 |
| Wind = Strong | D2, D6, D14 → 3 | 3/5 = 0.600 |

### Step 3: Apply Naive Bayes Formula

**For "Yes":**
$$P(\text{Yes}) \cdot P(\text{Sunny}|Y) \cdot P(\text{Cool}|Y) \cdot P(\text{High}|Y) \cdot P(\text{Strong}|Y)$$
$$= 0.643 \times 0.222 \times 0.333 \times 0.333 \times 0.333$$
$$= 0.643 \times 0.222 \times 0.333 \times 0.333 \times 0.333$$
$$= 0.00528$$

**For "No":**
$$P(\text{No}) \cdot P(\text{Sunny}|N) \cdot P(\text{Cool}|N) \cdot P(\text{High}|N) \cdot P(\text{Strong}|N)$$
$$= 0.357 \times 0.600 \times 0.200 \times 0.800 \times 0.600$$
$$= 0.02057$$

### Step 4: Compare and Decide

- Score for Yes = 0.00528
- Score for No = 0.02057

Since **0.02057 > 0.00528**, the classifier predicts:

$$\boxed{v_{NB} = \text{No (PlayTennis = No)}}$$

### Step 5 (Optional): Normalize to Probabilities

$$P(\text{Yes} | \text{instance}) = \frac{0.00528}{0.00528 + 0.02057} = \frac{0.00528}{0.02585} = 0.204$$

$$P(\text{No} | \text{instance}) = \frac{0.02057}{0.02585} = 0.795$$

So we are about **79.5% confident** the player will NOT play tennis.

### Step-by-Step Recap:

| Step | Action |
|------|--------|
| 1 | Count Yes/No examples → priors P(Yes), P(No) |
| 2 | Count attribute values within each class → P(a_i\|v_j) |
| 3 | Multiply prior × conditional probabilities for each class |
| 4 | Pick class with highest score → answer |

---

## Q20. NUMERICAL: Stolen Car Naive Bayes Classifier (10 Marks)

### Given Training Data:

| Color | Type | Origin | Stolen |
|-------|------|--------|--------|
| Red | Sports | Domestic | Yes |
| Red | Sports | Domestic | No |
| Red | Sports | Domestic | Yes |
| Yellow | Sports | Domestic | No |
| Yellow | Sports | Imported | Yes |
| Yellow | SUV | Imported | No |
| Yellow | SUV | Imported | Yes |
| Yellow | SUV | Domestic | No |
| Red | SUV | Imported | No |
| Red | Sports | Imported | Yes |

**Classify:** ⟨Color=Red, Type=SUV, Origin=Domestic⟩

### Step 1: Calculate Prior Probabilities

Total examples = 10
- Stolen = Yes → 5 examples (rows 1, 3, 5, 7, 10)
- Stolen = No → 5 examples (rows 2, 4, 6, 8, 9)

$$P(\text{Yes}) = \frac{5}{10} = 0.5$$
$$P(\text{No}) = \frac{5}{10} = 0.5$$

### Step 2: Calculate Conditional Probabilities

**For Stolen = Yes (5 examples):**

| Attribute | Count | P(a_i \| Yes) |
|-----------|-------|---------------|
| Color = Red | rows 1, 3, 10 → 3 | 3/5 = 0.6 |
| Type = SUV | row 7 → 1 | 1/5 = 0.2 |
| Origin = Domestic | rows 1, 3 → 2 | 2/5 = 0.4 |

**For Stolen = No (5 examples):**

| Attribute | Count | P(a_i \| No) |
|-----------|-------|--------------|
| Color = Red | rows 2, 9 → 2 | 2/5 = 0.4 |
| Type = SUV | rows 6, 8, 9 → 3 | 3/5 = 0.6 |
| Origin = Domestic | rows 2, 4, 8 → 3 | 3/5 = 0.6 |

### Verification of Counts (let me re-verify):

**Stolen = Yes rows:**
- Row 1: Red, Sports, Domestic, **Yes** ✓
- Row 3: Red, Sports, Domestic, **Yes** ✓
- Row 5: Yellow, Sports, Imported, **Yes** ✓
- Row 7: Yellow, SUV, Imported, **Yes** ✓
- Row 10: Red, Sports, Imported, **Yes** ✓

For Yes:
- Red: rows 1, 3, 10 = **3**
- SUV: row 7 = **1**
- Domestic: rows 1, 3 = **2**

**Stolen = No rows:**
- Row 2: Red, Sports, Domestic, **No** ✓
- Row 4: Yellow, Sports, Domestic, **No** ✓
- Row 6: Yellow, SUV, Imported, **No** ✓
- Row 8: Yellow, SUV, Domestic, **No** ✓
- Row 9: Red, SUV, Imported, **No** ✓

For No:
- Red: rows 2, 9 = **2**
- SUV: rows 6, 8, 9 = **3**
- Domestic: rows 2, 4, 8 = **3**

All verified! ✓

### Step 3: Apply Naive Bayes Formula

**For Stolen = Yes:**
$$P(\text{Yes}) \cdot P(\text{Red}|Y) \cdot P(\text{SUV}|Y) \cdot P(\text{Domestic}|Y)$$
$$= 0.5 \times 0.6 \times 0.2 \times 0.4$$
$$= 0.024$$

**For Stolen = No:**
$$P(\text{No}) \cdot P(\text{Red}|N) \cdot P(\text{SUV}|N) \cdot P(\text{Domestic}|N)$$
$$= 0.5 \times 0.4 \times 0.6 \times 0.6$$
$$= 0.072$$

### Step 4: Compare and Decide

- Score for Yes = 0.024
- Score for No = 0.072

Since **0.072 > 0.024**, the classifier predicts:

$$\boxed{v_{NB} = \text{No (Car will NOT be stolen)}}$$

### Step 5 (Optional): Normalize to Probabilities

$$P(\text{Yes} | \text{instance}) = \frac{0.024}{0.024 + 0.072} = \frac{0.024}{0.096} = 0.25 = 25\%$$

$$P(\text{No} | \text{instance}) = \frac{0.072}{0.096} = 0.75 = 75\%$$

So a Red SUV from Domestic origin has **75% probability of NOT being stolen**.

### Step-by-Step Recap:

| Step | Action |
|------|--------|
| 1 | Calculate P(Yes) and P(No) priors from class counts |
| 2 | For each class, calculate P(attribute_value\|class) |
| 3 | Multiply prior × all conditional probabilities |
| 4 | Highest score wins → predicted class |
| 5 | (Optional) Normalize to get actual probabilities |

### Pattern to Notice:

**For ALL Naive Bayes problems**, the procedure is exactly the same:
1. Count → Priors
2. Count → Conditionals  
3. Multiply
4. Compare → Decide

This is why I grouped them together! Once you do one, you can do them all.

---


# SECTION D: EM ALGORITHM

## Q17. Explain Expectation-Maximization (EM) Algorithm in Detail. (10 Marks)

### Definition:

The **Expectation-Maximization (EM) algorithm** is an iterative method used to find Maximum Likelihood (ML) or Maximum A Posteriori (MAP) estimates of parameters in statistical models when **some data is missing or hidden (latent variables)**.

### When is EM Used?

When we have:
- **Observed data X** (visible)
- **Hidden/Latent variables Z** (not observable)
- **Parameters θ** that we want to estimate

We cannot directly maximize $P(X | \theta)$ because of the hidden variables. EM solves this iteratively.

### Two Steps of EM:

The algorithm alternates between two steps:

#### **E-Step (Expectation Step):**
Using current parameter estimate θ, calculate the **expected value** of the hidden variables.

$$Q(\theta | \theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log P(X, Z | \theta)]$$

In simple terms: "Given current parameters, what's the most likely value of hidden variables?"

#### **M-Step (Maximization Step):**
Maximize the expected log-likelihood found in E-step to update parameters:

$$\theta^{(t+1)} = \arg\max_{\theta} Q(\theta | \theta^{(t)})$$

In simple terms: "Now find new parameters that best explain the data given those expected hidden values."

### Algorithm Flow:

```
1. Initialize parameters θ^(0) randomly
2. Repeat until convergence:
    a. E-Step: Compute expected values of hidden variables using θ^(t)
    b. M-Step: Update θ^(t+1) by maximizing likelihood with these expected values
3. Output final θ
```

### Classic Example: Estimating Means of Two Gaussians

**Problem:** We have a dataset of points generated from a mixture of two Gaussian distributions $N(\mu_1, \sigma^2)$ and $N(\mu_2, \sigma^2)$, but we don't know which point came from which distribution.

**Hidden Variable:** $z_i$ = which Gaussian generated point $x_i$

**Steps:**

**Initialization:** Pick random initial $\mu_1$ and $\mu_2$.

**E-Step:** For each point $x_i$, compute the probability it came from Gaussian 1 vs Gaussian 2:
$$E[z_{i1}] = \frac{P(x_i | \mu_1)}{P(x_i | \mu_1) + P(x_i | \mu_2)}$$
$$E[z_{i2}] = 1 - E[z_{i1}]$$

**M-Step:** Update means using weighted averages:
$$\mu_1^{new} = \frac{\sum_i E[z_{i1}] \cdot x_i}{\sum_i E[z_{i1}]}$$
$$\mu_2^{new} = \frac{\sum_i E[z_{i2}] \cdot x_i}{\sum_i E[z_{i2}]}$$

**Repeat** E-step and M-step until $\mu_1, \mu_2$ stop changing.

### Properties of EM:

1. **Always Converges:** Each iteration improves (or maintains) the likelihood.
2. **Local Optima:** Can get stuck in local maxima (run with multiple initializations).
3. **Monotonic Increase:** Likelihood never decreases.
4. **No Guarantee of Global Optimum:** Final answer depends on initialization.

### Applications:

1. **Gaussian Mixture Models (GMM)** — clustering
2. **Hidden Markov Models (HMM)** — speech recognition, NLP
3. **Bayesian networks with missing data**
4. **Image segmentation**
5. **Document clustering**

### Step-by-Step Procedure:

| Step | Action |
|------|--------|
| 1 | Initialize parameters θ randomly |
| 2 | **E-Step:** Compute expected values of hidden variables given θ |
| 3 | **M-Step:** Update θ to maximize likelihood given expected hidden values |
| 4 | Check for convergence (do parameters stop changing?) |
| 5 | If not converged, go to step 2 |
| 6 | Output final parameters |

### Visual Intuition:

```
  Initial guess (bad)
         ↓
   E-step (estimate hidden)
         ↓
   M-step (update parameters)
         ↓
   E-step (estimate hidden again, better now)
         ↓
   M-step (update again)
         ↓
   ... continue until converge ...
         ↓
   Final answer (good estimate)
```

### Why EM Works:

The trick is: instead of trying to maximize the **observed-data likelihood** directly (which is hard), EM maximizes a **lower bound** of it iteratively. Each iteration tightens this bound.

---

## Q8. Differentiate between Naive Bayes and EM Algorithm with Example. (8 Marks)

### Comparison Table:

| Aspect | Naive Bayes | EM Algorithm |
|--------|-------------|--------------|
| **Type** | Classification algorithm | Parameter estimation algorithm |
| **Goal** | Predict class label of new instances | Estimate parameters of probabilistic models with hidden data |
| **Data Required** | Fully observed labeled training data | Can work with missing/hidden data |
| **Approach** | Direct application of Bayes theorem | Iterative two-step (E and M) procedure |
| **Output** | Class labels (predictions) | Estimated parameters (means, variances, weights) |
| **Independence Assumption** | Assumes attributes are conditionally independent given class | No such assumption; uses any probabilistic model |
| **Iterations Needed** | No - single pass over data | Yes - iterates until convergence |
| **Hidden Variables** | Cannot handle | Specifically designed to handle |
| **Computational Complexity** | Low (simple counting) | Higher (multiple iterations) |
| **Convergence** | Always works in one pass | May converge to local maximum |
| **Use Case** | Supervised learning (labels available) | Unsupervised/semi-supervised (labels may be missing) |
| **Formula** | $v_{NB} = \arg\max_{v_j} P(v_j) \prod P(a_i\|v_j)$ | E-step + M-step iterations |

### Examples:

#### Naive Bayes Example: Email Spam Classification

**Setup:** We have 1000 emails labeled as Spam or Not-Spam. Each email has features (words present).

**Process:**
1. Count occurrences of each word in each class
2. Compute P(Spam), P(Not-Spam)
3. Compute P(word | Spam), P(word | Not-Spam)
4. For new email: multiply probabilities and pick higher

**Result:** Direct classification — "This is Spam" or "This is Not Spam"

#### EM Example: Customer Segmentation (Clustering)

**Setup:** We have customer purchase data but **no labels** about customer types. We believe there are 3 hidden customer types.

**Process:**
1. Initialize: 3 random customer "types" with random parameters
2. **E-Step:** For each customer, calculate probability they belong to each of the 3 types
3. **M-Step:** Update the type parameters (mean spending, etc.) using weighted customer data
4. Repeat until parameters stabilize

**Result:** Hidden customer segments are discovered (e.g., "Budget shoppers", "Premium buyers", "Casual buyers")

### Key Differentiating Example:

**Same Problem, Two Approaches:**

Suppose we have student exam scores and want to know if students belong to "weak" or "strong" group.

**Naive Bayes Approach (Supervised):**
- Need labels: "Student A is weak, Student B is strong, ..." for training
- Use these labels to learn parameters
- Then classify new students

**EM Approach (Unsupervised):**
- No labels available, just scores
- EM discovers the two groups by iterating
- Initially random, gets refined: which students are "weak", which are "strong"

### Step-by-Step Differences:

| Step | Naive Bayes | EM |
|------|-------------|-----|
| 1 | Have labeled data | Have unlabeled or partially labeled data |
| 2 | Count frequencies | Initialize parameters randomly |
| 3 | Compute P(class), P(feature\|class) | E-step: estimate hidden labels |
| 4 | Apply Bayes formula to classify | M-step: update parameters |
| 5 | Done | Repeat until convergence |

### Summary:

- **Naive Bayes** = "I know the labels, help me classify new instances"
- **EM** = "I don't know everything, help me figure out hidden structure"

They're complementary tools, used in different scenarios!

---


# SECTION E: MAP / POSTERIOR PROBABILITY NUMERICAL PROBLEMS

> **All three questions in this section use the SAME approach:**
> 1. Identify the two hypotheses (h₁ and h₂)
> 2. Write down priors P(h₁), P(h₂)
> 3. Write down likelihoods P(D|h₁), P(D|h₂)
> 4. Compute P(h|D) ∝ P(D|h) × P(h) for each
> 5. Pick the larger one → that's h_MAP

> **Master formula for this section:**
> $$h_{MAP} = \arg\max_{h} P(D|h) \cdot P(h)$$
> 
> **And to get actual probabilities, normalize:**
> $$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)} \text{ where } P(D) = \sum_h P(D|h) P(h)$$

---

## Q11. NUMERICAL: Medical Diagnosis (Cancer Test) — MAP Hypothesis (10 Marks)

### Problem Statement:

Two hypotheses:
- $h_1$ = patient has cancer
- $h_2 = \neg cancer$ = patient does NOT have cancer

**Given Information:**
- Test returns **positive** for 98% of cases when disease is present → P(+ | cancer) = 0.98
- Test returns **negative** for 97% of cases when disease is absent → P(− | ¬cancer) = 0.97
- 0.008 (0.8%) of population has cancer → P(cancer) = 0.008

**Question:** Patient's test came back POSITIVE. Does the patient have cancer (using MAP)?

### Step 1: List All Probabilities

**Priors:**
- P(cancer) = 0.008
- P(¬cancer) = 1 − 0.008 = 0.992

**Likelihoods:**
- P(+ | cancer) = 0.98 (true positive rate)
- P(− | cancer) = 1 − 0.98 = 0.02 (false negative rate)
- P(− | ¬cancer) = 0.97 (true negative rate)
- P(+ | ¬cancer) = 1 − 0.97 = 0.03 (false positive rate)

### Step 2: Identify Data D

The observed data D = "test result is positive" = "+"

### Step 3: Apply MAP Formula

$$h_{MAP} = \arg\max_{h \in \{cancer, \neg cancer\}} P(D|h) \cdot P(h)$$

**For h₁ = cancer:**
$$P(+ | cancer) \cdot P(cancer) = 0.98 \times 0.008 = 0.00784$$

**For h₂ = ¬cancer:**
$$P(+ | \neg cancer) \cdot P(\neg cancer) = 0.03 \times 0.992 = 0.02976$$

### Step 4: Compare and Decide

- Score for cancer = **0.00784**
- Score for ¬cancer = **0.02976**

Since **0.02976 > 0.00784**:

$$\boxed{h_{MAP} = \neg cancer}$$

**The MAP hypothesis is that the patient does NOT have cancer.**

### Step 5: Calculate Actual Posterior Probabilities

**Find P(D) = P(+):**
$$P(+) = P(+|cancer) P(cancer) + P(+|\neg cancer) P(\neg cancer)$$
$$P(+) = 0.00784 + 0.02976 = 0.0376$$

**Posterior probabilities:**
$$P(cancer | +) = \frac{0.00784}{0.0376} = 0.2085 \approx 20.85\%$$

$$P(\neg cancer | +) = \frac{0.02976}{0.0376} = 0.7915 \approx 79.15\%$$

### Important Insight:

Even though the test is **98% accurate**, when the patient tests POSITIVE, there's only a **20.85% chance** they actually have cancer!

Why? Because cancer is rare (0.8% of population), so most positive tests are **false positives** from the large healthy population. This is the famous **base rate fallacy** in medical statistics.

### Step-by-Step Recap:

| Step | Action | Calculation |
|------|--------|-------------|
| 1 | Identify hypotheses | h₁ = cancer, h₂ = ¬cancer |
| 2 | Write priors | P(cancer)=0.008, P(¬cancer)=0.992 |
| 3 | Write likelihoods | P(+\|cancer)=0.98, P(+\|¬cancer)=0.03 |
| 4 | Compute P(D\|h)·P(h) | 0.00784 vs 0.02976 |
| 5 | Pick larger → h_MAP | ¬cancer wins |
| 6 | Normalize for true probabilities | 20.85% vs 79.15% |

---

## Q15. NUMERICAL: Cricket Match — Team 0 vs Team 1 (10 Marks)

### Problem Statement:

- Team 0 wins 95% of all matches → P(Team 0 wins) = 0.95
- Team 1 wins 5% of all matches → P(Team 1 wins) = 0.05
- Among Team 0's wins, only 30% happen on Team 1's field → P(Team 1's field | Team 0 wins) = 0.30
- Among Team 1's wins, 75% happen at home (Team 1's field) → P(Team 1's field | Team 1 wins) = 0.75
- **The next match will be played at Team 1's field (home).**

**Question:** Which team will most likely emerge as the winner?

### Step 1: Identify Variables

Let:
- A = "Team 0 wins"
- B = "Team 1 wins"
- F = "Match is played at Team 1's field"

**We want to find:**
- P(A | F) = probability Team 0 wins given match is at Team 1's field
- P(B | F) = probability Team 1 wins given match is at Team 1's field

### Step 2: List All Probabilities

**Priors:**
- P(A) = P(Team 0 wins) = 0.95
- P(B) = P(Team 1 wins) = 0.05

**Likelihoods (probability of venue given winner):**
- P(F | A) = 0.30 (Team 0 wins on Team 1's field 30% of times)
- P(F | B) = 0.75 (Team 1 wins on its field 75% of times)

### Step 3: Apply Bayes Theorem (MAP)

$$h_{MAP} = \arg\max_{h \in \{A, B\}} P(F | h) \cdot P(h)$$

**For h = A (Team 0 wins):**
$$P(F | A) \cdot P(A) = 0.30 \times 0.95 = 0.285$$

**For h = B (Team 1 wins):**
$$P(F | B) \cdot P(B) = 0.75 \times 0.05 = 0.0375$$

### Step 4: Compare and Decide

- Score for Team 0 winning = **0.285**
- Score for Team 1 winning = **0.0375**

Since **0.285 > 0.0375**:

$$\boxed{h_{MAP} = \text{Team 0 will win}}$$

**Even though the match is at Team 1's home, Team 0 is more likely to win!**

### Step 5: Calculate Actual Posterior Probabilities

**Find P(F):**
$$P(F) = P(F|A) P(A) + P(F|B) P(B) = 0.285 + 0.0375 = 0.3225$$

**Posteriors:**
$$P(A | F) = \frac{0.285}{0.3225} = 0.8837 \approx 88.37\%$$

$$P(B | F) = \frac{0.0375}{0.3225} = 0.1163 \approx 11.63\%$$

### Interpretation:

Even though Team 1 has home advantage, Team 0's overall dominance (95% win rate) is so strong that Team 0 still has an **88.37% chance of winning** the match at Team 1's home ground.

Team 1 only has **11.63% chance** of winning despite playing at home.

### Step-by-Step Recap:

| Step | Action | Calculation |
|------|--------|-------------|
| 1 | Identify hypotheses | A = Team 0 wins, B = Team 1 wins |
| 2 | Write priors | P(A)=0.95, P(B)=0.05 |
| 3 | Write likelihoods | P(F\|A)=0.30, P(F\|B)=0.75 |
| 4 | Compute P(F\|h)·P(h) | 0.285 vs 0.0375 |
| 5 | Pick larger → h_MAP | Team 0 wins |
| 6 | Normalize | 88.37% vs 11.63% |

---

## Q19. NUMERICAL: Spam Email Classifier (10 Marks)

### Problem Statement:

- 80% of all emails are spam → P(Spam) = 0.80
- If email is Spam, P("Congrats..." phrase) = 0.95 → P(C | Spam) = 0.95
- If email is Not Spam, P("Congrats..." phrase) = 0.05 → P(C | ¬Spam) = 0.05

**A new email contains "Congratulations, you've won!"**

**Questions:**
- (a) Which hypothesis maximizes posterior probability — Spam or Not Spam?
- (b) Calculate posterior probabilities for both hypotheses.

### Step 1: Identify Variables

- $h_1$ = Spam
- $h_2$ = Not Spam (¬Spam)
- D = email contains the phrase C = "Congratulations, you've won!"

### Step 2: List All Probabilities

**Priors:**
- P(Spam) = 0.80
- P(¬Spam) = 1 − 0.80 = 0.20

**Likelihoods:**
- P(C | Spam) = 0.95
- P(C | ¬Spam) = 0.05

### Step 3: Apply MAP Formula

**For h₁ = Spam:**
$$P(C | Spam) \cdot P(Spam) = 0.95 \times 0.80 = 0.76$$

**For h₂ = ¬Spam:**
$$P(C | \neg Spam) \cdot P(\neg Spam) = 0.05 \times 0.20 = 0.01$$

### Step 4: Compare and Decide (Part a)

- Score for Spam = **0.76**
- Score for Not Spam = **0.01**

Since **0.76 > 0.01**:

$$\boxed{h_{MAP} = \text{Spam}}$$

**The MAP hypothesis is that the email IS Spam.**

### Step 5: Calculate Posterior Probabilities (Part b)

**Find P(C) = total probability of seeing the phrase:**
$$P(C) = P(C|Spam) P(Spam) + P(C|\neg Spam) P(\neg Spam)$$
$$P(C) = 0.76 + 0.01 = 0.77$$

**Posterior for Spam:**
$$P(Spam | C) = \frac{P(C|Spam) P(Spam)}{P(C)} = \frac{0.76}{0.77} = 0.9870 \approx 98.70\%$$

**Posterior for Not Spam:**
$$P(\neg Spam | C) = \frac{P(C|\neg Spam) P(\neg Spam)}{P(C)} = \frac{0.01}{0.77} = 0.0130 \approx 1.30\%$$

### Final Answer:

| Hypothesis | Posterior Probability |
|------------|----------------------|
| **Spam** | **98.70%** ← Winner |
| Not Spam | 1.30% |

**Conclusion:** The email is almost certainly Spam (98.70% confidence).

### Step-by-Step Recap:

| Step | Action | Calculation |
|------|--------|-------------|
| 1 | Identify hypotheses | h₁ = Spam, h₂ = ¬Spam |
| 2 | Write priors | P(Spam)=0.80, P(¬Spam)=0.20 |
| 3 | Write likelihoods | P(C\|Spam)=0.95, P(C\|¬Spam)=0.05 |
| 4 | Compute P(D\|h)·P(h) | 0.76 vs 0.01 |
| 5 | Pick larger → h_MAP | Spam wins |
| 6 | Normalize for posteriors | 98.70% vs 1.30% |

### Verification: Posteriors should sum to 1
0.9870 + 0.0130 = 1.000 ✓

---

# UNIVERSAL EXAM CHEATSHEET (Last-Minute Revision)

## Common Formula for ALL MAP Numericals (Q11, Q15, Q19):

```
                P(D|h) × P(h)
P(h|D) = ─────────────────────────
            ∑_h P(D|h) × P(h)
```

**Steps to solve ANY MAP problem:**
1. Identify two hypotheses
2. Find priors P(h₁), P(h₂) (must sum to 1)
3. Find likelihoods P(D|h₁), P(D|h₂)
4. Multiply: P(D|h_i) × P(h_i)
5. Higher product = h_MAP
6. Normalize by sum for actual probabilities

## Common Formula for ALL Naive Bayes Numericals (Q18, Q20):

$$v_{NB} = \arg\max_{v_j} P(v_j) \prod_i P(a_i | v_j)$$

**Steps to solve ANY Naive Bayes problem:**
1. Count Yes/No (or class) examples → priors
2. Count attribute values within each class → conditional probabilities
3. For test instance: multiply prior × all conditionals
4. Pick class with highest score

## Common Formula for ALL Bayesian Network Numericals (Q7):

$$P(x_1, x_2, ..., x_n) = \prod_i P(x_i | \text{Parents}(x_i))$$

**Steps:**
- Joint probability: Multiply along the chain
- Marginal probability: Sum over unobserved variables (Total Probability)

---

# QUICK COMPARISON TABLE OF ALL NUMERICALS

| Question | Type | Hypothesis 1 (Prior × Likelihood) | Hypothesis 2 (Prior × Likelihood) | Answer |
|----------|------|-----------------------------------|-----------------------------------|--------|
| Q7 | BBN Joint | — | — | P(C,R,W)=0.378; P(W)=0.50 |
| Q11 | MAP Cancer | 0.008 × 0.98 = 0.00784 | 0.992 × 0.03 = 0.02976 | ¬cancer (79.15%) |
| Q15 | MAP Cricket | 0.95 × 0.30 = 0.285 | 0.05 × 0.75 = 0.0375 | Team 0 (88.37%) |
| Q19 | MAP Spam | 0.80 × 0.95 = 0.76 | 0.20 × 0.05 = 0.01 | Spam (98.70%) |
| Q18 | Naive Bayes | (Yes) 0.00528 | (No) 0.02057 | No (79.5%) |
| Q20 | Naive Bayes | (Yes) 0.024 | (No) 0.072 | No (75%) |

---

# IMPORTANT POINTS TO REMEMBER FOR EXAM

## 1. Always show:
- The formula being used
- Identification of priors and likelihoods clearly
- All calculations step-by-step
- Final answer in a box

## 2. Common mistakes to avoid:
- ❌ Forgetting to multiply prior with likelihood
- ❌ Confusing P(D|h) with P(h|D)
- ❌ Not normalizing when asked for "probability"
- ❌ For Naive Bayes: ignoring the prior P(class)

## 3. If a probability is 0:
- Use **Laplace/m-estimate smoothing** to avoid zero
$$P(a_i | v_j) = \frac{n_c + mp}{n + m}$$

## 4. Decision Rule:
- For **MAP**: pick hypothesis with highest P(D|h)·P(h)
- For **ML**: pick hypothesis with highest P(D|h) (when priors equal)

## 5. Trick to verify your answer:
- Posteriors should sum to 1 (after normalization)
- If they don't, you made a calculation error

---

# FINAL TIPS FOR THE EXAM

1. **Memorize ONLY the master formulas** — don't memorize each question separately.

2. **Practice the calculation pattern**:
   - Bayes formula → multiply prior × likelihood → compare → answer

3. **Show ALL steps** — examiners give marks for method, not just final answer.

4. **For theoretical questions**: 
   - Define the term
   - Give formula
   - Explain components
   - Give example
   - List advantages/disadvantages

5. **For numerical questions**:
   - State given data
   - Identify formula
   - Substitute values
   - Calculate step-by-step
   - Box the final answer

6. **Time management**: 
   - 5-mark questions: 8-10 minutes
   - 10-mark questions: 18-20 minutes
   - Numerical 10-mark: ~15 minutes

---

## All The Best for Your Exam! 🎯

Remember: **Bayesian Learning is just one formula applied repeatedly.** Once you understand 
$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$
you understand 80% of this entire module!

