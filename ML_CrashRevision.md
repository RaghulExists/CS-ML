# ML (CML23604C) — LAST-MINUTE CRASH REVISION
### All 28 key questions + numericals, in plain English. Read top to bottom.
*Global Academy of Technology / VTU. Stay calm — each answer below is enough to score full marks.*

> **How to use this:** Each answer = a definition + key points + formula + tiny example. Write the heading, the definition, then the bullet points. That's a full answer. Numericals at the end of each module show the exact steps.

---
---

# 🟦 MODULE 1 — FOUNDATIONS

---

## M1.1 ⭐ Design a Learning System (Checkers) / Well-Posed Learning Problem

**Well-posed learning problem** — a problem defined by **T, P, E**:
- **T (Task):** what the program must do (e.g., play checkers).
- **P (Performance measure):** how success is judged (e.g., % games won).
- **E (Experience):** the data it learns from (e.g., playing games against itself).

*Example:* Checkers → **T**: playing checkers, **P**: % of games won, **E**: games played against itself.

**Designing the Checkers learning system — 4 steps:**

1. **Choose the training experience** — games played against itself (indirect feedback: only final win/lose known).
2. **Choose the target function** — what to learn:
   - `ChooseMove : Board → Move` (hard to learn directly), so instead learn
   - **`V : Board → ℝ`** = a value telling how good a board state is.
3. **Choose representation of V** — a linear function of board features:

   **V̂(b) = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + w₅x₅ + w₆x₆**

   where x₁…x₆ = number of black pieces, red pieces, black kings, red kings, black pieces threatened, red pieces threatened. w₀…w₆ are weights to learn.
4. **Choose the learning algorithm** — adjust weights using training examples via the **LMS (Least Mean Squares) weight update rule**:

   **wᵢ ← wᵢ + η (V_train(b) − V̂(b)) xᵢ**

   - η = learning rate, V_train(b) = target value, V̂(b) = current estimate.
   - Training value estimated by: **V_train(b) ← V̂(Successor(b))**.

**Final design = 4 modules (draw this):**
```
Experiment Generator → (new problem) → Performance System → (game trace)
       ↑                                                          ↓
   Generalizer  ← (training examples) ←  Critic  ← (game history)
```
- **Performance System:** plays games using learned V̂.
- **Critic:** produces training examples from game history.
- **Generalizer:** learns V̂ (adjusts weights).
- **Experiment Generator:** proposes new problems to learn from.

---

## M1.2 Perspectives and Issues in Machine Learning

**Perspective:** Learning = **searching a very large space of hypotheses** to find the one that best fits the data and prior knowledge. Different algorithms = different hypothesis spaces + search strategies.

**Key issues in ML (write these as bullet points):**
1. What algorithms exist for learning, and which work best for which problems?
2. How much training data is enough?
3. When/how can prior knowledge guide learning?
4. What is the best strategy for choosing the next training experience?
5. How to reduce the learning task to function approximation?
6. How can the learner automatically alter its representation to improve?
7. How to avoid overfitting?
8. Handling noisy/incomplete data.

---

## M1.3 Types of Machine Learning

1. **Supervised learning** — learns from **labelled data** (input + correct output). Tasks: **Classification** (discrete output, e.g., spam/not-spam) and **Regression** (continuous output, e.g., house price). Examples: Decision Trees, Naive Bayes, kNN.
2. **Unsupervised learning** — **unlabelled data**, finds hidden structure. Tasks: **Clustering** (group similar items), **Association**, **Dimensionality reduction**. Example: k-means, EM.
3. **Semi-supervised learning** — small amount of labelled + large amount of unlabelled data. Useful when labelling is expensive.
4. **Reinforcement learning** — agent learns by **trial and error**, receiving **rewards/penalties** from the environment. Goal: maximise long-term reward. Example: game playing, robotics.

---

## M1.4 ML and Its Relationship with Other Fields

- **AI ⊃ ML ⊃ Deep Learning** (nested). ML is a subset of AI; Deep Learning is a subset of ML.
- **ML vs Statistics:** Statistics focuses on inference/explaining relationships with assumptions about distributions; ML focuses on **prediction accuracy** on new data, often with fewer assumptions.
- **ML vs Data Mining:** Data Mining = discovering patterns in large databases (uses ML methods). ML = building models that improve with experience.
- **ML vs Data Science:** Data Science is the broad field (collection, cleaning, analysis, visualization); ML is one tool within it.

```
        ┌──────────── AI ────────────┐
        │   ┌──────── ML ────────┐   │
        │   │   ┌── Deep ──┐     │   │
        │   │   │ Learning │     │   │
        │   │   └──────────┘     │   │
        │   └────────────────────┘   │
        └────────────────────────────┘
   Statistics + Data Mining overlap with ML
```

---

## M1.5 CRISP-DM (Cross-Industry Standard Process for Data Mining)

A **6-phase cyclic process** for data science projects:

1. **Business Understanding** — define the objective.
2. **Data Understanding** — collect & explore data.
3. **Data Preparation** — clean, transform, select features.
4. **Modeling** — apply ML algorithms.
5. **Evaluation** — check if results meet business goals.
6. **Deployment** — put the model into real use.

It is **iterative** — you can loop back (e.g., Evaluation → Business Understanding). Data is at the centre.

```
Business → Data → Data → Modeling → Evaluation → Deployment
Underst.  Underst. Prep.      ↑__________↓ (loop back)
```

---

## M1.6 Knowledge Pyramid (DIKW)

5 levels, bottom to top (each level adds meaning):

1. **Data** — raw facts, no meaning (e.g., "170, 57").
2. **Information** — processed/organized data with context (e.g., "Height=170cm, Weight=57kg").
3. **Knowledge** — patterns/rules derived (e.g., "this person is of normal weight").
4. **Intelligence** — applied knowledge for decisions.
5. **Wisdom** — using intelligence with judgment for best action.

```
        /\
       /Wis\        Wisdom
      /-----\
     / Intel \      Intelligence
    /---------\
   / Knowledge \    Knowledge
  /-------------\
 /  Information  \  Information
/-----------------\
/      Data        \ Data (raw)
--------------------
```

---
---

# 🟩 MODULE 2 — CONCEPT LEARNING

**Hypothesis notation:** h = ⟨a₁, a₂, …, aₙ⟩, each constraint is a **specific value**, **"?"** (any value), or **"∅"** (no value).

---

## M2.1 ⭐ Candidate Elimination Algorithm

**Idea:** Maintains the **entire version space** using two boundaries — **S** (most specific) and **G** (most general). Uses **both** positive and negative examples.

**Algorithm:**
- Initialise **S = {⟨∅,∅,…,∅⟩}** and **G = {⟨?,?,…,?⟩}**.
- **For a POSITIVE example:**
  - Remove from G any hypothesis inconsistent with it.
  - Generalise S minimally so it covers the example.
- **For a NEGATIVE example:**
  - Remove from S any hypothesis inconsistent with it.
  - Specialise G minimally so it rejects the example.
- Remove any hypothesis in S more general than another in S; remove any in G more specific than another in G.

**Rule of thumb:** *Positive → generalise S. Negative → specialise G.*

**EnjoySport result (memorise):**
- S = {⟨Sunny, Warm, ?, Strong, ?, ?⟩}
- G = {⟨Sunny,?,?,?,?,?⟩, ⟨?,Warm,?,?,?,?⟩}

---

## M2.2 FIND-S Algorithm + Properties

**Idea:** Finds the **most specific hypothesis** consistent with the **positive** examples. **Ignores negatives.**

**Algorithm:**
1. Initialise h = ⟨∅,∅,…,∅⟩ (most specific).
2. For each **positive** example x:
   - For each attribute: if h's constraint already matches → keep; else replace with **"?"** (or set it if it was ∅).
3. Skip all negative examples.
4. Output h.

**EnjoySport result:** h = ⟨Sunny, Warm, ?, Strong, ?, ?⟩

**Key properties (write these):**
- Outputs the maximally specific consistent hypothesis.
- Ignores negative examples → can't detect noise.
- Converges only if target concept is in H and data is noise-free.
- Outputs only ONE hypothesis (no uncertainty info).

---

## M2.3 Version Space + S and G Boundaries

- **Version Space (VS):** the set of **all hypotheses consistent** with the training data. VS = {h ∈ H | Consistent(h, D)}.
- **Consistent:** h is consistent with D if it correctly classifies **every** training example (h(x) = c(x) for all examples).
- **Satisfies:** h satisfies instance x if h classifies x as **positive** (single instance only).
- **Satisfies = one instance, Consistent = all examples.** (Common exam trap.)
- **S-boundary:** set of **most specific** consistent hypotheses ("hugs" positives).
- **G-boundary:** set of **most general** consistent hypotheses ("hugs" negatives).
- VS = everything **between** S and G in the general-to-specific ordering.

---

## M2.4 Inductive Bias

- **Inductive bias** = the set of **assumptions** a learner uses to classify **unseen** instances (beyond the training data).
- Without bias, a learner **cannot generalise** — it can only repeat training data.
- **Formally:** bias B is the minimal assumptions such that (training data + B) ⊢ (classification of new instances) by pure deduction.
- **Bias of Candidate Elimination:** "the target concept is in H" (it's a conjunction).
- **Bias of FIND-S:** the above + "the most specific hypothesis is best."
- This is **Occam's Razor**: prefer the simplest hypothesis.

---

## M2.5 Biased Hypothesis Space & Unbiased Learner

- **Biased hypothesis space:** H does NOT contain all possible concepts (e.g., only conjunctions). EnjoySport conjunctive H = 973 concepts out of 2⁹⁶ possible. **It can't represent disjunctions** like "Sunny OR Cloudy."
- **Unbiased learner:** H = **all** possible Boolean functions (2^|X| of them).
- **The paradox:** an unbiased learner **cannot generalise at all** — for any unseen instance, exactly half the consistent hypotheses say positive and half say negative (50/50 vote). No useful prediction.
- **Conclusion (key line):** *"A learner that makes no a priori assumptions (no bias) cannot classify any unseen instance."* → **Bias is necessary for learning.**

---

## M2.6 Concept Learning + EnjoySport Task

- **Concept learning:** inferring a **Boolean-valued function** from labelled training examples (positive/negative).
- **EnjoySport task:** learn when Aldo enjoys his water sport.
- **6 attributes:** Sky (Sunny/Cloudy/Rainy), AirTemp (Warm/Cold), Humidity (Normal/High), Wind (Strong/Light), Water (Warm/Cool), Forecast (Same/Change).
- **Target concept:** EnjoySport = Yes/No.
- Counts: instances = **96**, syntactic hypotheses = **5120**, semantic = **973**.

---

## 🔢 MODULE 2 NUMERICALS (full solutions)

### (A) Counting Distinct Instances & Hypotheses (EnjoySport)

Attributes & values: Sky(3), AirTemp(2), Humidity(2), Wind(2), Water(2), Forecast(2).

**Distinct instances** = product of number of values:
= 3 × 2 × 2 × 2 × 2 × 2 = 3 × 32 = **96**

**Syntactically distinct hypotheses** = ∏ (nᵢ + 2) [each attribute: a value, OR "?", OR "∅"]:
= (3+2)(2+2)(2+2)(2+2)(2+2)(2+2) = 5 × 4 × 4 × 4 × 4 × 4 = 5 × 1024 = **5120**

**Semantically distinct hypotheses** = ∏ (nᵢ + 1) + 1 [value or "?"; all-∅ collapse to 1 null]:
= (3+1)(2+1)(2+1)(2+1)(2+1)(2+1) + 1 = 4 × 3 × 3 × 3 × 3 × 3 + 1 = 4 × 243 + 1 = 972 + 1 = **973**

---

### (B) FIND-S Trace (Face dataset)

Attributes: Eyes, Nose, Head, Fcolor, Hair, Smile.

| # | Example | Class | h after step |
|---|---------|-------|--------------|
| init | — | — | ⟨∅,∅,∅,∅,∅,∅⟩ |
| 1 | Round,Triangle,Round,Purple,Yes,Yes | **+** | ⟨Round,Triangle,Round,Purple,Yes,Yes⟩ |
| 2 | Square,Square,Square,Green,Yes,No | **+** | ⟨?,?,?,?,Yes,?⟩ |
| 3 | Square,Triangle,Round,Yellow,Yes,Yes | **+** | ⟨?,?,?,?,Yes,?⟩ (no change) |
| 4 | Round,Triangle,Round,Green,No,No | **−** | SKIP (negative) |
| 5 | Square,Square,Round,Yellow,Yes,Yes | **+** | ⟨?,?,?,?,Yes,?⟩ (no change) |

**Step logic:** Ex1 sets h to itself. Ex2: every attribute except Hair differs from h → replace those with "?", Hair stays Yes. Ex3 & Ex5 already match (all "?" + Hair=Yes). Ex4 skipped (negative).

**Final hypothesis: h = ⟨?, ?, ?, ?, Yes, ?⟩** → "A face is in the concept iff Hair = Yes."

---

### (C) Candidate Elimination Trace (Car dataset — empty VS case)

Attributes: Origin(Japan/USA), Mfr(Honda/Toyota/Chrysler), Color(Blue/Green/Red/White), Decade, Type(Economy/Sports).

| After Example (label) | S | G |
|------------------------|---|---|
| Init | {⟨∅,∅,∅,∅,∅⟩} | {⟨?,?,?,?,?⟩} |
| 1 (Japan,Honda,Blue,1980,Economy) **+** | {⟨J,Ho,Bl,1980,E⟩} | {⟨?,?,?,?,?⟩} |
| 2 (Japan,Toyota,Green,1970,Sports) **−** | {⟨J,Ho,Bl,1980,E⟩} | {⟨?,Ho,?,?,?⟩,⟨?,?,Bl,?,?⟩,⟨?,?,?,1980,?⟩,⟨?,?,?,?,E⟩} |
| 3 (Japan,Toyota,Blue,1990,Economy) **+** | {⟨J,?,Bl,?,E⟩} | {⟨?,?,Bl,?,?⟩,⟨?,?,?,?,E⟩} |
| 4 (USA,Chrysler,Red,1980,Economy) **−** | {⟨J,?,Bl,?,E⟩} | {⟨?,?,Bl,?,?⟩,⟨J,?,?,?,E⟩} |
| 5 (Japan,Honda,White,1980,Economy) **+** | {⟨J,?,?,?,E⟩} | {⟨J,?,?,?,E⟩} |
| 6 (Japan,Toyota,Green,1980,Economy) **+** | {⟨J,?,?,?,E⟩} | {⟨J,?,?,?,E⟩} |
| 7 (Japan,Honda,Red,1990,Economy) **−** | { } | { } |

**Conclusion:** After Ex7, S would need to reject a negative it currently covers (⟨J,?,?,?,E⟩ also matches the Red negative) → **version space becomes EMPTY**. Meaning: the true concept ("Japanese Economy but NOT Red") is a *negation*, which **cannot be represented** in the conjunctive hypothesis space H. (Great point to state in the exam.)

**General rule to remember:** Positive → generalise S (and drop inconsistent G members). Negative → specialise G (and drop inconsistent S members).

---
---

# 🟨 MODULE 3 — DECISION TREES

**Core formulas (memorise):**

**Entropy:** H(S) = − p₊ log₂ p₊ − p₋ log₂ p₋  
(p = proportion of each class. H=0 → pure; H=1 → 50/50.)

**Information Gain:** Gain(S,A) = H(S) − Σ (|Sᵥ|/|S|) H(Sᵥ)  
(reduction in entropy by splitting on attribute A. Pick the attribute with **highest gain**.)

---

## M3.1 ⭐ ID3 Algorithm

**ID3 = greedy, top-down decision tree builder using Information Gain.**

**Steps:**
1. Create a Root node.
2. If all examples same class → leaf with that class.
3. If no attributes left → leaf with majority class.
4. Else: pick **A\* = attribute with highest Information Gain**.
5. Make A\* the decision node; for each value of A\*, create a branch and **recurse** on the subset of examples.
6. Stop when leaves are pure or attributes exhausted.

- It's **greedy** (never backtracks) and prefers **shorter trees** (Occam's Razor).
- For PlayTennis: **Outlook** is chosen as root (Gain = 0.246).

---

## M3.2 Entropy & Information Gain (concepts + equations)

- **Entropy** = measure of impurity/disorder of a set. **H(S) = −Σ pᵢ log₂ pᵢ.** Pure set → 0; max mixed → 1 (for 2 classes).
- **Information Gain** = expected reduction in entropy after splitting on attribute A. **Gain(S,A) = H(S) − Σᵥ (|Sᵥ|/|S|) H(Sᵥ).**
- ID3 chooses the attribute with the **maximum gain** at each node.
- (PlayTennis: H(S)=0.940, Gain(Outlook)=0.246, Gain(Humidity)=0.152, Gain(Wind)=0.048 → Outlook wins.)

---

## M3.3 Issues in Decision Tree Learning + Continuous Attributes

**Issues (list them):**
1. **Overfitting** (biggest issue) → fix with **pruning**.
2. **Continuous-valued attributes** → use threshold splits.
3. **Attribute selection bias** — Gain favours many-valued attributes → use **GainRatio**.
4. **Missing attribute values** → use most-common value or fractional branching.
5. **Attributes with different costs** → use cost-penalised gain.

**Handling continuous attributes:**
- Sort examples by the attribute value.
- Find candidate **thresholds** midway between adjacent examples that have **different classes**.
- Create a binary test **(A ≤ T) vs (A > T)**.
- Pick the threshold T that gives the **highest Information Gain**, then treat it like a normal attribute.

---

## M3.4 Inductive Bias of Decision Tree Learning (ID3)

- ID3's bias is a **preference (search) bias**, NOT a restriction bias — it can represent any function but **prefers** certain trees.
- **Two parts:**
  1. **Shorter trees are preferred** over longer ones.
  2. **Attributes with high information gain are placed near the root.**
- This = **Occam's Razor** ("prefer the simplest hypothesis that fits the data").
- (Contrast: Candidate Elimination has a **restriction bias** — limited hypothesis space.)

---

## M3.5 Reduced Error Pruning & Rule Post-Pruning (avoid overfitting)

**Overfitting:** tree fits training data (incl. noise) too well → poor on new data.

**Reduced Error Pruning:**
- Split data into training + **validation** set.
- For each internal node, try replacing its subtree with a **leaf (majority class)**.
- If accuracy on the validation set **does not drop**, keep the pruning.
- Repeat until no more useful pruning. (Works bottom-up.)

**Rule Post-Pruning (used in C4.5):**
1. Grow the full tree.
2. Convert each **root-to-leaf path into an IF-THEN rule**.
3. **Prune each rule independently** — remove a condition if it improves/keeps accuracy.
4. **Sort rules** by accuracy; apply in that order.

**3 advantages of converting tree → rules before pruning:**
1. Each rule pruned **independently** (no effect on others).
2. Removes the distinction of attribute order near root vs leaf.
3. Improves readability.

---

## 🔢 MODULE 3 NUMERICALS (full solutions)

*Useful log₂ values:* log₂(½)=−1, log₂(⅓)=−1.585, log₂(⅔)=−0.585, log₂(¼)=−2, log₂(¾)=−0.415, log₂(⅕)=−2.322, log₂(⅖)=−1.322, log₂(⅗)=−0.737, log₂(⅘)=−0.322, log₂(3/7)=−1.222, log₂(4/7)=−0.807, log₂(6/7)=−0.222, log₂(1/7)=−2.807.

### (A) PlayTennis — Entropy + Information Gain (14 examples: 9 Yes, 5 No)

**Step 1 — Entropy of whole set S:**
H(S) = −(9/14)log₂(9/14) − (5/14)log₂(5/14)
= −(0.643)(−0.637) − (0.357)(−1.486)
= 0.410 + 0.530 = **0.940 bits**

**Step 2 — Gain(S, Outlook).** Split: Sunny(2Y,3N)=5, Overcast(4Y,0N)=4, Rain(3Y,2N)=5.
- H(Sunny) = −(2/5)log₂(2/5) − (3/5)log₂(3/5) = (0.4)(1.322)+(0.6)(0.737) = 0.529+0.442 = **0.971**
- H(Overcast) = 0 (pure — all Yes)
- H(Rain) = −(3/5)log₂(3/5) − (2/5)log₂(2/5) = **0.971** (same as Sunny)

Gain(Outlook) = 0.940 − [(5/14)(0.971) + (4/14)(0) + (5/14)(0.971)]
= 0.940 − [0.347 + 0 + 0.347] = 0.940 − 0.694 = **0.246 bits**

**Step 3 — Gain(S, Humidity).** Split: High(3Y,4N)=7, Normal(6Y,1N)=7.
- H(High) = −(3/7)log₂(3/7) − (4/7)log₂(4/7) = (0.429)(1.222)+(0.571)(0.807) = 0.524+0.461 = **0.985**
- H(Normal) = −(6/7)log₂(6/7) − (1/7)log₂(1/7) = (0.857)(0.222)+(0.143)(2.807) = 0.190+0.401 = **0.592**

Gain(Humidity) = 0.940 − [(7/14)(0.985) + (7/14)(0.592)] = 0.940 − [0.492+0.296] = 0.940 − 0.789 = **0.152 bits**

**(For completeness:** Gain(Wind)=0.048, Gain(Temperature)=0.029.)

**Step 4 — Decide root:** Highest gain = **Outlook (0.246)** → chosen as the root node.

Final tree: Outlook → {Sunny→test Humidity (High=No, Normal=Yes); Overcast→Yes; Rain→test Wind (Strong=No, Weak=Yes)}.

---

### (B) Customer Records — Entropy & Gain on "Discount Offered" (10 examples: 6 Yes, 4 No)

Discount=Yes: 4 purchased, 1 not (5 total). Discount=No: 2 purchased, 3 not (5 total).

**Step 1 — H(S):** = −(6/10)log₂(6/10) − (4/10)log₂(4/10) = (0.6)(0.737)+(0.4)(1.322) = 0.442+0.529 = **0.971 bits**

**Step 2 — Subset entropies:**
- H(Discount=Yes): −(4/5)log₂(4/5) − (1/5)log₂(1/5) = (0.8)(0.322)+(0.2)(2.322) = 0.258+0.464 = **0.722**
- H(Discount=No): −(2/5)log₂(2/5) − (3/5)log₂(3/5) = (0.4)(1.322)+(0.6)(0.737) = 0.529+0.442 = **0.971**

**Step 3 — Gain:** = 0.971 − [(5/10)(0.722) + (5/10)(0.971)] = 0.971 − [0.361+0.486] = 0.971 − 0.847 = **0.124 bits**

**Conclusion:** Gain = 0.124 > 0, so "Discount Offered" IS a useful (appropriate) attribute for splitting, but only moderately discriminating (the Discount=No branch is still impure, 0.971).

---

### (C) Boolean Function Decision Trees

Rule: test ONE variable per node, branch on its value (Yes/No or T/F), leaf = output.

**(a) A ∨ (B ∧ C):** Test A → A=Yes: **Yes**. A=No: test B → B=No: **No**; B=Yes: test C → C=Yes: **Yes**, C=No: **No**.

**(b) (A ∧ B) ∨ (C ∧ D):** Test A → A=Yes: test B (B=Yes:**Yes**; B=No: test C→D...). A=No: test C → C=Yes: test D (D=Yes:**Yes**, D=No:**No**); C=No:**No**.

**(c) A ∧ ¬B:** Test A → A=No: **No**. A=Yes: test B → B=Yes: **No**, B=No: **Yes**.

**(d) A XOR B:** Test A → A=No: test B (B=Yes:**Yes**, B=No:**No**). A=Yes: test B (B=Yes:**No**, B=No:**Yes**). (Exactly one true → Yes.)

---
---

# 🟧 MODULE 4 — BAYESIAN LEARNING

**Bayes Theorem (memorise):** **P(h|D) = P(D|h) · P(h) / P(D)**
- P(h) = prior, P(D|h) = likelihood, P(h|D) = posterior, P(D) = evidence.
- **P(D) = Σ P(D|hᵢ)P(hᵢ)** (normaliser).

---

## M4.1 ⭐ Naive Bayes Classifier

- **Idea:** classify using Bayes theorem + the **"naive" assumption** that attributes are **conditionally independent given the class**.
- **Formula:** **v_NB = argmax_v [ P(v) · ∏ᵢ P(aᵢ|v) ]**
- **Estimate from data:** P(v) = count(v)/total; P(aᵢ|v) = count(aᵢ and v)/count(v).
- **Laplace smoothing** (avoid zero probability): P(aᵢ|v) = (count + 1)/(count(v) + number of values).
- **Steps:** (1) compute priors P(v), (2) compute conditionals P(aᵢ|v), (3) for the new instance compute the product for each class, (4) pick the highest.
- **Used in:** spam filtering, text classification, sentiment analysis. Works well even when independence is not perfectly true.

---

## M4.2 Bayes Theorem — Features, Difficulties, Relevance

**Theorem:** P(h|D) = P(D|h)P(h)/P(D). Updates prior belief into posterior after seeing data.

**Features:**
- Combines prior knowledge with observed data.
- Gives probabilities (handles uncertainty), not just a single answer.
- Can combine multiple hypotheses (Bayes optimal classifier).

**Practical difficulties:**
1. Needs **prior probabilities** P(h) — often unknown / subjective.
2. **Computationally expensive** — must consider many hypotheses.
3. Needs accurate likelihood P(D|h).

**Relevance:**
- Provides a **gold standard** (Bayes optimal) for comparing algorithms.
- Naive Bayes, Bayesian networks, EM are practical and widely used.
- Explains other methods (e.g., least-squares = ML under Gaussian noise).

---

## M4.3 Bayesian Belief Networks (BBN) + Conditional Independence

- **BBN** = a **Directed Acyclic Graph (DAG)**: nodes = random variables, edges = direct dependencies. Each node has a **Conditional Probability Table (CPT)** = P(node | parents).
- **Conditional independence assumption (Markov condition):** each node is **conditionally independent of its non-descendants given its parents**.
- **Joint probability:** **P(x₁,…,xₙ) = ∏ᵢ P(xᵢ | Parents(xᵢ))** ← this is what makes BBNs compact.
- **Example (Alarm):** Burglary→Alarm←Earthquake, Alarm→JohnCalls, Alarm→MaryCalls. Given Alarm, JohnCalls is independent of Burglary.
- **Inference:** compute probability of query variables given evidence (e.g., P(Burglary | JohnCalls=T)).
- **Learning:** estimate CPTs from data (counting), or use EM if data has missing values.

```
 Burglary    Earthquake
      \         /
       v       v
         Alarm
        /     \
       v       v
  JohnCalls  MaryCalls
```

---

## M4.4 EM (Expectation-Maximization) Algorithm

- **Use:** finding Maximum Likelihood estimates when data has **hidden/missing (latent) variables** Z.
- **Two steps, repeated until convergence:**
  - **E-step (Expectation):** using current parameters h', estimate the expected values of the hidden variables → compute Q(h, h') = E[ln P(D,Z | h) | D, h'].
  - **M-step (Maximization):** update parameters h ← argmax Q(h, h') (maximise expected likelihood).
- **Property:** likelihood **never decreases** each iteration; converges to a **local maximum**.
- **Example:** Gaussian Mixture Models (clustering) — E-step assigns soft cluster memberships, M-step updates cluster means/variances.

```
 Initialise params → E-step (estimate hidden) → M-step (update params)
                          ↑__________________________↓  repeat until stable
```

---

## M4.5 MAP and ML Hypothesis — Definitions + Derivation

- **MAP (Maximum A Posteriori):** the most probable hypothesis given the data.
  - **h_MAP = argmax P(h|D) = argmax [ P(D|h) · P(h) ]** (P(D) dropped — same for all h).
- **ML (Maximum Likelihood):** the hypothesis that makes the data most likely.
  - **h_ML = argmax P(D|h)**.
- **Derivation:** Start from h_MAP = argmax P(D|h)P(h)/P(D). Since P(D) is constant → h_MAP = argmax P(D|h)P(h). If we assume **all priors equal** (uniform P(h)) → the P(h) term drops → **h_ML = argmax P(D|h)**.
- **So ML = MAP with a uniform prior.**

---

## M4.6 Minimum Description Length (MDL) Principle

- **Idea (Occam's Razor in bits):** prefer the hypothesis that **minimises the total description length** of (the hypothesis) + (the data given the hypothesis).
- **Equation:** **h_MDL = argmin [ L_C(h) + L_C(D|h) ]**
  - L_C(h) = bits to encode hypothesis, L_C(D|h) = bits to encode data using h.
- **Derivation:** from h_MAP = argmax [log P(D|h) + log P(h)] = argmin [−log P(D|h) − log P(h)]. Using Shannon: optimal code length = −log₂ P. So **−log P(h) = L_C(h)** and **−log P(D|h) = L_C(D|h)** → h_MDL = argmin[L_C(h)+L_C(D|h)].
- **Trade-off:** simple hypothesis (short L_C(h)) vs good fit (short L_C(D|h)). → avoids overfitting.

---

## 🔢 MODULE 4 NUMERICALS (full solutions)

**General MAP method:** for each hypothesis compute (prior × likelihood) = P(h)·P(D|h); pick the largest = h_MAP. To get actual probabilities, normalise: divide each by the sum.

### (A) Cancer Diagnosis (MAP)

Given: P(cancer)=0.008, P(¬cancer)=0.992. P(+|cancer)=0.98, P(+|¬cancer)=1−0.97=0.03. Test result = **positive**.

- P(+|cancer)·P(cancer) = 0.98 × 0.008 = **0.00784**
- P(+|¬cancer)·P(¬cancer) = 0.03 × 0.992 = **0.02976**

Normalise: total = 0.00784 + 0.02976 = 0.0376
- P(cancer|+) = 0.00784 / 0.0376 = **0.2085 (≈20.9%)**
- P(¬cancer|+) = 0.02976 / 0.0376 = **0.7915 (≈79.1%)**

**h_MAP = ¬cancer.** Despite a positive test, the patient most likely does NOT have cancer — because the disease is very rare (low prior), false positives outnumber true positives.

---

### (B) Cricket Match (MAP)

Given: P(T0 wins)=0.95, P(T1 wins)=0.05. P(at T1's field | T0 wins)=0.30, P(at T1's field | T1 wins)=0.75. Match is **at Team 1's field**.

- P(field|T0)·P(T0) = 0.30 × 0.95 = **0.285**
- P(field|T1)·P(T1) = 0.75 × 0.05 = **0.0375**

Normalise: total = 0.285 + 0.0375 = 0.3225
- P(T0 wins | field) = 0.285 / 0.3225 = **0.8837 (≈88.4%)**
- P(T1 wins | field) = 0.0375 / 0.3225 = **0.1163 (≈11.6%)**

**h_MAP = Team 0 wins** — Team 0's strong overall record (0.95 prior) outweighs Team 1's home advantage.

---

### (C) Spam Classifier (MAP)

Given: P(spam)=0.80, P(¬spam)=0.20. P(phrase|spam)=0.95, P(phrase|¬spam)=0.05. Email **contains the phrase**.

- P(phrase|spam)·P(spam) = 0.95 × 0.80 = **0.76**
- P(phrase|¬spam)·P(¬spam) = 0.05 × 0.20 = **0.01**

Normalise: total = 0.76 + 0.01 = 0.77
- P(spam|phrase) = 0.76 / 0.77 = **0.987 (≈98.7%)**
- P(¬spam|phrase) = 0.01 / 0.77 = **0.013 (≈1.3%)**

**h_MAP = Spam.** Both the prior and likelihood favour spam → very high posterior.

---

### (D) Naive Bayes — PlayTennis, classify ⟨Outlook=Sunny, Temp=Cool, Humidity=High, Wind=Strong⟩

Dataset: 9 Yes, 5 No. **Priors:** P(Yes)=9/14, P(No)=5/14.

**Conditional probabilities (count within each class):**
- Among 9 Yes: Sunny=2/9, Cool=3/9, High=3/9, Strong=3/9.
- Among 5 No: Sunny=3/5, Cool=1/5, High=4/5, Strong=3/5.

**Score(Yes)** = (9/14)·(2/9)·(3/9)·(3/9)·(3/9)
= 0.643 × 0.222 × 0.333 × 0.333 × 0.333 = **0.00529**

**Score(No)** = (5/14)·(3/5)·(1/5)·(4/5)·(3/5)
= 0.357 × 0.6 × 0.2 × 0.8 × 0.6 = **0.02057**

Normalise: total = 0.00529 + 0.02057 = 0.02586
- P(Yes) = 0.00529/0.02586 = **0.205 (≈20.5%)**
- P(No) = 0.02057/0.02586 = **0.795 (≈79.5%)**

**Answer: PlayTennis = No** (do not play).

---

### (E) Naive Bayes — Car Stolen, classify ⟨Color=Red, Type=SUV, Origin=Domestic⟩

Dataset: 5 Yes (stolen), 5 No. **Priors:** P(Yes)=5/10=0.5, P(No)=0.5.

**Conditionals:**
- Among 5 Yes: Red=3/5, SUV=1/5, Domestic=2/5.
- Among 5 No: Red=2/5, SUV=3/5, Domestic=3/5.

**Score(Yes)** = 0.5 × (3/5) × (1/5) × (2/5) = 0.5 × 0.6 × 0.2 × 0.4 = **0.024**
**Score(No)** = 0.5 × (2/5) × (3/5) × (3/5) = 0.5 × 0.4 × 0.6 × 0.6 = **0.072**

Normalise: total = 0.024 + 0.072 = 0.096
- P(Yes) = 0.024/0.096 = **0.25 (25%)**
- P(No) = 0.072/0.096 = **0.75 (75%)**

**Answer: Stolen = No** (car will NOT be stolen).

---

### (F) Bayesian Network — Cloudy(C) → Rain(R) → WetGrass(W)

Given: P(C=T)=0.6. P(R=T|C=T)=0.7, P(R=T|C=F)=0.2. P(W=T|R=T)=0.9, P(W=T|R=F)=0.1.

**(i) Joint P(C=T, R=T, W=T):** Use chain rule P(C)·P(R|C)·P(W|R):
= 0.6 × 0.7 × 0.9 = **0.378**

**(ii) Marginal P(W=T):** First find P(R=T):
P(R=T) = P(R=T|C=T)P(C=T) + P(R=T|C=F)P(C=F) = (0.7)(0.6) + (0.2)(0.4) = 0.42 + 0.08 = **0.50**
So P(R=F) = 0.50.

Then: P(W=T) = P(W=T|R=T)P(R=T) + P(W=T|R=F)P(R=F)
= (0.9)(0.50) + (0.1)(0.50) = 0.45 + 0.05 = **0.50**

**Answer: P(W=T) = 0.50.**

---
---

# 🟥 MODULE 5 — INSTANCE-BASED LEARNING

**Euclidean distance:** d(xᵢ,xⱼ) = √[ Σᵣ (aᵣ(xᵢ) − aᵣ(xⱼ))² ]

---

## M5.1 ⭐ k-Nearest Neighbour (kNN) Algorithm

- **Idea:** a **lazy learner** — stores all data, classifies a new point by looking at its **k closest** stored examples.
- **Discrete (classification):** **f̂(xq) = majority class among the k nearest** → f̂(xq) = argmax_v Σ δ(v, f(xᵢ)).
- **Continuous (regression):** **f̂(xq) = mean of the k nearest values** → f̂(xq) = (1/k) Σ f(xᵢ).
- **Steps:** (1) compute distance from query to all training points, (2) sort, (3) take k nearest, (4) vote (discrete) or average (continuous).
- **Pseudocode:**
```
Training: store all examples.
Classify xq:
  for each xi: compute d(xq, xi)
  pick k nearest
  discrete  → return majority class
  continuous→ return average value
```
- **1-NN** implicitly builds a **Voronoi diagram** (decision boundary).

---

## M5.2 Locally Weighted Regression (LWR)

- **Idea:** generalises kNN — instead of averaging, it **fits an explicit (linear) function** over the neighbourhood of the query, **weighting nearby points more**.
- **Local linear model:** f̂(x) = w₀ + w₁a₁(x) + … + wₙaₙ(x).
- **Weighted error to minimise:** E = ½ Σᵢ K(d(xq,xᵢ)) (f(xᵢ) − f̂(xᵢ))²
- **Kernel (Gaussian):** K(d) = e^(−d²/2σ²) — closer points get more weight.
- A **new local model is fitted for each query** → captures local trends better than kNN (which gives a flat average).

---

## M5.3 Radial Basis Function (RBF) + RBF Network

- **RBF:** a function whose value depends only on **distance from a centre** xᵤ. Gaussian: Kᵤ(d) = e^(−d²/2σᵤ²).
- **RBF Network output:** **f(x) = w₀ + Σᵤ wᵤ Kᵤ(d(xᵤ, x))**
- **Architecture = 2 layers:**
  - **Input layer** — one node per attribute.
  - **Hidden layer** — kernel units (each a Gaussian centred at some xᵤ).
  - **Output layer** — linear weighted sum of the hidden units.
- **Training (2 stages):** (1) choose centres/widths (e.g., k-means or one per example), (2) solve the linear output weights wᵤ.
- It's an **eager** learner (unlike kNN). Builds a global approximation from many local "bumps".

```
 Inputs       Hidden (RBF kernels)     Output
  x1 ─┐        (K1)─w1─┐
  x2 ─┼──────► (K2)─w2─┼──► f(x)=w0+Σ wu Ku
  x3 ─┘        (K3)─w3─┘
```

---

## M5.4 Case-Based Reasoning (CBR) + CADET

- **CBR:** instance-based method using **rich symbolic case descriptions** (not numeric distance). Solves new problems by **retrieving and adapting** similar past cases.
- **CBR cycle (4 R's):** **Retrieve → Reuse → Revise → Retain.**
- **CADET:** a CBR system for **designing mechanical devices** (e.g., water faucets). Each case = ⟨**function**, **structure**⟩.
- **T-junction pipe example:** a stored case whose function is "merge two flows into one" (Qm = Qc + Qh). To design a faucet (mix hot + cold water), CADET **retrieves the T-junction**, reuses its flow-merging structure, adds temperature-control parts, revises, and retains the new design.
- Uses **graph/structural matching**, not Euclidean distance — that's why plain kNN can't do it.

---

## M5.5 Instance-Based Learning + Advantages/Disadvantages

- **Instance-Based (lazy) learning:** stores training examples, does **no work until a query arrives**, then forms a **local approximation**. Examples: kNN, LWR, CBR.
- **Lazy vs Eager:** Lazy = build model at query time, per-query local model (kNN, LWR, CBR). Eager = build one global model at training time (decision trees, neural nets, Naive Bayes, RBF).
- **Advantages:**
  1. Simple; no training cost.
  2. Can model **complex/local** functions (different approximation per query).
  3. Easy to add new data (incremental).
- **Disadvantages:**
  1. **Slow at query time** (compute distance to all points).
  2. **High memory** (stores all data).
  3. **Curse of dimensionality** — irrelevant attributes distort distance.
  4. Sensitive to attribute scaling → needs normalisation.

**Bonus quick definitions (Module 5):**
- **Regression** = approximating a real-valued (continuous) target.
- **Residual** = (actual − predicted) = yᵢ − ŷᵢ.
- **Kernel function** = weight that decreases with distance, K(d) = e^(−d²/2σ²).

**kNN drawbacks + fixes (if asked):** curse of dimensionality → attribute weighting/feature selection; slow queries → kd-trees; large memory → keep only prototypes; scaling → normalise; noise → distance weighting / larger k.

---

## 🔢 MODULE 5 NUMERICALS (full solutions)

### (A) kNN Classification — Height/Weight, classify Query (Height=170, Weight=57)

Distance formula: d = √[(H − 170)² + (W − 57)²].

| Point | (H, W) | Calculation | Distance | Class |
|-------|--------|-------------|----------|-------|
| D7 | (169,58) | √(1+1)=√2 | **1.41** | Normal |
| D9 | (170,55) | √(0+4)=√4 | **2.00** | Normal |
| D8 | (173,57) | √(9+0)=√9 | **3.00** | Normal |
| D6 | (174,56) | √(16+1)=√17 | 4.12 | Underweight |
| D1 | (167,51) | √(9+36)=√45 | 6.71 | Underweight |
| D4 | (173,64) | √(9+49)=√58 | 7.62 | Normal |
| D5 | (172,65) | √(4+64)=√68 | 8.25 | Normal |
| D2 | (182,62) | √(144+25)=√169 | 13.00 | Normal |
| D3 | (176,69) | √(36+144)=√180 | 13.42 | Normal |

**K=3 nearest:** D7, D9, D8 → all **Normal** → vote = Normal.
**K=5 nearest:** D7, D9, D8 (Normal), D6, D1 (Underweight) → 3 Normal vs 2 Underweight → Normal.

**Answer: Class = Normal** (robust for both k=3 and k=5).

---

### (B) kNN Regression — predict Weight of Rani (Age=11, Gender=1)

This is **regression** (Weight is continuous) → predict by **averaging** the k nearest neighbours' weights.
Distance: d = √[(Age − 11)² + (Gender − 1)²].

| Name | (Age,Gen) | Calculation | Distance | Weight |
|------|-----------|-------------|----------|--------|
| Smith | (15,0) | √(16+1)=√17 | **4.12** | 42.2 |
| Sanjana | (16,1) | √(25+0)=√25 | **5.00** | 40.0 |
| Pooja | (20,1) | √(81+0)=√81 | **9.00** | 50.2 |
| Anjali | (32,0) | √(441+1) | 21.02 | 55.2 |
| Anjana | (34,1) | √(529+0) | 23.00 | 50.7 |
| Madhu | (40,0) | √(841+1) | 29.02 | 48.5 |
| Rahul | (40,0) | √(841+1) | 29.02 | 48.5 |
| Laxmi | (55,1) | √(1936+0) | 44.00 | 67.0 |
| Saket | (55,0) | √(1936+1) | 44.01 | 75.6 |

**K=3 nearest:** Smith(42.2), Sanjana(40.0), Pooja(50.2).
Predicted weight = (42.2 + 40.0 + 50.2) / 3 = 132.4 / 3 = **44.13 kg**.

(K=1 → Smith → 42.2 kg.) Note: Age dominates distance because its range is far larger than Gender (0/1); ideally normalise both attributes first.

---

### (C) Locally Weighted Regression — predict at X=2.5, Bandwidth τ=1

Data: X = [1,2,3,4], Y = [2,3,2,5]. Kernel weight: wᵢ = e^(−(xᵢ−2.5)² / (2·1²)).

| xᵢ | (xᵢ−2.5)² | wᵢ = e^(−d²/2) | yᵢ | wᵢ·yᵢ |
|----|-----------|----------------|----|-------|
| 1 | 2.25 | e^(−1.125)=**0.3247** | 2 | 0.6494 |
| 2 | 0.25 | e^(−0.125)=**0.8825** | 3 | 2.6475 |
| 3 | 0.25 | e^(−0.125)=**0.8825** | 2 | 1.7650 |
| 4 | 2.25 | e^(−1.125)=**0.3247** | 5 | 1.6235 |

Σwᵢ = 0.3247 + 0.8825 + 0.8825 + 0.3247 = **2.4144**
Σwᵢyᵢ = 0.6494 + 2.6475 + 1.7650 + 1.6235 = **6.6854**

**Prediction (weighted average):** f̂(2.5) = Σwᵢyᵢ / Σwᵢ = 6.6854 / 2.4144 = **2.77**

(If you do the full weighted linear fit instead: ŷ = 1.43 + 0.54x → at x=2.5 gives 1.43 + 1.34 = **2.77** — same answer, due to symmetry of the weights around 2.5.)

**Answer: f̂(2.5) ≈ 2.77.**

---
---

# 🧒 PLAIN-ENGLISH EXAMPLES (assume I know nothing)
### Every topic with a tiny real-life example + every term explained simply.

> Read this if the formal answers feel confusing. Each one is the SAME idea, told simply.

---

## 🟦 MODULE 1 examples

**M1.1 Checkers / T,P,E** — Think of teaching a kid to play a game.
- **Task (T)** = the game itself (play checkers).
- **Performance (P)** = how we measure if the kid is good (how many games they win).
- **Experience (E)** = the practice they get (playing lots of games).
- The computer can't "see" a good move directly, so it learns a **score for each board** (V). A board with more of my pieces = high score. It plays toward high-score boards.
- It fixes its scoring using: *new weight = old weight + (small step) × (how wrong it was) × (feature value)*.

**M1.3 Types of learning** — by how the data is labelled.
- **Supervised** = you study with an answer key. Example: 100 emails already marked "spam"/"not spam" → learn to mark new ones.
- **Unsupervised** = no answer key, just group similar things. Example: a shop groups customers into "big spenders" and "bargain hunters" without being told the groups.
- **Reinforcement** = learn by reward/punishment. Example: a dog gets a treat for sitting → learns to sit.

**M1.6 DIKW pyramid** — example with one number.
- **Data**: "37" (just a number, meaningless).
- **Information**: "Body temperature = 37°C" (now it has context).
- **Knowledge**: "37°C is normal, the person is healthy."
- **Intelligence/Wisdom**: "No medicine needed; keep monitoring."

---

## 🟩 MODULE 2 examples

**The setup (EnjoySport):** We watch a boy, **Aldo**. Some days he plays water-sport (😀 = **positive**), some days he doesn't (☹️ = **negative**). We describe each day by: Sky, AirTemp, Humidity, Wind, Water, Forecast. We want a rule for "when does Aldo play?"

- A **hypothesis** = a guessed rule, e.g. ⟨Sunny, Warm, ?, Strong, ?, ?⟩ means *"Aldo plays if Sky=Sunny AND AirTemp=Warm AND Wind=Strong, regardless of the others (the ? marks)."*
- **"?"** = "any value is fine here." **"∅"** = "nothing fits here" (the empty/start rule).

**M2.2 FIND-S** — "start strict, loosen only when forced."
- Start: ⟨∅,∅,∅,∅,∅,∅⟩ (impossibly strict — plays on no day).
- See a 😀 day (Sunny,Warm,High,Strong,Warm,Same) → rule becomes exactly that day.
- See another 😀 day (Sunny,Warm,**Normal**,Strong,Warm,Same) → Humidity now differs → replace with "?" → ⟨Sunny,Warm,**?**,Strong,Warm,Same⟩.
- Ignore all ☹️ days. Final = the most specific rule that covers all 😀 days.

**M2.1 Candidate Elimination** — keep TWO rules: the strictest (S) and the loosest (G), and squeeze them together.
- 😀 day → loosen S a bit (so it covers the happy day).
- ☹️ day → tighten G a bit (so it rejects the sad day).
- Everything between S and G = all rules still possible = the **version space**.

**M2.5 Unbiased learner is useless** — tiny example. Suppose you've only seen 2 days. A brand-new day appears. With no assumptions, half the surviving rules say "plays", half say "doesn't" — a 50/50 coin flip. So you've learned nothing about new days. → You NEED assumptions (bias) to predict.

---

## 🟨 MODULE 3 examples

**The setup (PlayTennis):** 14 days, each labelled "Play=Yes" or "Play=No". Attributes: Outlook, Temperature, Humidity, Wind. We build a **flowchart (tree)** of yes/no questions ending in Play=Yes/No.

**M3.2 Entropy** = "how mixed is the group?"
- A bag of 10 balls, all red → entropy **0** (totally pure, no surprise).
- 5 red + 5 blue → entropy **1** (maximum mix, total uncertainty).
- Formula just turns the mix-ratio into this 0-to-1 number.

**M3.2 Information Gain** = "how much does asking this question clean up the mess?"
- Before splitting: messy group (entropy 0.94).
- Ask "Is Outlook Sunny/Overcast/Rain?" → the Overcast branch becomes ALL Yes (pure!). Overall mess drops to 0.69.
- Gain = 0.94 − 0.69 = **0.25**. Outlook cleaned up the most → it becomes the **first question (root)** of the tree.

**M3.1 ID3** = "keep asking the most informative question first."
- Pick the attribute with the highest gain → make it a node → split → repeat on each branch → stop when a branch is pure (all Yes or all No).

**M3.5 Overfitting & pruning** — example.
- A tree that memorises *every* training day perfectly (even one weird noisy day) will fail on new days. That's **overfitting** (like memorising answers instead of understanding).
- **Pruning** = chop off the over-detailed branches and replace with a simple "most common answer" leaf, if it doesn't hurt accuracy on a fresh test set.

---

## 🟧 MODULE 4 examples

**M4.2 Bayes Theorem** = "update your belief after seeing evidence."
- Plain words: **(belief after) = (belief before) × (how well evidence fits) ÷ (total chance of evidence).**
- Terms: **Prior** = belief before (P(h)). **Likelihood** = how well h explains the data (P(D|h)). **Posterior** = updated belief (P(h|D)).

**M4.1 Naive Bayes** — classic example: is an email **Spam** or **Ham (not spam)**?
- "Naive" = we pretend each word appears independently of the others (a simplification that works well anyway).
- For a new email, compute: P(Spam) × P(word1|Spam) × P(word2|Spam)... and the same for Ham. **Bigger number wins.**
- Example: email says "Win Prize". If spam emails very often contain "Win" and "Prize", the Spam score is high → label **Spam**.

**The cancer example (why priors matter)** — the eye-opener:
- A test is 98% accurate, your result is **positive**. Scary, right?
- BUT only 0.8% of people actually have the disease (the **prior** is tiny).
- After Bayes: P(actually sick) ≈ **21%**. So most likely you're FINE — because false alarms from the huge healthy crowd outnumber true cases. → *Never ignore the base rate.*

**M4.3 Bayesian Network** — a diagram of "what causes what."
- Example: **Burglary** or **Earthquake** can set off an **Alarm**; the Alarm makes **John** and **Mary** call you.
- "Conditional independence": once you KNOW the alarm is ringing, whether John calls doesn't depend on the burglary anymore — the alarm already tells the story.

**M4.4 EM algorithm** — "guess, improve, repeat" when some info is hidden.
- Example: you have people's heights but DON'T know who is male/female (hidden label). 
- **E-step:** guess each person's gender from current average heights. **M-step:** recompute male/female average heights from those guesses. Repeat until stable. → discovers the two groups.

---

## 🟥 MODULE 5 examples

**M5.1 kNN** = "you are like your closest neighbours."
- Example: to guess if a new fruit is an apple or orange, look at the **k** most similar fruits you already know (by size/colour) and take a **majority vote**.
- **Lazy** = it does nothing until you ask a question, then compares to all stored examples.
- **Distance** = a number for "how different" two things are (small = similar).

**M5.1 kNN regression** — same idea but predict a **number** instead of a label.
- To guess a house price, average the prices of the 3 most similar houses nearby. That average = your prediction.

**M5.2 Locally Weighted Regression** = "kNN, but draw a little trend-line near the query."
- Instead of just averaging neighbours, fit a small straight line through the nearby points (closer points pulled harder) and read the value off that line. Captures local slope, not just a flat average.

**M5.3 RBF network** = "many bell-shaped detectors added up."
- Each hidden unit is a **bell curve (Gaussian)** centred at some point — it lights up only when the input is near its centre. The output adds these bells with weights → builds any smooth shape. (Like covering a hilly landscape with lots of little round tents.)

**M5.4 Case-Based Reasoning (CADET)** = "solve new problems by reusing old solutions."
- Example: designing a water tap that mixes hot + cold. The system **retrieves** a stored "T-junction pipe" design (which merges two flows), **reuses/adapts** it, **revises** if needed, and **retains** the final design for next time (the 4 R's).
- Unlike kNN, it compares **structures/designs**, not just numbers.

**M5.5 Lazy vs Eager** — exam-style one-liner:
- **Lazy** (kNN) = student who only opens the book when asked a question (no prep, slow answer).
- **Eager** (decision tree, neural net) = student who studies everything beforehand (slow prep, instant answer).

---

# 🎯 ONE-PAGE PANIC SHEET (read this last, in the exam hall)

| Module | Must-know formula | One-line answer |
|--------|-------------------|-----------------|
| **M1** | V_train(b)←V̂(Successor(b)); wᵢ←wᵢ+η(V_train−V̂)xᵢ | Checkers = T,P,E + 4 modules (Perf System, Critic, Generalizer, Exp Generator) |
| **M2** | VS between S and G | CE: positive→generalise S, negative→specialise G |
| **M3** | H=−Σp log₂p; Gain=H(S)−Σ(|Sᵥ|/|S|)H(Sᵥ) | ID3 picks highest-gain attribute as root |
| **M4** | P(h|D)=P(D|h)P(h)/P(D); v_NB=argmax P(v)∏P(aᵢ|v) | MAP = argmax P(D|h)P(h); ML = uniform prior |
| **M5** | d=√Σ(aᵣ−aᵣ)²; f(x)=w₀+ΣwᵤKᵤ | kNN = vote/average of k nearest; lazy learner |

**Remember the patterns:**
- "Positive → generalise S, Negative → specialise G" (Candidate Elimination)
- "Highest Information Gain → root" (ID3)
- "Prior × Likelihood → pick biggest" (MAP / Naive Bayes)
- "k nearest → vote or average" (kNN)
- "Occam's Razor = prefer simplest" (appears in M2 bias, M3 ID3 bias, M4 MDL)

**You know more than you think. Write the definition, the formula, then 3–4 bullets. That's a full-marks answer. Go get it. 🚀**

*CML23604C — Crash Revision — Global Academy of Technology, Bengaluru — VTU*
