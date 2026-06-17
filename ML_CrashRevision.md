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

## 🔢 MODULE 2 NUMERICALS (fast method)

**(A) Counting** (attribute i has nᵢ values):
- Instances = ∏ nᵢ → EnjoySport: 3×2×2×2×2×2 = **96**
- Syntactic hypotheses = ∏ (nᵢ + 2) → 5×4×4×4×4×4 = **5120**
- Semantic hypotheses = ∏ (nᵢ + 1) + 1 → 4×3×3×3×3×3 + 1 = **973**

**(B) FIND-S trace** — start ⟨∅,…,∅⟩, for each **positive** example keep matching attributes, replace mismatches with "?", skip negatives. (Face example answer: **⟨?,?,?,?,Yes,?⟩**.)

**(C) Candidate Elimination trace** — keep S and G tables; positive → generalise S; negative → specialise G. (If no hypothesis fits all data → **empty version space** = concept not representable in H.)

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

## 🔢 MODULE 3 NUMERICALS (fast method)

**(A) Entropy + Info Gain:**
1. H(S) = −Σ pᵢ log₂ pᵢ for the whole set.
2. For attribute A: split into subsets, compute each H(Sᵥ), then Gain = H(S) − Σ(|Sᵥ|/|S|)H(Sᵥ).
3. Highest gain wins.
- *Useful log₂ values:* log₂(½)=−1, log₂(⅓)=−1.585, log₂(⅔)=−0.585, log₂(¼)=−2, log₂(¾)=−0.415, log₂(⅕)=−2.322, log₂(⅖)=−1.322, log₂(⅗)=−0.737, log₂(⅘)=−0.322.
- *PlayTennis pre-computed:* H(S)=0.940, Gain(Outlook)=**0.246**, Gain(Humidity)=**0.152**.

**(B) Boolean function decision trees:** test one variable per node, branch True/False, leaf = output. E.g. **A ∧ ¬B**: test A → if No: No; if Yes: test B → B=Yes: No, B=No: Yes.

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

## 🔢 MODULE 4 NUMERICALS (fast method)

**(A) MAP problems (cancer, cricket, spam)** — for each hypothesis compute **P(D|h)·P(h)**, pick the largest; normalise by dividing by their sum.

- **Cancer:** P(cancer)=0.008, P(+|cancer)=0.98, P(+|¬cancer)=0.03.
  - cancer: 0.98×0.008 = **0.00784**; ¬cancer: 0.03×0.992 = **0.02976** → **h_MAP = ¬cancer** (79% no cancer despite + test).
- **Cricket:** T0: 0.30×0.95 = **0.285**; T1: 0.75×0.05 = **0.0375** → **Team 0 wins** (88%).
- **Spam:** spam: 0.95×0.80 = **0.76**; not: 0.05×0.20 = **0.01** → **Spam** (98.7%).

**(B) Naive Bayes (PlayTennis / Car-stolen)** — compute P(class)·∏P(aᵢ|class) for each class, pick higher.
- **PlayTennis** ⟨Sunny,Cool,High,Strong⟩: Yes = (9/14)(2/9)(3/9)(3/9)(3/9)=**0.0053**; No = (5/14)(3/5)(1/5)(4/5)(3/5)=**0.0206** → **No** (79.5%).
- **Car stolen** ⟨Red,SUV,Domestic⟩: Yes=0.5×(3/5)(1/5)(2/5)=**0.024**; No=0.5×(2/5)(3/5)(3/5)=**0.072** → **No / not stolen** (75%).

**(C) BBN joint/marginal** (Cloudy→Rain→Wet):
- Joint: P(C,R,W) = P(C)·P(R|C)·P(W|R). E.g. P(C=T,R=T,W=T)=0.6×0.7×0.9 = **0.378**.
- Marginal: P(W=T) = sum over hidden vars = **0.50** (here).

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

## 🔢 MODULE 5 NUMERICALS (fast method)

**(A) kNN classification (Height/Weight):**
- Compute d = √[(H−Hq)² + (W−Wq)²] for each, sort, take k nearest, majority vote.
- *Query (170,57):* nearest 3 = D7(1.41,Normal), D9(2.0,Normal), D8(3.0,Normal) → **Normal**.

**(B) kNN regression (Rani, Age=11, Gender=1):**
- Distance on (Age, Gender); k=3 nearest = Smith(42.2), Sanjana(40.0), Pooja(50.2) → average = **44.13 kg**. (Note: Age dominates — ideally normalise.)

**(C) Locally Weighted Regression (X=2.5, BW=1):**
- Weight each point wᵢ = e^(−(xᵢ−2.5)²/2). Then prediction = Σwᵢyᵢ / Σwᵢ.
- Answer = **≈ 2.77**.

---
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
