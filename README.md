# X-TuckER

### Exact Tensor Attribution for Explainable Knowledge Graph Embeddings

X-TuckER is a model-aligned explainability framework for **TuckER-based Knowledge Graph Embedding (KGE) models**.

The project uses the mathematical structure of the Tucker decomposition to derive an **exact attribution of the model's trilinear scoring function**, exposing the individual interactions within the Tucker core that contribute to a prediction.

For a triple $(h,r,t)$, the attribution tensor is:

$$
C_{ijk}=W_{ijk}e_h[i]w_r[j]e_t[k]
$$

where each $C_{ijk}$ represents the contribution of one interaction between the head entity, relation, tail entity, and Tucker core.

Because these contributions directly arise from the model's scoring function,

$$
\sum_{i,j,k}C_{ijk}=f_{\text{raw}}(h,r,t)
$$

providing an **algebraically exact decomposition of the raw trilinear score**.

---

## Why X-TuckER?

Knowledge Graph Embedding models can achieve strong link-prediction performance while remaining difficult to interpret.

Many explainability techniques treat the trained model as a black box and estimate feature importance through gradients, perturbations, or surrogate explanations.

X-TuckER instead exploits the **internal mathematical structure of the model itself**.

This provides two complementary forms of analysis:

* **Exact attribution:** What interactions mathematically contribute to the score?
* **Faithfulness:** Do those attributed interactions actually influence the trained model's prediction when intervened upon?

The framework therefore separates **mathematical correctness** from **empirical explanation quality**.

---

## Framework

```text
                 Knowledge Graph
                       │
                       ▼
                    TuckER
                       │
              ┌────────┴────────┐
              │                 │
        Entity/Relation     Tucker Core
         Embeddings             W
              │                 │
              └────────┬────────┘
                       ▼
              Exact Attribution
                    C[i,j,k]
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Attribution   Faithfulness  Counterfactual
       Analysis      Analysis       Analysis
```

The framework supports both **tensor-level** and **relation-level** attribution by aggregating the contribution tensor across its dimensions.

---

## Explanation Methods

The project implements and evaluates the following attribution and explanation approaches:

| Method                       | Role                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------- |
| **Exact Tensor Attribution** | Exact decomposition of the TuckER trilinear computation                      |
| **Integrated Gradients**     | Gradient/path-based attribution under the specified baseline formulation     |
| **SHAP**                     | Shapley-value-based attribution                                              |
| **Gradient × Input**         | Gradient-based saliency baseline                                             |
| **Random**                   | Model-independent attribution baseline                                       |
| **CTIS**                     | Counterfactual Tensor Influence Score for tensor-level intervention analysis |

These methods provide complementary perspectives on model behaviour.

**Exact Tensor Attribution** is derived directly from the TuckER scoring function, while Integrated Gradients, SHAP, Gradient × Input, and Random provide comparison or alternative attribution perspectives. CTIS is used for counterfactual intervention and influence analysis.

---

## Exact Attribution

For a given triple, X-TuckER computes:

$$
C_{ijk}=W_{ijk}e_h[i]w_r[j]e_t[k]
$$

This produces a contribution tensor with the same interaction structure as the Tucker scoring function.

The framework validates the decomposition through:

* **Completeness**
* **Linearity**
* **Dummy-feature tests**
* **Tensor-ordering checks**
* **Model-score consistency**

The summed attribution tensor is compared directly with the corresponding raw trilinear score.

The notebook reports near-zero reconstruction errors, up to numerical precision, when comparing the reconstructed score with the model's corresponding raw score.

---

## Attribution Aggregation

The full contribution tensor can be aggregated to obtain higher-level explanations.

For example, relation-level attribution can be obtained by summing over the head and tail dimensions:

$$
A_j=\sum_{i,k}C_{ijk}
$$

Similarly, head and tail embedding dimensions can be analysed by aggregating across the corresponding tensor dimensions.

This allows X-TuckER to examine both:

* individual tensor interactions, and
* aggregated contributions associated with relation or embedding dimensions.

---

## Faithfulness & Counterfactual Evaluation

Exact attribution alone does not guarantee that an explanation is useful for understanding the complete trained model.

X-TuckER therefore performs interventions on the model's core tensor and evaluates how predictions change.

### Fidelity

Highly attributed tensor components are removed and the resulting change in model score is measured.

### Necessity

The analysis tests whether attributed components are necessary for maintaining the prediction.

### Sufficiency

A selected subset of important components is retained to determine whether it can preserve the prediction.

### Stability

The consistency of explanations is evaluated across repeated or perturbed evaluations.

These interventions provide an empirical evaluation of whether the components identified by the attribution are actually influential to the model's prediction.

---

## Counterfactual Tensor Influence Score

**Counterfactual Tensor Influence Score (CTIS)** is used to analyse the influence of tensor groups through counterfactual intervention.

The basic idea is to modify or remove selected groups of the Tucker core and measure the resulting change in the model's prediction.

Conceptually:

```text
Original Tucker Core
        │
        ▼
     TuckER
        │
        ▼
 Original Score
        │
        │ intervention on selected tensor groups
        ▼
Modified Tucker Core
        │
        ▼
     TuckER
        │
        ▼
Counterfactual Score
        │
        ▼
 Prediction Change
```

The intervention is evaluated using the model's actual scoring computation rather than treating attribution magnitude alone as evidence of influence.

CTIS therefore provides an intervention-based perspective that complements the exact mathematical decomposition.

---

## Integrated Gradients Connection

An important theoretical result explored in the project is the relationship between Exact Tensor Attribution and Integrated Gradients.

For the raw multilinear function with a zero tensor baseline:

$$
\begin{aligned}
IG_{ijk}
&=W_{ijk}e_h[i]w_r[j]e_t[k] \
&=C_{ijk}
\end{aligned}
$$

Thus, under this **specific scoring formulation and zero-baseline setting**, Integrated Gradients recovers the same tensor contribution analytically.

This makes the comparison useful as a **theoretical validation of the attribution**, rather than simply a comparison between two unrelated explainers.

The equivalence is specific to the stated function and baseline and should not be interpreted as a general equivalence between Integrated Gradients and Exact Tensor Attribution for arbitrary models or baselines.

---

## Experiments

The framework is evaluated across two experimental settings.

### FB15k-237

The primary X-TuckER experiments are performed on **FB15k-237**.

The experiments cover:

* TuckER link prediction
* Exact Tensor Attribution
* Alternative attribution methods
* Completeness and mathematical validation
* Faithfulness
* Stability
* Counterfactual analysis
* Baseline comparisons
* Failure-case analysis
* Attribution visualization

The complete experimental workflow and results are contained in the corresponding notebook.

### WN18RR

The framework is additionally evaluated on **WN18RR** to examine whether the attribution methodology remains applicable to a different Knowledge Graph dataset.

The WN18RR notebook extends the same core methodology with:

* Exact Tensor Attribution
* Alternative attribution methods
* Baseline analysis
* Faithfulness evaluation
* Stability analysis
* Counterfactual analysis
* XAI validation

---

## Baseline Analysis

The project evaluates Exact Tensor Attribution against alternative approaches under controlled interventions.

The baseline experiments include comparisons involving:

* **X-TuckER Exact Tensor Attribution**
* **Random feature selection**
* **Gradient × Input**

using the same intervention framework.

The purpose is to determine whether components identified through the model's exact mathematical decomposition have greater influence on predictions than components selected without using the model's attribution structure.

---

## Key Results

The experiments investigate three complementary properties of the proposed attribution framework.

### Mathematical Exactness

The contribution tensor reconstructs the corresponding TuckER raw trilinear score with near-zero numerical error, validating the completeness of the decomposition.

### Theoretical Consistency

Under the specified zero-baseline formulation, Exact Tensor Attribution has an analytical equivalence with Integrated Gradients.

### Empirical Faithfulness

Controlled tensor interventions evaluate whether highly attributed components produce greater changes in model predictions than components selected through baseline strategies.

Detailed numerical results, evaluation metrics, tables, and visualizations are contained directly within the experimental notebooks.

---

## Repository Structure

The repository intentionally keeps the experimental implementation and evaluation within the notebooks rather than duplicating the same code and results across separate `src/`, `evaluation/`, and `results/` directories.

```text
X-TuckER/
│
├── X_TuckER_Refactored.ipynb
├── X_TuckER_Refactored_WN18RR.ipynb
├── model.py
├── requirements.txt
└── README.md
```

### Notebooks

The notebooks contain the complete experimental workflow, including:

* Dataset preparation
* Model setup
* Checkpoint loading
* Attribution calculations
* Validation tests
* Baseline methods
* Faithfulness experiments
* Counterfactual analysis
* Evaluation
* Visualization
* Experimental results

The notebooks therefore serve as the primary experimental record of the project.

### `model.py`

Contains the TuckER model implementation used by the project.

The underlying TuckER architecture is **prior work** and is not claimed as an original contribution of X-TuckER.

The implementation is based on the publicly available TuckER repository:

**TuckER — Tensor Factorization for Knowledge Graph Completion**

https://github.com/ibalazevic/TuckER

X-TuckER builds its attribution, explainability, intervention, and evaluation framework around the TuckER scoring function.

### `requirements.txt`

Contains the Python dependencies required to run the model and notebooks.

---

## Reproducing the Experiments

### 1. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Obtain the TuckER implementation

Clone the original open-source TuckER repository:

```bash
git clone https://github.com/ibalazevic/TuckER.git
```

For Google Colab:

```python
!git clone -q https://github.com/ibalazevic/TuckER.git /content/TuckER
```

### 3. Run the notebooks

Open:

```text
X_TuckER_Refactored.ipynb
```

for the primary FB15k-237 experiment.

For the WN18RR extension, open:

```text
X_TuckER_Refactored_WN18RR.ipynb
```

The notebooks contain the model setup, checkpoint loading, attribution calculations, validation tests, evaluation procedures, and visualizations.

Dataset paths and model checkpoints may need to be configured depending on the execution environment.

---

## Reproducibility

The notebooks contain the code required to reproduce the project's experiments.

For reproducibility, the following experimental conditions should be kept consistent:

* Python environment
* Package versions
* Dataset version
* Model checkpoint
* Entity and relation embedding dimensions
* Attribution configuration
* Intervention parameters
* Evaluation settings

The Python dependencies used by the project are specified in `requirements.txt`.

---

## Limitations

### Model-specific formulation

The exact attribution is derived from TuckER's multilinear scoring structure and does not automatically generalize to arbitrary KGE architectures.

### Raw-score attribution

The completeness property applies directly to the raw trilinear score. It should not automatically be interpreted as an exact decomposition of every downstream quantity derived from that score.

### Explanation versus influence

An exact mathematical decomposition does not by itself establish behavioural importance. The project therefore evaluates faithfulness separately through controlled interventions.

### Computational cost

Tensor-level attribution and counterfactual analysis can become computationally expensive as embedding dimensions and tensor sizes increase.

### Baseline dependence

Gradient-based methods such as Integrated Gradients depend on their baseline and formulation. Comparisons should therefore be interpreted under the specific experimental configuration used.

---

## Research Direction

The broader objective is to investigate **structurally grounded explainability for Knowledge Graph Embedding models**.

Future work includes:

* Extending exact attribution to other KGE architectures
* Evaluating additional Knowledge Graph datasets
* Improving counterfactual explanation methods
* Developing more efficient tensor-level interventions
* Investigating human-interpretable aggregation of tensor interactions
* Conducting broader controlled comparisons between explanation methods

---

## Acknowledgements

X-TuckER builds upon the **TuckER** Knowledge Graph Embedding architecture introduced by Balažević, Allen, and Hospedales.

The underlying TuckER architecture and implementation are prior work. X-TuckER focuses on exact tensor attribution, explainability, intervention-based faithfulness analysis, and counterfactual evaluation built around the TuckER scoring function.

Original implementation:

https://github.com/ibalazevic/TuckER

Original paper:

> Balažević, I., Allen, C., & Hospedales, T. M. (2019). *TuckER: Tensor Factorization for Knowledge Graph Completion.*

---

## Citation

If you use X-TuckER, please cite this work:

```bibtex
@article{xtucker,
  title   = {X-TuckER: Exact Tensor Attribution for Knowledge Graph Embeddings},
  author  = {Aritra Ghosh Dastidar and Samrudhi Rao},
  year    = {2026}
}
```

Please also cite the original TuckER work:

```bibtex
@inproceedings{balazevic2019tucker,
  title     = {TuckER: Tensor Factorization for Knowledge Graph Completion},
  author    = {Balažević, Ivana and Allen, Carl and Hospedales, Timothy M.},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year      = {2019}
}
```
