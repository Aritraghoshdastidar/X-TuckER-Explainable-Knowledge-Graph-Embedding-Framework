# X-TuckER

### Exact Tensor Attribution for Explainable Knowledge Graph Embeddings

X-TuckER is a model-aligned explainability framework for **TuckER-based Knowledge Graph Embedding (KGE) models**. It exploits the algebraic structure of the Tucker decomposition to derive an **exact, closed-form attribution** of TuckER's trilinear scoring function, exposing the individual head–relation–tail–core interactions that drive a prediction — and then validates those attributions empirically by intervening on the trained model itself.

For a triple $(h, r, t)$, the attribution tensor is:

$$
C_{ijk} = \hat{\mathcal{W}}_{ijk}\, e_h[i]\, w_r[j]\, e_t[k]
$$

where $\hat{\mathcal{W}}$ is the Tucker core tensor reshaped to align with the attribution's index ordering, $e_h, e_t$ are entity embeddings, and $w_r$ is the relation embedding. Each $C_{ijk}$ is the exact contribution of one (head-dim, relation-dim, tail-dim) interaction to the raw trilinear score:

$$
\sum_{i,j,k} C_{ijk} = s_{\text{raw}}(h, r, t)
$$

---

## Why X-TuckER?

KGE models can achieve strong link-prediction performance while remaining difficult to interpret. Most explainability techniques treat the trained model as a black box and estimate feature importance through gradients, perturbations, or surrogate models. X-TuckER instead exploits the **internal mathematical structure of TuckER itself**, producing two complementary layers of analysis:

* **Exact attribution** — what interactions mathematically constitute the score, derived directly from the model's own computation graph, with zero approximation error.
* **Faithfulness** — whether those attributed interactions actually drive the trained model's prediction when intervened upon, evaluated through the real forward pass rather than assumed from attribution magnitude alone.

Separating these two questions — mathematical correctness vs. behavioral importance — is the core design principle of the framework.

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
         Embeddings           𝒲
              │                 │
              └────────┬────────┘
                       ▼
              Exact Attribution
                 C[i,j,k]  (raw-score space)
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
      Attribution   Faithfulness    Counterfactual
       Analysis    (model.forward   Analysis (CTIS,
     (raw score)     space)         relation swap)
```

Attribution can be examined at the individual tensor-entry level, or aggregated along any dimension — for example, relation-level attribution:

$$
A_j = \sum_{i,k} C_{ijk}
$$

Head- and tail-dimension attributions are obtained analogously by summing over the complementary axes.

---

## Explanation Methods

| Method | Role |
|---|---|
| **Exact Tensor Attribution** | Closed-form decomposition of the TuckER trilinear score; zero approximation error |
| **Integrated Gradients (rigorous)** | Path-integral attribution from a zero-tensor baseline, with an explicit convergence check as the number of integration steps increases |
| **Gradient × Input** | Standard gradient-based saliency baseline |
| **Random** | Model-independent baseline used to test whether attribution-driven ablation outperforms chance |
| **CTIS (Counterfactual Tensor Influence Score)** | Group-wise counterfactual intervention on the core tensor, scored via `model.forward()` |
| **Relation-Swap Counterfactual** | Optimizes a perturbation on the relation embedding to find the minimal causal change that flips or destroys the model's prediction |

Exact Tensor Attribution is the framework's central contribution; the remaining methods serve as theoretical or empirical points of comparison.

---

## Attribution Validity: Sanity and Consistency Checks

Following standard XAI validation practice (Adebayo et al., 2018), the notebooks verify:

* **Completeness** — $\sum C_{ijk}$ reconstructs $s_{\text{raw}}$ to within floating-point error.
* **Linearity** — scaling the core tensor by $\alpha$ scales $\sum C_{ijk}$ by $\alpha$.
* **Dummy-feature test** — zeroing a core-tensor entry zeros the corresponding attribution entry.
* **Parameter randomization** — attribution computed on a randomized core tensor is uncorrelated with the real attribution.
* **Input randomization** — shuffling entity/relation embeddings destroys the attribution pattern.
* **Integrated Gradients equivalence** — for the raw trilinear function with a zero-tensor baseline, IG is shown analytically (and verified numerically, with convergence tracked over integration steps) to equal $C_{ijk}$ exactly.

$$
\text{IG}_{ijk} = \hat{\mathcal{W}}_{ijk}\, e_h[i]\, w_r[j]\, e_t[k] = C_{ijk}
$$

This equivalence holds specifically for this trilinear scoring function under a zero baseline, and should not be read as a general equivalence between Integrated Gradients and exact attribution for arbitrary models.

---
## Faithfulness & Counterfactual Evaluation

Exact attribution does not by itself establish behavioral importance, since it is computed on the raw score rather than the deployed model. X-TuckER closes this gap by intervening on the trained model directly, using `model.forward()` (BatchNorm, dropout, and all) for every evaluation metric:

* **Fidelity** — ablates the top-k highest-attributed core-tensor entries and measures the resulting drop in model score, compared against ablating the same number of random entries. K is scaled proportionally to tensor size (roughly 0.01%–5% of entries) so the signal is not swamped by noise.
* **Sufficiency@k** — retains only the top-k attributed entries and checks whether the model's score is preserved.
* **Necessity@k** — removes the top-k attributed entries and checks whether the model's score collapses.
* **Stability** — measures how consistent the attribution ranking remains under small input perturbations and across repeated evaluations.

---

## Counterfactual Tensor Influence Score (CTIS)

CTIS quantifies the causal influence of a *group* of core-tensor entries by intervening on them and re-running the full model:

$$
\text{CTIS}(G; h, r, t) = \frac{\Delta_{\min}(G)}{|\text{score}(h, r, t)|}
$$

```text
Original Core Tensor 𝒲 ──► TuckER.forward() ──► Original Score
        │
        │ intervene on group G
        ▼
Modified Core Tensor 𝒲' ──► TuckER.forward() ──► Counterfactual Score
        │
        ▼
   Prediction Change
```

Because the reference score and the ablated score are both computed through the real model forward pass, CTIS reflects actual causal influence rather than attribution magnitude alone. The framework additionally implements a **relation-swap counterfactual**, which searches over alternative relation embeddings to find the smallest perturbation that flips the model's prediction, and reports counterfactual validity across a batch of triples.

---

## Repository Structure

```text
X-TuckER/
│
├── X_TuckER_Refactored.ipynb          # Primary experiments — FB15k-237
├── X_TuckER_Refactored_WN18RR.ipynb   # Extension experiments — WN18RR
├── model.py                           # TuckER architecture (prior work) + our trained checkpoint
├── requirements.txt                   # Python dependencies
└── README.md
```

The experimental implementation, validation, and results are kept within the notebooks themselves.

### Notebooks

Each notebook walks through the full pipeline in order:

1. Environment setup and dependency installation
2. Dataset loading (FB15k-237 or WN18RR)
3. Entity/relation index loading and TuckER checkpoint restoration
4. Derivation of the forward pass, the exact attribution tensor, and the completeness proof
5. Sanity/randomization tests and the Integrated Gradients equivalence proof
6. Core attribution, gradient-based, and Integrated Gradients implementations
7. Counterfactual optimization and CTIS
8. Fidelity, sufficiency, and necessity evaluation (in `model.forward()` space)
9. Path-based explanations, multi-seed stability, and a human-interpretability proxy
10. Failure-case and explainability classification
11. Batch evaluation across multiple triples, baseline comparisons, and publication-quality visualizations
12. Final metrics summary and a consolidated method comparison table

### `model.py`

Contains the trained TuckER model used by the project. The **architecture** follows the publicly available reference TuckER implementation, but the **weights, checkpoint, and core tensor were trained by us** on the target dataset (FB15k-237 / WN18RR) and are the actual model the attribution and faithfulness experiments run against.

Checkpoints were saved as:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'core_tensor': model.W.detach()
}, 'checkpoint.pth')
```

along with the corresponding entity/relation index mappings (`entity_idxs.pkl`, `relation_idxs.pkl`) and an extracted core tensor (`core_tensor.pt`) used to independently verify the checkpoint on load.

Reference architecture: [github.com/ibalazevic/TuckER](https://github.com/ibalazevic/TuckER)


---

## Reproducing the Experiments

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain the TuckER implementation

```bash
git clone https://github.com/ibalazevic/TuckER.git
```

For Google Colab, the notebooks handle this automatically:

```python
!git clone -q https://github.com/ibalazevic/TuckER.git /content/TuckER
```

### 3. Train or provide a TuckER checkpoint

The notebooks expect a trained TuckER checkpoint (entity/relation embeddings and core tensor) — they do not train a model from scratch inline. You can train your own using the training script in the original TuckER repository, matching the embedding dimensions used in this project (d_e = d_r = 200). Configure the checkpoint and core-tensor paths at the top of the notebook to point at your trained model before running.

### 4. Run the notebooks

```text
X_TuckER_Refactored.ipynb          # FB15k-237 (primary experiment)
X_TuckER_Refactored_WN18RR.ipynb   # WN18RR (extension)
```

Each notebook is self-contained: dataset download, checkpoint loading, attribution computation, validation, faithfulness evaluation, and visualization all run in sequence from top to bottom.

---

## Experiments

### FB15k-237 (primary)

The primary experiments cover: TuckER link prediction, exact tensor attribution, alternative attribution methods, completeness and axiom validation, fidelity/sufficiency/necessity, stability, counterfactual and CTIS analysis, baseline comparisons (tensor attribution vs. gradient × input vs. random), failure-case analysis, and publication-quality visualization across a batch of evaluated triples.

### WN18RR (extension)

The same methodology — exact attribution, alternative attribution baselines, faithfulness evaluation, stability, and counterfactual analysis — is re-run on WN18RR to test whether the attribution framework generalizes beyond FB15k-237's relational structure.

---

## Key Results

* **Mathematical exactness.** The attribution tensor reconstructs the raw trilinear score with error bounded only by floating-point precision, and passes all completeness, linearity, dummy-feature, and randomization sanity checks.
* **Theoretical consistency.** Under a zero-tensor baseline, Integrated Gradients is shown — analytically and numerically, with explicit convergence tracking — to equal Exact Tensor Attribution.
* **Empirical faithfulness.** Ablating the top-k attributed core-tensor entries reduces the model's actual `forward()` score by more than ablating random entries of the same size, across the evaluated batch of triples.
* **Structural honesty.** A large share of triples are classified as *diffuse* rather than *explainable*, reflecting TuckER's use of BatchNorm and dropout to distribute representation across its full embedding space. The framework reports this explicitly rather than masking it.

Full numerical results, tables, and visualizations are contained in the notebooks themselves.

---

## Limitations

* **Model-specific formulation.** The exact attribution is derived from TuckER's trilinear scoring structure and does not automatically generalize to other KGE architectures.
* **Raw-score vs. deployed-model gap.** $C_{ijk}$ is exact for the raw trilinear score; it is not, by itself, an exact decomposition of `model.forward()`'s output, which additionally passes through BatchNorm, dropout, and a sigmoid. The framework addresses this by evaluating all faithfulness metrics in `model.forward()` space rather than assuming raw-score attribution transfers unchanged.
* **Diffuse attribution is expected, not a defect.** BatchNorm-regularized TuckER models distribute score across many core-tensor entries; low sufficiency/fidelity at very small k for such triples reflects model structure, not attribution failure.
* **Baseline dependence.** The Integrated Gradients equivalence holds under a specific zero-tensor baseline; results should be interpreted under that stated configuration.
* **Computational cost.** Tensor-level attribution and counterfactual/CTIS analysis scale with the cube of the embedding dimension and can become expensive for large $d_e$.

---

## Research Direction

* Extending exact attribution to other multilinear or bilinear KGE architectures.
* Evaluating additional knowledge graph datasets beyond FB15k-237 and WN18RR.
* More efficient tensor-level counterfactual interventions at scale.
* Human-interpretable aggregation of high-dimensional tensor interactions.
* Broader controlled comparisons against additional post-hoc explainers (e.g., SHAP).

---

## Acknowledgements

X-TuckER builds upon the TuckER knowledge graph embedding architecture introduced by Balažević, Allen, and Hospedales. The architecture and reference implementation are prior work; we trained this implementation ourselves to produce the model checkpoints used throughout this repository. X-TuckER's contribution is the exact tensor attribution, faithfulness evaluation, and counterfactual framework built around TuckER's scoring function.

Original implementation: [github.com/ibalazevic/TuckER](https://github.com/ibalazevic/TuckER)

> Balažević, I., Allen, C., & Hospedales, T. M. (2019). *TuckER: Tensor Factorization for Knowledge Graph Completion.* EMNLP-IJCNLP 2019.

---

## Citation

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
