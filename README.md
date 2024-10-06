This repository contains code implementations for the following conformal prediction methods:
* [**Conformal Thresholded Intervals for Efficient Regression**](https://arxiv.org/abs/2407.14495)
* [**Entropy Reweighted Conformal Classification**](https://proceedings.mlr.press/v230/luo24a.html)

This project is built upon the Python toolbox [TorchCP](https://github.com/ml-stat-Sustech/TorchCP).

# Conformal Thresholded Intervals (CTI) for Efficient Regression

This paper introduces **Conformal Thresholded Intervals (CTI)**, a novel conformal regression method that aims to produce the smallest possible prediction set with guaranteed coverage.

## Key Points

1. Utilizes multi-output quantile regression to estimate conditional interquantile intervals.
2. Constructs prediction sets by thresholding these intervals based on their length.
3. Guarantees marginal coverage and can achieve optimal conditional coverage under certain conditions.
4. Computationally efficient and avoids estimating full conditional distributions.
5. Produces potentially non-convex prediction sets that adapt to local data density.

## Key Definitions

**Interquantile intervals:**
$I_k(x) = (q_{k-1}(x), q_k(x)] \quad \text{for } k = 1, \dots, K$

**Prediction set:**
$C(x) = \bigcup \{ I_k(x) \mid \mu(I_k(x)) \leq t, k = 1, \dots, K \}$

Where:
- $q_k(x)$ = estimated $k/K$ quantile
- $t$ = threshold determined on calibration set
- $\mu$ = Lebesgue measure

## Algorithm Overview

1. Train a quantile regression model on training data.
2. Estimate interquantile intervals on calibration and test data.
3. Determine threshold $t$ using the calibration set.
4. Construct prediction sets for test points by thresholding intervals.

## Theoretical Guarantees

- **Marginal coverage:** $P(Y \in C(X)) \geq 1 - \alpha$
- Can achieve optimal conditional coverage and smallest expected prediction set size under certain conditions.

# Entropy Reweighted Conformal Classification

This paper proposes **Entropy Reweighted Conformal Classification**, a novel approach to improve the efficiency of prediction sets in conformal classification.

## Key Points

1. Leverages the uncertainty of the classifier by using entropy-based reweighting.
2. Applies a reweighting to the logits of the classification model based on the entropy of the predicted probability distribution.
3. The reweighted logits are defined as:
   $\tilde{z}_k(X) = \frac{z_k(X)}{H(X) \cdot T}$
   where:
   - $z_k(X)$ = original logit
   - $H(X)$ = entropy of the predicted probability distribution
   - $T$ = temperature parameter
4. The reweighted probabilities are obtained by applying the softmax function to the reweighted logits:
   $\tilde{f}_k(X) = \frac{\exp(\tilde{z}_k(X))}{\sum_j \exp(\tilde{z}_j(X))}$
5. These reweighted probabilities are used to compute the conformity scores for conformal prediction.
6. The temperature parameter $T$ is optimized using cross-validation on a separate validation set.
7. Aims to improve the efficiency of prediction sets while maintaining the coverage guarantees of conformal prediction.
8. Experimental results on various datasets (AG News, CAREER, MNIST, Fashion MNIST) demonstrate improved performance in terms of prediction efficiency and accuracy compared to existing techniques.

## Key Definitions

**Prediction Set:**
```math
C_A = \{ y_{N+1} \in \mathcal{Y} |\sum_{n \in \mathcal{I}_2} \mathbb{1}(A_n \leq A_{N+1}) \leq n\alpha \} = \{ y_{N+1} \in \mathcal{Y} | A_{N+1} \leq Q_A \}
```
Where:
- $A_n = a(f(X_n), Y_n)$, conformity score
- $A_{N+1} = a(f(X_{N+1}), y_{N+1})$
- $Q_A$ is the $(1-\alpha)$-th sample quantile of $A_1, \dots, A_N$
- $n\alpha = \lceil (1-\alpha)(|\mathcal{I}_2| + 1) \rceil$

**Adaptive Prediction Sets (APS) Score:**
$A_n = a(f(X_n), Y_n) = \sum_{i=1}^{r(Y_n, f(X_n)) - 1} f_{(i)}(X_n) + U f_{(r(Y_n, f(X_n)))}(X_n)$

Where:
- $f_{(i)}(X_n)$ denotes the $i$-th largest element of the probability vector $f(X_n)$.
- $r(Y_n, f(X_n))$ is the rank of the true label $Y_n$ in the probability vector $f(X_n)$.
- $U \sim \text{Uniform}(0,1)$ is independent of everything else.

## Algorithm Overview

1. Train a classification model $f$ on training data.
2. Compute entropy $H(X)$ for each data point.
3. Apply entropy-based reweighting to logits.
4. Compute reweighted probabilities.
5. Calculate conformity scores using reweighted probabilities.
6. Construct prediction sets using the conformal prediction framework.
7. Optimize the temperature parameter $T$ using cross-validation.

## Theoretical Guarantees

- **Marginal coverage:** $\mathbb{P}(Y_{N+1} \in C(X_{N+1})) \geq 1 - \alpha$
- Maintains the validity of conformal prediction while improving efficiency.

This approach integrates accurate uncertainty quantification with the coverage guarantees of conformal prediction, addressing limitations observed in previous attempts to combine confidence calibration with conformal prediction.

# Reference
```
@inproceedings{luo2024entropy,
  title={Entropy Reweighted Conformal Classification},
  author={Luo, Rui and Colombo, Nicolo},
  booktitle={The 13th Symposium on Conformal and Probabilistic Prediction with Applications},
  pages={264--276},
  year={2024},
  organization={PMLR}
}

@article{luo2024conformal,
  title={Conformal Thresholded Intervals for Efficient Regression},
  author={Luo, Rui and Zhou, Zhixin},
  journal={arXiv preprint arXiv:2407.14495},
  year={2024}
}
```
