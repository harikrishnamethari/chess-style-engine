# 10 — Model Training, Loss Functions & Evaluation

## 1. Multi-Task Loss Formulation (`src/training/trainer.py`)
Training uses a joint multi-task objective function combining move classification with blunder prediction:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \lambda_{\text{blunder}} \cdot \mathcal{L}_{\text{blunder}}$$

1. **Policy Loss ($\mathcal{L}_{\text{policy}}$)**:
   Standard Cross-Entropy Loss over the 4,672 move classes:
   $$\mathcal{L}_{\text{policy}} = -\sum_{i=1}^{C} y_i \log(\hat{p}_i)$$
   where $y$ is the one-hot encoded human move, and $\hat{p} = \text{Softmax}(\mathbf{z}_{\text{policy}})$.

2. **Blunder Loss ($\mathcal{L}_{\text{blunder}}$)**:
   Binary Cross-Entropy Loss predicting if the move led to an immediate collapse in centipawn evaluation:
   $$\mathcal{L}_{\text{blunder}} = -\left[ y_b \log(\hat{y}_b) + (1 - y_b) \log(1 - \hat{y}_b) \right]$$
   *(Default weight: $\lambda_{\text{blunder}} = 0.25$)*.

---

## 2. Optimizer & Training Configuration
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-4`).
- **Learning Rate Schedule**: Cosine Annealing scheduler decaying learning rate down to `1e-5` across epochs.
- **Batch Size**: 256 or 512 (ideal for GPU training in Google Colab).
- **Gradient Clipping**: Clipped at `max_norm=1.0` to prevent exploding gradients in the recurrent GRU layers.

---

## 3. Evaluation Metrics (`src/evaluation/metrics.py`)
Traditional accuracy is insufficient for chess engines because multiple moves are often reasonable. The evaluation suite computes:

1. **Top-1 Move Accuracy**:
   $$\text{Acc}_1 = \frac{1}{N} \sum_{k=1}^N \mathbb{I}(\operatorname{argmax}(\mathbf{z}) = y)$$
   Percentage of positions where the model's highest-probability move exactly matches the player's choice.
2. **Top-3 / Top-5 Move Accuracy**:
   Percentage of positions where the human move is contained within the model's top-$k$ predictions.
3. **Legal Move Compliance**:
   Percentage of model predictions that correspond to strictly legal moves before masking (measures whether the model learned the fundamental geometry of chess).
4. **Style Consistency (Kullback-Leibler Divergence)**:
   Measures the distance between move probability distributions when conditioned on Style A vs Style B:
   $$D_{\text{KL}}(P_A \parallel P_B) = \sum_{m} P_A(m) \log \left( \frac{P_A(m)}{P_B(m)} \right)$$
   A non-zero KL divergence proves that the style embedding actively changes the predicted move distribution on identical board states.

