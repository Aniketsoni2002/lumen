# Machine Learning Study Notes (Sample)

A sample knowledge base you can index to try AgentRAG immediately.

## Gradient Descent

Gradient descent is an optimization algorithm that minimizes a loss function by
iteratively moving in the direction of steepest descent. The **learning rate**
controls the step size. Our team's recommended default learning rate for the
image-classification project is **0.001**.

## Overfitting

Overfitting happens when a model learns the training data too well, including its
noise, and fails to generalize. Common remedies are dropout, L2 regularization,
early stopping, and gathering more data. Our internal policy is to hold out
**20%** of data as a validation set.

## Transformer Attention

The self-attention mechanism computes a weighted sum of value vectors, where the
weights come from the scaled dot-product of queries and keys. The scaling factor
is the square root of the key dimension. For our internal models the key
dimension is **64**.

## Team Compute Budget

Each experiment is allocated a budget of **8 GPU-hours**. Experiments exceeding
this must be approved by the team lead.
