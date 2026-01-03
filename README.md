# Lagaris-type PINN for First-Order ODE

This repository contains a minimal Python implementation of a **Physics-Informed Neural Network (PINN)** for solving a first-order ordinary differential equation (ODE) using the method proposed by Lagaris, Likas, and Fotiadis in their 1998 paper:

> **"Artificial Neural Networks for Solving Ordinary and Partial Differential Equations"**  
> _IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998._

The method constructs a **trial solution** that automatically satisfies initial/boundary conditions and uses a small feedforward neural network to approximate the remaining part of the solution. The network is trained by minimizing the residual of the differential equation over a set of collocation points.

---

## Problem Solved

We solve the simple first-order ODE:

\[
\frac{du}{dx} = -u, \quad u(0) = 1, \quad x \in [0, 1]
\]

Exact solution:  
\[
u(x) = e^{-x}
\]

---

## Method Overview

The trial solution is constructed as:

\[
u_t(x) = u_0 + x \cdot N(x; \theta)
\]

where:

- \( u_0 = 1 \) is the initial condition,
- \( N(x; \theta) \) is a small neural network with parameters \( \theta \),
- The term \( x \) ensures that the initial condition is satisfied exactly for any \( N \).

The network \( N \) has the architecture:

\[
N(x) = w_2 \cdot \tanh(w_1 \cdot x + b_1) + b_2
\]

The loss function is the mean squared residual of the ODE over collocation points:

\[
L(\theta) = \frac{1}{m} \sum\_{i=1}^m \left[ \frac{du_t}{dx}(x_i) + u_t(x_i) \right]^2
\]

Gradients are computed analytically via backpropagation, and parameters are updated using gradient descent.

---

## Code Structure

The script is self-contained and requires only `numpy` and `matplotlib`.

Key functions:

- `forward_N(x, theta)` – computes network output \( N(x) \)
- `u_trial(x, theta)` – trial solution \( u_t(x) \)
- `du_trial_dx(x, theta)` – derivative of trial solution
- `residuals(x, theta)` – computes PDE residual
- `loss(theta)` – mean squared residual over collocation points
- `grads(theta, x_points)` – analytically computed gradient of loss

---

## Results

After training, the neural network approximates the solution \( u(x) = e^{-x} \) with high accuracy using only **4 parameters** and **30 collocation points**.

### Figure 1: Solution Comparison

![Solution Comparison](Figure_1.png)

### Figure 2: Training Loss

![Training Loss](Figure_2.png)

---

## Why This Is a "Lagaris-type PINN"

This implementation follows the core idea from the Lagaris et al. paper:

1. **Trial solution construction** that exactly satisfies initial/boundary conditions.
2. Use of a **simple feedforward neural network** as the adjustable part.
3. **Collocation method** – training on a set of points inside the domain.
4. **Minimization of the PDE residual** via gradient-based optimization.

This is essentially an early form of what is now called a **Physics-Informed Neural Network (PINN)**, where the physics (the ODE) is embedded into the loss function.

---

## References

- Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). _Artificial Neural Networks for Solving Ordinary and Partial Differential Equations_. IEEE Transactions on Neural Networks, 9(5), 987–1000.
