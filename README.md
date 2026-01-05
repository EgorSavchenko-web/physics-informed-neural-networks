# Lagaris-type PINN for First-Order ODE

This repository contains a minimal Python implementation of a **Physics-Informed Neural Network (PINN)** for solving a first-order ordinary differential equation (ODE) using the method proposed by Lagaris, Likas, and Fotiadis in their 1998 paper:

> **"Artificial Neural Networks for Solving Ordinary and Partial Differential Equations"**  
> _IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998._

The method constructs a **trial solution** that automatically satisfies the initial condition and uses a small feedforward neural network to approximate the remaining part of the solution. The network is trained by minimizing the residual of the differential equation over a set of collocation points using **gradient descent**.

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
- The term \( x \) ensures that the initial condition \( u_t(0) = u_0 \) is satisfied exactly for any \( N \).

The network \( N \) has a minimal architecture with one hidden neuron and a \(\tanh\) activation:
\[
N(x) = w_2 \cdot \tanh(w_1 \cdot x + b_1) + b_2
\]
Thus, the trainable parameters are \( \theta = [w_1, b_1, w_2, b_2] \). In this implementation, they are initialized to \( [-1.0, 0.0, 1.0, 0.0] \).

The physics-informed loss function is the mean squared residual of the ODE over \( m \) collocation points:
\[
L(\theta) = \frac{1}{m} \sum_{i=1}^m \left[ \frac{du_t}{dx}(x_i) + u_t(x_i) \right]^2
\]
The residual \( R(x) = \frac{du_t}{dx} + u_t \) enforces the governing equation \( u' = -u \).

**Key difference from modern PINNs:** Instead of relying on automatic differentiation (backpropagation) through the residual, this implementation follows the original paper and uses **explicit, hand-coded analytic gradients** of the loss with respect to parameters \( \theta \). Parameters are updated via basic gradient descent: \( \theta \leftarrow \theta - \eta \, \nabla_\theta L \).

---

## Code Structure

The script is self-contained and requires only `numpy` and `matplotlib`.

Key functions:
- `forward_N(x, theta)` – computes network output \( N(x) \)
- `u_trial(x, theta)` – trial solution \( u_t(x) \)
- `du_trial_dx(x, theta)` – derivative of the trial solution
- `residuals(x, theta)` – computes PDE residual \( R(x) \)
- `loss(theta)` – mean squared residual over all collocation points
- `grads(theta, x_points)` – **analytically computed** gradient of the loss (not automatic differentiation)

---

## Results

After training (2000 epochs of gradient descent with learning rate 0.1), the neural network approximates the solution \( u(x) = e^{-x} \) with high accuracy using only **4 parameters** and **30 uniformly spaced collocation points**.

The final mean squared residual is typically below \( 10^{-5} \), and the maximum absolute error against the exact solution is of order \( 10^{-4} \).

### Output Figures
1.  **Solution Comparison:** Shows the exact solution \( e^{-x} \) and the neural network trial solution \( u_t(x) \).
2.  **Training History:** Plots the decay of the loss function (MSE residual) in log-scale.


---

## References

- Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). _Artificial Neural Networks for Solving Ordinary and Partial Differential Equations_. IEEE Transactions on Neural Networks, 9(5), 987–1000.
