import numpy as np
import matplotlib.pyplot as plt

u0 = 1.0                  # initial condition u(0)=1
x_min, x_max = 0.0, 1.0   # interval
n_collocation = 30        # number of collocation points for training

# uniform collocation points
x_coll = np.linspace(x_min, x_max, n_collocation)

# right-hand side of ODE: u' = f(x, u). Here: f = -u
def f_rhs(x, u):
    return -u

# -------------------- Model Parameters --------------------
# Parameters theta = [w1 (first weight), b1 (first bias), w2, b2]
# Initialization with small random numbers
theta = np.array([ -1.0, 0.0,  1.0, 0.0], dtype=float)  # starting values
# The initialization can be changed.
# Based on my small tests, the larger the weights, the closer the error decay plot is to a decaying one, like e^(-x).
# It's important to note that no matter how absurdly I change the weights, the result still turns out suspiciously accurate.


def unpack(theta):
    # Unpack into individual parameters
    w1, b1, w2, b2 = theta
    return w1, b1, w2, b2

def forward_N(x, theta):
    # Compute N(x)
    w1, b1, w2, b2 = unpack(theta) # Get parameters
    z = w1 * x + b1 # Output of first neuron
    t = np.tanh(z) # Apply activation function
    return w2 * t + b2, t  # Output of second neuron

def dN_dx(x, theta, t=None):
    # Compute network derivative dN/dx
    # Network from previous function: N(x) = w2*[tanh(w1*x + b1)]+b2
    w1, b1, w2, b2 = unpack(theta) # Get parameters again
    if t is None:
        _, t = forward_N(x, theta) # Apparently, this function in Python means to accept t
                                   # or, if not accepted, calculate it based on the previous function.
    s = 1.0 - t**2  # Known derivative of tanh
    return w2 * s * w1 # dN/dx = w2*(1−t^2)*w1 by chain rule

def u_trial(x, theta):
    # Trial solution u = u0 + x * N(x)
    Nval, _ = forward_N(x, theta)
    return u0 + x * Nval # Exact formulation of the trial solution for first-order ODE from the article.

def du_trial_dx(x, theta):
    # Derivative of Trial solution du/dx = N + x * dN/dx"""
    Nval, t = forward_N(x, theta)
    return Nval + x * dN_dx(x, theta, t)

def residuals(x, theta):
    # Compute residual
    u = u_trial(x, theta)
    du = du_trial_dx(x, theta)
    r = du - f_rhs(x, u)
    return r

def loss(theta):
    # Compute residual at collocation points
    r = residuals(x_coll, theta)
    # Return mean squared residual (mean squared error)
    return np.mean(r**2)


def grads(theta, x_points):
    # Find the gradient of the error with respect to parameters theta (w1, b1, w2, b2)

    w1, b1, w2, b2 = unpack(theta)
    Np = x_points.size
    grad = np.zeros_like(theta) # array of zeros
    x = x_points
    z = w1 * x + b1
    t = np.tanh(z)
    s = 1.0 - t**2
    N = w2 * t + b2
    dNdx = w2 * s * w1

    u = u0 + x * N
    du_dx = N + x * dNdx
    r = du_dx + u  # since f = -u
    # Everything above is already obtained data/functions/parameters.

    # Manually calculated partial derivatives of N, s, dN, r.
    dN_dw1 = w2 * s * x
    dN_db1 = w2 * s
    dN_dw2 = t
    dN_db2 = np.ones_like(x)

    ds_dz = -2.0 * t * s
    ds_dw1 = ds_dz * x
    ds_db1 = ds_dz * 1.0

    ddNdx_dw2 = w1 * s
    ddNdx_db2 = np.zeros_like(x)
    ddNdx_dw1 = w2 * (s + w1 * ds_dw1)
    ddNdx_db1 = w2 * w1 * ds_db1

    dr_dw1 = dN_dw1 * (1.0 + x) + x * ddNdx_dw1
    dr_db1 = dN_db1 * (1.0 + x) + x * ddNdx_db1
    dr_dw2 = dN_dw2 * (1.0 + x) + x * ddNdx_dw2
    dr_db2 = dN_db2 * (1.0 + x) + x * ddNdx_db2

    coef = 2.0 / float(Np)
    grad[0] = coef * np.sum(r * dr_dw1)
    grad[1] = coef * np.sum(r * dr_db1)
    grad[2] = coef * np.sum(r * dr_dw2)
    grad[3] = coef * np.sum(r * dr_db2)

    return grad

# Gradient descent. lr, n_epochs can be tuned.
lr = 0.1
n_epochs = 2000

loss_history = []

for epoch in range(n_epochs):
    L = loss(theta)
    loss_history.append(L)
    g = grads(theta, x_coll)
    theta = theta - lr * g
    if epoch % 200 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:4d}  Loss = {L:.4e}  theta = {theta}")

# -------------------- Results and Visualization --------------------
x_test = np.linspace(x_min, x_max, 201)
u_pred = u_trial(x_test, theta)
u_exact = np.exp(-x_test)

print("\nFinal loss (MSE on collocation):", loss(theta))
print("Maximum absolute error between u_pred and u_exact:",
      np.max(np.abs(u_pred - u_exact)))

# Plots
plt.figure(figsize=(8,4))
plt.plot(x_test, u_exact, label="exact u = exp(-x)", linewidth=2)
plt.plot(x_test, u_pred, '--', label="u_trial (NN)", linewidth=2)
plt.scatter(x_coll, u_trial(x_coll, theta), c='red', s=30, label="collocation points (train)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Comparison of exact solution and approximation (trial)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("Loss (MSE)")
plt.title("Training history (Loss)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output parameters and a few values
print("\nFinal parameters theta = [w1, b1, w2, b2] =", theta)
for xx in [0.0, 0.25, 0.5, 0.75, 1.0]:
    Nval, tval = forward_N(np.array([xx]), theta)
    print(f"x={xx:.2f}: u_trial={u_trial(xx,theta):.6f}, N={Nval[0]:.6f}")