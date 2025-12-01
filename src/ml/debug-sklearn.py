"""
Debug sklearn GP internals to understand the differences
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Test data
X_train = np.array([[0], [1], [2], [3], [4]])
y_train = np.array([0, 1, 0, 1, 0])

# Create GP with fixed kernel (no optimization)
kernel = RBF(length_scale=1.0)
kernel.length_scale_bounds = "fixed"  # Don't optimize

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, optimizer=None)
gp.fit(X_train, y_train)

print("=== sklearn internals ===")
print(f"y_train_mean_: {gp._y_train_mean}")
print(f"y_train (centered): {y_train - gp._y_train_mean}")
print()

# Compute K matrix
K = kernel(X_train)
print("K matrix:")
print(K)
print()

# K + alpha*I
K_noise = K + 0.1 * np.eye(5)
print("K + alpha*I:")
print(K_noise)
print()

# Cholesky
L = np.linalg.cholesky(K_noise)
print("L (Cholesky):")
print(L)
print()

# Alpha vector (sklearn stores this)
print(f"alpha_ (stored by sklearn): {gp.alpha_}")
print()

# Predictions
y_pred = gp.predict(X_train)
print(f"Predictions: {y_pred}")

# Manual calculation
print("\n=== Manual calculation ===")
y_centered = y_train - gp._y_train_mean
print(f"y centered: {y_centered}")

# Solve L @ L.T @ alpha = y_centered
alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
print(f"alpha vector: {alpha_vec}")

# K* @ alpha + mean
K_star = kernel(X_train, X_train)  # same as K for training points
pred_centered = K_star @ alpha_vec
pred = pred_centered + gp._y_train_mean
print(f"Manual prediction: {pred}")
