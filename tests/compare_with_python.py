#!/usr/bin/env python3
"""
Compare @tangent.to/ds outputs with Python implementations
to verify numerical correctness after safeguards implementation.
"""

import numpy as np
import json
import sys
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compare_pca():
    """Compare PCA with sklearn"""
    print("=" * 60)
    print("Testing PCA against sklearn")
    print("=" * 60)

    # Simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 4)

    # Fit sklearn PCA
    pca = SklearnPCA(n_components=2)
    pca.fit(X)

    results = {
        "test": "PCA",
        "data": X.tolist(),
        "n_components": 2,
        "sklearn": {
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "mean": pca.mean_.tolist()
        }
    }

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    return results

def compare_kmeans():
    """Compare KMeans with sklearn"""
    print("\n" + "=" * 60)
    print("Testing KMeans against sklearn")
    print("=" * 60)

    # Simple dataset
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 2) + [0, 0],
        np.random.randn(30, 2) + [5, 5],
        np.random.randn(30, 2) + [0, 5]
    ])

    # Fit sklearn KMeans
    kmeans = SklearnKMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    results = {
        "test": "KMeans",
        "data": X.tolist(),
        "k": 3,
        "sklearn": {
            "centers": kmeans.cluster_centers_.tolist(),
            "labels": kmeans.labels_.tolist(),
            "inertia": float(kmeans.inertia_)
        }
    }

    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
    print(f"Inertia: {kmeans.inertia_:.4f}")
    print(f"Unique labels: {np.unique(kmeans.labels_)}")

    return results

def compare_logistic_regression():
    """Compare logistic regression with sklearn"""
    print("\n" + "=" * 60)
    print("Testing Logistic Regression against sklearn")
    print("=" * 60)

    # Simple binary classification dataset
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)
    # Create linear decision boundary
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.1 > 0).astype(int)

    # Fit sklearn logistic regression
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(X, y)

    predictions = lr.predict(X)
    accuracy = np.mean(predictions == y)

    results = {
        "test": "LogisticRegression",
        "X": X.tolist(),
        "y": y.tolist(),
        "sklearn": {
            "coefficients": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
            "accuracy": float(accuracy),
            "predictions": predictions.tolist()
        }
    }

    print(f"Coefficients: {lr.coef_[0]}")
    print(f"Intercept: {lr.intercept_[0]:.4f}")
    print(f"Training accuracy: {accuracy:.4f}")

    return results

def compare_linear_regression():
    """Compare linear regression with scipy"""
    print("\n" + "=" * 60)
    print("Testing Linear Regression against scipy")
    print("=" * 60)

    # Simple linear regression dataset
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 2)
    true_coef = np.array([2.5, -1.3])
    true_intercept = 0.7
    noise = np.random.randn(n) * 0.5
    y = X @ true_coef + true_intercept + noise

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(n), X])

    # Fit with scipy
    from scipy.linalg import lstsq
    coef, residuals, rank, s = lstsq(X_with_intercept, y)

    predictions = X_with_intercept @ coef
    r_squared = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))

    results = {
        "test": "LinearRegression",
        "X": X.tolist(),
        "y": y.tolist(),
        "scipy": {
            "intercept": float(coef[0]),
            "coefficients": coef[1:].tolist(),
            "r_squared": float(r_squared),
            "predictions": predictions.tolist()
        }
    }

    print(f"Coefficients: {coef[1:]}")
    print(f"Intercept: {coef[0]:.4f}")
    print(f"R²: {r_squared:.4f}")

    return results

def main():
    print("Starting comparison tests with Python implementations")
    print("=" * 60)

    all_results = {
        "pca": compare_pca(),
        "kmeans": compare_kmeans(),
        "logistic": compare_logistic_regression(),
        "linear": compare_linear_regression()
    }

    # Save results to JSON for JS to read
    with open('/tmp/python_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ All Python reference results saved to /tmp/python_comparison_results.json")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
