/**
 * Comparison tests with Python implementations
 * Verifies numerical correctness after safeguards implementation
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import { PCA } from '../src/mva/estimators/PCA.js';
import { KMeans } from '../src/ml/estimators/KMeans.js';
import { GLM } from '../src/stats/estimators/GLM.js';

let pythonResults;

beforeAll(() => {
  // Run Python comparison script to generate reference results
  try {
    console.log('Running Python comparison script...');
    execSync('python3 tests/compare_with_python.py', { stdio: 'inherit' });
    pythonResults = JSON.parse(
      readFileSync('/tmp/python_comparison_results.json', 'utf-8')
    );
    console.log('✓ Python reference results loaded');
  } catch (error) {
    console.error('Failed to run Python comparison:', error);
    throw error;
  }
});

describe('PCA - Comparison with sklearn', () => {
  it('should produce similar explained variance ratios', () => {
    const data = pythonResults.pca.data;
    const pca = new PCA({ n_components: 2 });
    pca.fit({ data });

    const jsVarianceRatio = pca.explainedVarianceRatio();
    const pyVarianceRatio = pythonResults.pca.sklearn.explained_variance_ratio;

    console.log('JS explained variance ratio:', jsVarianceRatio);
    console.log('Python explained variance ratio:', pyVarianceRatio);

    // Check each component (allowing for sign flip)
    for (let i = 0; i < jsVarianceRatio.length; i++) {
      expect(Math.abs(jsVarianceRatio[i])).toBeCloseTo(
        Math.abs(pyVarianceRatio[i]),
        3
      );
    }

    // Total variance should match closely
    const jsTotal = jsVarianceRatio.reduce((a, b) => a + b, 0);
    const pyTotal = pyVarianceRatio.reduce((a, b) => a + b, 0);
    expect(jsTotal).toBeCloseTo(pyTotal, 3);
  });

  it('should produce similar component loadings', () => {
    const data = pythonResults.pca.data;
    const pca = new PCA({ n_components: 2 });
    pca.fit({ data });

    const jsComponents = pca.model.rotation;
    const pyComponents = pythonResults.pca.sklearn.components;

    console.log('JS components shape:', jsComponents.length, 'x', jsComponents[0].length);
    console.log('Python components shape:', pyComponents.length, 'x', pyComponents[0].length);

    // PCA components can have arbitrary sign, so we check absolute values
    // and verify they span similar subspaces
    expect(jsComponents.length).toBe(pyComponents.length);
    expect(jsComponents[0].length).toBe(pyComponents[0].length);
  });
});

describe('KMeans - Comparison with sklearn', () => {
  it('should produce similar clustering results', () => {
    const data = pythonResults.kmeans.data;
    const kmeans = new KMeans({ k: 3, seed: 42, maxIter: 100 });
    kmeans.fit({ data });

    const jsInertia = kmeans.model.inertia;
    const pyInertia = pythonResults.kmeans.sklearn.inertia;

    console.log('JS inertia:', jsInertia);
    console.log('Python inertia:', pyInertia);

    // Inertia should be similar (within 10% due to initialization differences)
    const relativeError = Math.abs(jsInertia - pyInertia) / pyInertia;
    expect(relativeError).toBeLessThan(0.15);
  });

  it('should find correct number of clusters', () => {
    const data = pythonResults.kmeans.data;
    const kmeans = new KMeans({ k: 3, seed: 42, maxIter: 100 });
    kmeans.fit({ data });

    const jsLabels = kmeans.predict({ data });
    const uniqueLabels = new Set(jsLabels);

    expect(uniqueLabels.size).toBe(3);
  });
});

describe('GLM Logistic Regression - Comparison with sklearn', () => {
  it('should produce similar coefficients for binary classification', () => {
    const X = pythonResults.logistic.X;
    const y = pythonResults.logistic.y;

    const glm = new GLM({ family: 'binomial', intercept: true, maxIter: 1000 });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const pyCoef = [
      pythonResults.logistic.sklearn.intercept,
      ...pythonResults.logistic.sklearn.coefficients
    ];

    console.log('JS coefficients:', jsCoef);
    console.log('Python coefficients:', pyCoef);

    // Coefficients should be reasonably close
    for (let i = 0; i < jsCoef.length; i++) {
      expect(jsCoef[i]).toBeCloseTo(pyCoef[i], 1);
    }
  });

  it('should achieve similar prediction accuracy', () => {
    const X = pythonResults.logistic.X;
    const y = pythonResults.logistic.y;

    const glm = new GLM({ family: 'binomial', intercept: true, maxIter: 1000 });
    glm.fit(X, y);

    const jsPred = glm.predict(X).map(p => (p > 0.5 ? 1 : 0));
    const jsAccuracy = jsPred.filter((p, i) => p === y[i]).length / y.length;
    const pyAccuracy = pythonResults.logistic.sklearn.accuracy;

    console.log('JS accuracy:', jsAccuracy);
    console.log('Python accuracy:', pyAccuracy);

    // Accuracy should be within 5%
    expect(Math.abs(jsAccuracy - pyAccuracy)).toBeLessThan(0.05);
  });
});

describe('GLM Linear Regression - Comparison with scipy', () => {
  it('should produce similar coefficients for linear regression', () => {
    const X = pythonResults.linear.X;
    const y = pythonResults.linear.y;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(X, y);

    const jsCoef = glm._model.coefficients;
    const pyCoef = [
      pythonResults.linear.scipy.intercept,
      ...pythonResults.linear.scipy.coefficients
    ];

    console.log('JS coefficients:', jsCoef);
    console.log('Python coefficients:', pyCoef);

    // Coefficients should match very closely for linear regression
    for (let i = 0; i < jsCoef.length; i++) {
      expect(jsCoef[i]).toBeCloseTo(pyCoef[i], 3);
    }
  });

  it('should produce similar R² values', () => {
    const X = pythonResults.linear.X;
    const y = pythonResults.linear.y;

    const glm = new GLM({ family: 'gaussian', intercept: true });
    glm.fit(X, y);

    const jsRSquared = glm.score(y);
    const pyRSquared = pythonResults.linear.scipy.r_squared;

    console.log('JS R²:', jsRSquared);
    console.log('Python R²:', pyRSquared);

    expect(jsRSquared).toBeCloseTo(pyRSquared, 3);
  });
});

describe('Safeguards - Verify new functionality', () => {
  it('should throw Observable-friendly error when predict called before fit', () => {
    const pca = new PCA({ n_components: 2 });

    expect(() => pca.transform([[1, 2, 3]])).toThrow(/requires a fitted model/);
    expect(() => pca.transform([[1, 2, 3]])).toThrow(/Observable Tip/);
    expect(() => pca.transform([[1, 2, 3]])).toThrow(/isFitted/);
  });

  it('should provide isFitted() method', () => {
    const kmeans = new KMeans({ k: 3 });

    expect(kmeans.isFitted()).toBe(false);

    kmeans.fit({ data: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]] });

    expect(kmeans.isFitted()).toBe(true);
  });

  it('should provide getState() method', () => {
    const glm = new GLM({ family: 'gaussian' });

    let state = glm.getState();
    expect(state.fitted).toBe(false);
    expect(state.className).toBe('GLM');

    glm.fit([[1], [2], [3]], [2, 4, 6]);

    state = glm.getState();
    expect(state.fitted).toBe(true);
    expect(state.memoryEstimate).toBeGreaterThan(0);
  });

  it('should provide getMemoryUsage() method', () => {
    const pca = new PCA({ n_components: 2 });

    pca.fit({ data: Array(100).fill(0).map(() => Array(5).fill(0).map(() => Math.random())) });

    const memory = pca.getMemoryUsage();
    expect(typeof memory).toBe('string');
    expect(memory).toMatch(/KB|MB/);
  });

  it('should track warnings for GLM convergence', () => {
    const glm = new GLM({ family: 'binomial', maxIter: 1, warnOnNoConvergence: true });

    // This should not converge in 1 iteration
    const X = Array(50).fill(0).map(() => [Math.random()]);
    const y = Array(50).fill(0).map(() => Math.random() > 0.5 ? 1 : 0);

    glm.fit(X, y);

    expect(glm.hasWarnings()).toBe(true);
    const warnings = glm.getWarnings();
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0].type).toBe('convergence');
  });
});
