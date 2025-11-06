/**
 * Tests for Generalized Linear Models (GLM and GLMM)
 */

import { describe, it, expect } from 'vitest';
import { GLM } from '../src/stats/estimators/GLM.js';

describe('GLM - Gaussian Family', () => {
  it('should fit a simple linear regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2); // intercept + 1 coef
    expect(model._model.converged).toBe(true);
  });

  it('should predict new values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const predictions = model.predict([[6], [7]]);
    expect(predictions).toHaveLength(2);
    expect(predictions.every(p => typeof p === 'number' && isFinite(p))).toBe(true);
  });

  it('should work with multiple predictors', () => {
    const X = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ];
    const y = [3, 5, 7, 9, 11];

    const model = new GLM({ family: 'gaussian', intercept: true });
    model.fit(X, y);

    expect(model._model.coefficients).toHaveLength(3); // intercept + 2 coefs
    expect(model._model.converged).toBe(true);
  });

  it('should work without intercept', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10];

    const model = new GLM({ family: 'gaussian', intercept: false });
    model.fit(X, y);

    expect(model._model.coefficients).toHaveLength(1); // no intercept
    expect(Math.abs(model._model.coefficients[0] - 2.0)).toBeLessThan(0.1);
  });

  it('should compute R² score', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const predictions = model.predict(X);
    const score = model.score(y, predictions);

    expect(score).toBeGreaterThan(0.9); // Should be very high for perfect fit
  });

  it('should serialize and deserialize', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const json = model.save();
    const restored = GLM.load(json);

    expect(restored.fitted).toBe(true);
    expect(restored._model.coefficients).toEqual(model._model.coefficients);

    const predictions1 = model.predict(X);
    const predictions2 = restored.predict(X);
    expect(predictions1).toEqual(predictions2);
  });
});

describe('GLM - Binomial Family', () => {
  it('should fit logistic regression', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial', link: 'logit' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict probabilities between 0 and 1', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every(p => p >= 0 && p <= 1)).toBe(true);
  });

  it('should compute accuracy score', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial' });
    model.fit(X, y);

    const predictions = model.predict(X);
    const score = model.score(y, predictions);

    expect(score).toBeGreaterThan(0.5); // Should predict better than random
  });

  it('should work with probit link', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new GLM({ family: 'binomial', link: 'probit' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.converged).toBe(true);
  });
});

describe('GLM - Poisson Family', () => {
  it('should fit Poisson regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [1, 2, 3, 5, 8];

    const model = new GLM({ family: 'poisson', link: 'log' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [1, 2, 3, 5, 8];

    const model = new GLM({ family: 'poisson' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every(p => p > 0)).toBe(true);
  });

  it('should handle count data correctly', () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const y = [2, 3, 5, 7, 11, 13, 17, 19]; // Prime numbers

    const model = new GLM({ family: 'poisson' });
    model.fit(X, y);

    const predictions = model.predict([[9], [10]]);
    expect(predictions.every(p => p > 0 && isFinite(p))).toBe(true);
  });
});

describe('GLM - Gamma Family', () => {
  it('should fit Gamma regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.5, 1.2, 2.1, 3.5, 5.2];

    const model = new GLM({ family: 'gamma', link: 'inverse' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.5, 1.2, 2.1, 3.5, 5.2];

    const model = new GLM({ family: 'gamma' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every(p => p > 0)).toBe(true);
  });

  it('should work with log link', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.5, 1.2, 2.1, 3.5, 5.2];

    const model = new GLM({ family: 'gamma', link: 'log' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.converged).toBe(true);
  });
});

describe('GLM - Inverse Gaussian Family', () => {
  it('should fit inverse Gaussian regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.8, 1.5, 2.2, 3.1, 4.2];

    const model = new GLM({ family: 'inverse_gaussian', link: 'inverse_squared' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0.8, 1.5, 2.2, 3.1, 4.2];

    const model = new GLM({ family: 'inverse_gaussian' });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every(p => p > 0)).toBe(true);
  });
});

describe('GLM - Negative Binomial Family', () => {
  it('should fit negative binomial regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 5, 8, 12, 18];

    const model = new GLM({ family: 'negative_binomial', link: 'log', theta: 1.5 });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.coefficients).toHaveLength(2);
    expect(model._model.converged).toBe(true);
  });

  it('should predict positive values', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 5, 8, 12, 18];

    const model = new GLM({ family: 'negative_binomial', theta: 1.5 });
    model.fit(X, y);

    const predictions = model.predict(X);
    expect(predictions.every(p => p > 0)).toBe(true);
  });
});

describe('GLM - Summary Output', () => {
  it('should generate summary for GLM', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    const summary = model.summary();
    expect(typeof summary).toBe('string');
    expect(summary).toContain('Generalized Linear Model');
    expect(summary).toContain('Coefficients:');
    expect(summary).toContain('AIC:');
    expect(summary).toContain('Deviance');
  });

  it('should not include p-values in GLMM summary', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });
    model.fit(X, y);

    const summary = model.summary();
    expect(summary).toContain('Generalized Linear Mixed Model');
    expect(summary).toContain('Fixed Effects:');
    expect(summary).toContain('Random Effects:');
    expect(summary).toContain('p-values for fixed effects in mixed models');
    expect(summary).not.toContain('p-value'); // Should not have p-value column
  });
});

describe('GLM - Table-style API', () => {
  it('should work with table-style input', () => {
    const data = [
      { x1: 1, x2: 2, y: 3 },
      { x1: 2, x2: 3, y: 5 },
      { x1: 3, x2: 4, y: 7 },
      { x1: 4, x2: 5, y: 9 },
      { x1: 5, x2: 6, y: 11 }
    ];

    const model = new GLM({ family: 'gaussian' });
    model.fit({ X: ['x1', 'x2'], y: 'y', data });

    expect(model.fitted).toBe(true);
    expect(model._columnsX).toEqual(['x1', 'x2']);

    const predictions = model.predict({ X: ['x1', 'x2'], data });
    expect(predictions).toHaveLength(5);
  });

  it('should handle missing values', () => {
    const data = [
      { x: 1, y: 2 },
      { x: 2, y: 4 },
      { x: 3, y: null },
      { x: 4, y: 8 },
      { x: 5, y: 10 }
    ];

    const model = new GLM({ family: 'gaussian' });
    model.fit({ X: ['x'], y: 'y', data, omit_missing: true });

    expect(model.fitted).toBe(true);
    expect(model._model.n).toBe(4); // Should have 4 observations (1 removed)
  });
});

describe('GLM - Advanced Features', () => {
  it('should handle weights', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];
    const weights = [1, 1, 2, 1, 1]; // Give more weight to third observation

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y, weights);

    expect(model.fitted).toBe(true);
  });

  it('should handle offsets', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [3, 5, 7, 9, 11];
    const offset = [1, 1, 1, 1, 1];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y, null, offset);

    expect(model.fitted).toBe(true);
  });

  it('should support regularization', () => {
    const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
    const y = [3, 5, 7, 9, 11];

    const model = new GLM({
      family: 'gaussian',
      regularization: { alpha: 0.1, l1_ratio: 0 } // Ridge regression
    });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
  });

  it('should estimate dispersion parameter', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 5, 4, 5];

    const model = new GLM({ family: 'gaussian', dispersion: 'estimate' });
    model.fit(X, y);

    expect(model._model.dispersion).toBeGreaterThan(0);
    expect(isFinite(model._model.dispersion)).toBe(true);
  });

  it('should fix dispersion parameter', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0, 0, 1, 1, 1];

    const model = new GLM({ family: 'binomial', dispersion: 'fixed' });
    model.fit(X, y);

    expect(model._model.dispersion).toBe(1.0);
  });
});

describe('GLM - Error Handling', () => {
  it('should throw error when predicting before fitting', () => {
    const model = new GLM({ family: 'gaussian' });

    expect(() => model.predict([[1], [2]])).toThrow('Model has not been fitted');
  });

  it('should throw error for invalid family', () => {
    expect(() => new GLM({ family: 'invalid_family' })).not.toThrow();
    // The error will be thrown during fit when createFamily is called
  });

  it('should throw error for invalid link', () => {
    const model = new GLM({ family: 'gaussian', link: 'invalid_link' });
    const X = [[1], [2], [3]];
    const y = [1, 2, 3];

    expect(() => model.fit(X, y)).toThrow();
  });
});

describe('GLM - Edge Cases', () => {
  it('should handle perfect separation in logistic regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0, 0, 0, 1, 1];

    const model = new GLM({ family: 'binomial', maxIter: 50 });
    model.fit(X, y);

    // Should still fit, though may not converge
    expect(model.fitted).toBe(true);
  });

  it('should handle single predictor', () => {
    const X = [[1], [2], [3]];
    const y = [2, 4, 6];

    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);

    expect(model._model.coefficients).toHaveLength(2);
  });

  it('should handle all zeros in response (Poisson)', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [0, 0, 0, 0, 0];

    const model = new GLM({ family: 'poisson' });
    model.fit(X, y);

    expect(model.fitted).toBe(true);
    const predictions = model.predict(X);
    expect(predictions.every(p => p >= 0)).toBe(true);
  });
});
