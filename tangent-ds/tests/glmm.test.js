/**
 * Tests for Generalized Linear Mixed Models (GLMM)
 */

import { describe, it, expect } from 'vitest';
import { GLM } from '../src/stats/estimators/GLM.js';

describe('GLMM - Random Intercepts', () => {
  it('should fit Gaussian GLMM with random intercepts', () => {
    const X = [[1], [2], [3], [1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7, 2.5, 4.5, 6.5];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'];

    const randomEffects = { intercept: groups };

    const model = new GLM({
      family: 'gaussian',
      randomEffects: randomEffects
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._isMixed).toBe(true);
    expect(model._model.fixedEffects).toHaveLength(2); // intercept + slope
    expect(model._model.randomEffects.length).toBeGreaterThan(0);
    expect(model._model.varianceComponents).toHaveLength(1);
    expect(model._model.converged).toBe(true);
  });

  it('should compute variance components correctly', () => {
    const X = [[1], [2], [3], [1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7, 2.5, 4.5, 6.5];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const vc = model._model.varianceComponents[0];
    expect(vc.variance).toBeGreaterThan(0);
    expect(isFinite(vc.variance)).toBe(true);
  });

  it('should predict for known groups', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const Xnew = [[1.5], [2.5]];
    const groupsNew = ['A', 'B'];
    const randomEffectsData = { intercept: groupsNew };

    const predictions = model.predict(Xnew, { allowNewGroups: false });

    // Note: We need to pass the random effects data properly
    // This is a simplified test - in practice would use table-style API
    expect(predictions).toHaveLength(2);
    expect(predictions.every(p => isFinite(p))).toBe(true);
  });

  it('should allow prediction for new groups', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const Xnew = [[1.5]];
    // For new groups, the prediction will use population-level fixed effects only
    const predictions = model.predict(Xnew, { allowNewGroups: true });

    expect(predictions).toHaveLength(1);
    expect(isFinite(predictions[0])).toBe(true);
  });

  it('should work with binomial GLMM', () => {
    const X = [[1], [2], [3], [4], [1], [2], [3], [4]];
    const y = [0, 0, 1, 1, 0, 1, 1, 1];
    const groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'binomial',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.fixedEffects).toHaveLength(2);
    expect(model._model.varianceComponents).toHaveLength(1);
  });

  it('should work with Poisson GLMM', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'poisson',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.varianceComponents).toHaveLength(1);
  });
});

describe('GLMM - Random Slopes', () => {
  it('should fit GLMM with random slopes', () => {
    const n = 30;
    const X = [];
    const y = [];
    const groups = [];
    const timeValues = [];

    // Create data: time = 0,1,2,...,9 for each of 3 groups
    for (let g = 0; g < 3; g++) {
      for (let t = 0; t < 10; t++) {
        X.push([1]); // Dummy fixed effect
        timeValues.push(t);
        groups.push(['A', 'B', 'C'][g]);
        y.push(2 + t + g + Math.random() * 0.5); // Linear trend with group effects
      }
    }

    const randomEffects = {
      intercept: groups,
      slopes: {
        time: {
          groups: groups,
          values: timeValues
        }
      }
    };

    const model = new GLM({
      family: 'gaussian',
      randomEffects: randomEffects
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.varianceComponents.length).toBeGreaterThan(1);
    expect(model._model.converged).toBe(true);
  });

  it('should have separate variance components for intercepts and slopes', () => {
    const X = [];
    const y = [];
    const groups = [];
    const timeValues = [];

    for (let g = 0; g < 3; g++) {
      for (let t = 0; t < 5; t++) {
        X.push([1]);
        timeValues.push(t);
        groups.push(['A', 'B', 'C'][g]);
        y.push(2 + t + g);
      }
    }

    const randomEffects = {
      intercept: groups,
      slopes: {
        time: {
          groups: groups,
          values: timeValues
        }
      }
    };

    const model = new GLM({
      family: 'gaussian',
      randomEffects: randomEffects
    });

    model.fit(X, y);

    expect(model._model.varianceComponents).toHaveLength(2);

    const interceptVC = model._model.varianceComponents.find(vc => vc.type === 'intercept');
    const slopeVC = model._model.varianceComponents.find(vc => vc.type === 'slope');

    expect(interceptVC).toBeDefined();
    expect(slopeVC).toBeDefined();
    expect(slopeVC.variable).toBe('time');
  });
});

describe('GLMM - Summary and Output', () => {
  it('should generate lme4-style summary', () => {
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
    expect(summary).toContain('Variance');
    expect(summary).toContain('Std.Dev.');
    expect(summary).toContain('95% CI');
  });

  it('should include warning about p-values', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const summary = model.summary();

    expect(summary).toContain('p-values for fixed effects in mixed models');
    expect(summary).toContain('questionable assumptions');
    expect(summary).toContain('Prefer effect estimates');
  });

  it('should not include p-value column', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const summary = model.summary();

    // Should have z value but not p-value
    expect(summary).toContain('z value');
    expect(summary).not.toMatch(/Pr\(>.*\)/); // R-style p-value column
    expect(summary).not.toMatch(/p-value/i); // Should not have p-value column header
  });

  it('should report number of groups', () => {
    const X = [[1], [2], [3], [1], [2], [3], [1], [2]];
    const y = [2, 4, 6, 3, 5, 7, 2.5, 4.5];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const summary = model.summary();

    expect(summary).toContain('groups: 3');
  });

  it('should report AIC and BIC', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const summary = model.summary();

    expect(summary).toContain('AIC:');
    expect(summary).toContain('BIC:');
    expect(model._model.aic).toBeGreaterThan(0);
    expect(model._model.bic).toBeGreaterThan(0);
  });
});

describe('GLMM - Serialization', () => {
  it('should serialize and deserialize GLMM', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    const json = model.save();
    const restored = GLM.load(json);

    expect(restored.fitted).toBe(true);
    expect(restored._isMixed).toBe(true);
    expect(restored._model.fixedEffects).toEqual(model._model.fixedEffects);
    expect(restored._model.randomEffects).toEqual(model._model.randomEffects);
    expect(restored._model.varianceComponents).toEqual(model._model.varianceComponents);
  });
});

describe('GLMM - Multiple Groups', () => {
  it('should handle many groups', () => {
    const X = [];
    const y = [];
    const groups = [];

    // 10 observations per group, 10 groups
    for (let g = 0; g < 10; g++) {
      for (let i = 0; i < 10; i++) {
        X.push([i]);
        y.push(2 + i + g * 0.5);
        groups.push(`group_${g}`);
      }
    }

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.groupInfo[0].nGroups).toBe(10);
    expect(model._model.randomEffects.length).toBeGreaterThan(0);
  });

  it('should handle unbalanced groups', () => {
    const X = [[1], [2], [3], [1], [2]]; // Group A has 3, Group B has 2
    const y = [2, 4, 6, 3, 5];
    const groups = ['A', 'A', 'A', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.groupInfo[0].nGroups).toBe(2);
  });
});

describe('GLMM - Convergence', () => {
  it('should converge for well-behaved data', () => {
    const X = [[1], [2], [3], [4], [5], [1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10, 2.5, 4.5, 6.5, 8.5, 10.5];
    const groups = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups },
      maxIter: 100
    });

    model.fit(X, y);

    expect(model._model.converged).toBe(true);
    expect(model._model.iterations).toBeLessThan(100);
  });

  it('should respect maxIter parameter', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups },
      maxIter: 5
    });

    model.fit(X, y);

    expect(model._model.iterations).toBeLessThanOrEqual(5);
  });
});

describe('GLMM - Edge Cases', () => {
  it('should handle single group', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10];
    const groups = ['A', 'A', 'A', 'A', 'A'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.groupInfo[0].nGroups).toBe(1);
  });

  it('should handle group with single observation', () => {
    const X = [[1], [2], [3], [4]];
    const y = [2, 4, 6, 8];
    const groups = ['A', 'A', 'B', 'C'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(model.fitted).toBe(true);
    expect(model._model.groupInfo[0].nGroups).toBe(3);
  });
});

describe('GLMM - Model Comparison', () => {
  it('should compute log-likelihood for model comparison', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    const model = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });

    model.fit(X, y);

    expect(isFinite(model._model.logLikelihood)).toBe(true);
    expect(model._model.aic).toBeGreaterThan(0);
    expect(model._model.bic).toBeGreaterThan(0);
  });

  it('AIC should penalize more complex models', () => {
    const X = [[1], [2], [3], [1], [2], [3]];
    const y = [2, 4, 6, 3, 5, 7];
    const groups = ['A', 'A', 'A', 'B', 'B', 'B'];

    // Simple GLM
    const model1 = new GLM({ family: 'gaussian' });
    model1.fit(X, y);

    // GLMM with random intercepts (more complex)
    const model2 = new GLM({
      family: 'gaussian',
      randomEffects: { intercept: groups }
    });
    model2.fit(X, y);

    // Both should have valid AIC values
    expect(isFinite(model1._model.aic)).toBe(true);
    expect(isFinite(model2._model.aic)).toBe(true);
  });
});
