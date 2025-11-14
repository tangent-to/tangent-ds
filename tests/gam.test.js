import { describe, it, expect } from 'vitest';
import { GAMRegressor, GAMClassifier } from '../src/ml/index.js';

describe('GAM estimators', () => {
  describe('Basic functionality', () => {
    it('should fit smooth regression curve', () => {
      const X = [];
      const y = [];
      for (let i = 0; i < 80; i++) {
        const x = -2 + (4 * i) / 79;
        X.push([x]);
        y.push(Math.sin(x));
      }

      const gam = new GAMRegressor({ nSplines: 5 });
      gam.fit(X, y);

      const preds = gam.predict([[0], [Math.PI / 2]]);
      expect(preds[0]).toBeCloseTo(0, 1);
      expect(preds[1]).toBeGreaterThan(0.7);
    });

    it('should separate two classes', () => {
      const X = [];
      const y = [];
      for (let i = 0; i < 50; i++) {
        const x = i / 50;
        X.push([x]);
        y.push(x > 0.5 ? 'B' : 'A');
      }

      const gam = new GAMClassifier({ nSplines: 4, maxIter: 50 });
      gam.fit(X, y);

      const preds = gam.predict([[0.2], [0.8]]);
      expect(preds).toEqual(['A', 'B']);
    });
  });

  describe('Basis types', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 50; i++) {
      const x = i / 49;
      X.push([x]);
      y.push(Math.sin(2 * Math.PI * x));
    }

    it('should work with truncated power basis (tp)', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'tp' });
      gam.fit(X, y);
      const preds = gam.predict([[0.25], [0.75]]);
      expect(preds[0]).toBeCloseTo(1, 0);
      expect(preds[1]).toBeCloseTo(-1, 0);
    });

    it('should work with cubic regression splines (cr)', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'cr' });
      gam.fit(X, y);
      const preds = gam.predict([[0.25], [0.75]]);
      expect(preds[0]).toBeGreaterThan(0.5);
      expect(preds[1]).toBeLessThan(-0.5);
    });

    it('should work with B-splines (bs)', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'bs' });
      gam.fit(X, y);
      const preds = gam.predict([[0.25], [0.75]]);
      expect(preds[0]).toBeGreaterThan(0.5);
      expect(preds[1]).toBeLessThan(-0.5);
    });
  });

  describe('Smoothness selection', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 100; i++) {
      const x = i / 99;
      X.push([x]);
      y.push(Math.sin(2 * Math.PI * x) + (Math.random() - 0.5) * 0.1);
    }

    it('should use GCV for automatic smoothness selection', () => {
      const gam = new GAMRegressor({ nSplines: 10, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      // Should smooth out noise
      const preds = gam.predict([[0.25], [0.75]]);
      expect(preds[0]).toBeGreaterThan(0.5);
      expect(preds[1]).toBeLessThan(-0.5);

      // Should have EDF < number of basis functions (due to penalty)
      const summary = gam.summary();
      expect(summary.edf).toBeLessThan(15);
    });

    it('should use REML for automatic smoothness selection', () => {
      const gam = new GAMRegressor({ nSplines: 10, basis: 'cr', smoothMethod: 'REML' });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary.edf).toBeGreaterThan(1);
      expect(summary.edf).toBeLessThan(15);
    });

    it('should use fixed lambda when specified', () => {
      const gam = new GAMRegressor({ nSplines: 10, basis: 'cr', lambda: 0.01 });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary.edf).toBeGreaterThan(1);
      expect(summary.edf).toBeLessThan(15);
    });

    it('should use no penalty when smoothMethod and lambda are null', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'cr', smoothMethod: null, lambda: null });
      gam.fit(X, y);

      const summary = gam.summary();
      // With no penalty, EDF should be close to number of parameters
      expect(summary.edf).toBeGreaterThan(5);
    });
  });

  describe('Statistical inference', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 100; i++) {
      const x = i / 99;
      X.push([x]);
      y.push(2 * x + Math.sin(4 * Math.PI * x));
    }

    it('should compute EDF for smooth terms', () => {
      const gam = new GAMRegressor({ nSplines: 8, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary.smoothTerms).toHaveLength(1);
      expect(summary.smoothTerms[0].edf).toBeGreaterThan(0);
      expect(summary.smoothTerms[0].edf).toBeLessThan(summary.smoothTerms[0].refDf);
    });

    it('should compute p-values for smooth terms', () => {
      const gam = new GAMRegressor({ nSplines: 8, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary.smoothTerms[0].pValue).toBeGreaterThan(0);
      expect(summary.smoothTerms[0].pValue).toBeLessThan(1);
    });

    it('should compute summary statistics', () => {
      const gam = new GAMRegressor({ nSplines: 8, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary).toHaveProperty('coefficients');
      expect(summary).toHaveProperty('smoothTerms');
      expect(summary).toHaveProperty('residualStdError');
      expect(summary).toHaveProperty('rSquared');
      expect(summary).toHaveProperty('edf');
      expect(summary).toHaveProperty('n');
      expect(summary.rSquared).toBeGreaterThan(0.8);
    });
  });

  describe('Confidence intervals', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 80; i++) {
      const x = -2 + (4 * i) / 79;
      X.push([x]);
      y.push(Math.sin(x) + (Math.random() - 0.5) * 0.1);
    }

    it('should compute confidence intervals for predictions', () => {
      const gam = new GAMRegressor({ nSplines: 8, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const result = gam.predictWithInterval([[0], [Math.PI / 2]], 0.95);

      expect(result).toHaveProperty('fitted');
      expect(result).toHaveProperty('se');
      expect(result).toHaveProperty('lower');
      expect(result).toHaveProperty('upper');

      expect(result.fitted).toHaveLength(2);
      expect(result.lower[0]).toBeLessThan(result.fitted[0]);
      expect(result.upper[0]).toBeGreaterThan(result.fitted[0]);
    });

    it('should have narrower intervals with higher confidence level', () => {
      const gam = new GAMRegressor({ nSplines: 8, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const result90 = gam.predictWithInterval([[0]], 0.90);
      const result95 = gam.predictWithInterval([[0]], 0.95);

      const width90 = result90.upper[0] - result90.lower[0];
      const width95 = result95.upper[0] - result95.lower[0];

      expect(width90).toBeLessThan(width95);
    });
  });

  describe('Multiple features', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 100; i++) {
      const x1 = i / 99;
      const x2 = (i % 10) / 9;
      X.push([x1, x2]);
      y.push(Math.sin(2 * Math.PI * x1) + 0.5 * Math.cos(4 * Math.PI * x2));
    }

    it('should fit GAM with multiple smooth terms', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary.smoothTerms).toHaveLength(2);
      expect(summary.smoothTerms[0].term).toBe('s(x0)');
      expect(summary.smoothTerms[1].term).toBe('s(x1)');
    });

    it('should provide EDF for each smooth term', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'cr', smoothMethod: 'GCV' });
      gam.fit(X, y);

      const summary = gam.summary();
      expect(summary.smoothTerms[0].edf).toBeGreaterThan(0);
      expect(summary.smoothTerms[1].edf).toBeGreaterThan(0);
    });
  });

  describe('Knot placement', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 100; i++) {
      const x = i / 99;
      X.push([x]);
      y.push(Math.sin(2 * Math.PI * x));
    }

    it('should work with quantile knot placement', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'cr', knotPlacement: 'quantile' });
      gam.fit(X, y);

      const preds = gam.predict([[0.25], [0.75]]);
      expect(preds[0]).toBeGreaterThan(0.5);
      expect(preds[1]).toBeLessThan(-0.5);
    });

    it('should work with uniform knot placement', () => {
      const gam = new GAMRegressor({ nSplines: 6, basis: 'cr', knotPlacement: 'uniform' });
      gam.fit(X, y);

      const preds = gam.predict([[0.25], [0.75]]);
      expect(preds[0]).toBeGreaterThan(0.5);
      expect(preds[1]).toBeLessThan(-0.5);
    });
  });
});
