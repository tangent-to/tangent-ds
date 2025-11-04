import { describe, it, expect } from 'vitest';
import { fit, transform } from '../src/mva/rda.js';
import { RDA, pca as pcaFns } from '../src/mva/index.js';
import { approxEqual } from '../src/core/math.js';

describe('RDA - Redundancy Analysis', () => {
  describe('fit', () => {
    it('should fit RDA model', () => {
      // Simple case: Y depends on X
      const Y = [
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8]
      ];
      const X = [
        [1],
        [2],
        [3],
        [4]
      ];
      
      const model = fit(Y, X);
      
      expect(model.canonicalScores.length).toBe(4);
      expect(model.canonicalLoadings.length).toBe(2);
      expect(model.eigenvalues.length).toBeGreaterThan(0);
      expect(model.constrainedVariance).toBeGreaterThan(0);
      expect(model.constrainedVariance).toBeLessThanOrEqual(1);
    });

    it('should explain high variance for strong relationship', () => {
      // Strong linear relationship
      const Y = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]];
      const X = [[1], [2], [3], [4], [5]];
      
      const model = fit(Y, X);
      
      // Should explain most of the variance
      expect(model.constrainedVariance).toBeGreaterThan(0.9);
    });

    it('should handle multiple response variables', () => {
      const Y = [
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
      ];
      const X = [[1], [2], [3]];
      
      const model = fit(Y, X);
      
      expect(model.canonicalLoadings.length).toBe(3);
      expect(model.q).toBe(3);
    });

    it('should handle multiple explanatory variables', () => {
      const Y = [[1], [2], [3], [4]];
      const X = [[1, 0], [2, 0], [3, 0.1], [4, 0]];
      
      const model = fit(Y, X);
      
      expect(model.p).toBe(2);
      expect(model.coefficients.length).toBe(1); // 1 response variable
      expect(model.coefficients[0].length).toBe(2); // 2 predictors
    });

    it('should support residual (unconstrained) ordination', () => {
      const Y = [
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
        [4, 8, 12],
      ];
      const X = [[1], [2], [3], [4]];

      const constrainedModel = fit(Y, X, { constrained: true });
      const residualModel = fit(Y, X, { constrained: false });

      expect(constrainedModel.constrained).toBe(true);
      expect(residualModel.constrained).toBe(false);
      expect(residualModel.constraintScores.length).toBe(0);

      const residualPCA = pcaFns.fit(residualModel.rawResiduals, {
        scale: false,
        center: false,
        scaling: residualModel.scaling,
      });

      residualModel.eigenvalues.forEach((eig, idx) => {
        expect(eig).toBeCloseTo(residualPCA.eigenvalues[idx], 6);
      });
    });

    it('should retain response names when fitted from declarative data', () => {
      const data = [
        { y1: 1, y2: 2, x1: 5, x2: 10 },
        { y1: 2, y2: 4, x1: 6, x2: 12 },
        { y1: 3, y2: 6, x1: 7, x2: 14 },
        { y1: 4, y2: 8, x1: 8, x2: 16 }
      ];

      const model = fit({
        data,
        response: ['y1', 'y2'],
        predictors: ['x1', 'x2']
      });

      expect(model.responseNames).toEqual(['y1', 'y2']);
      expect(model.predictorNames).toEqual(['x1', 'x2']);
      expect(model.canonicalLoadings[0].variable).toBe('y1');
      expect(model.canonicalLoadings[1].variable).toBe('y2');
    });

    it('should throw error for mismatched dimensions', () => {
      const Y = [[1], [2]];
      const X = [[1]];
      
      expect(() => fit(Y, X)).toThrow();
    });

    it('should throw error for insufficient samples', () => {
      const Y = [[1], [2]];
      const X = [[1], [2]];
      
      expect(() => fit(Y, X)).toThrow();
    });
  });

  describe('transform', () => {
    it('should transform new data', () => {
      const Y = [[1, 2], [2, 4], [3, 6]];
      const X = [[1], [2], [3]];
      
      const model = fit(Y, X);
      
      const Ynew = [[4, 8]];
      const Xnew = [[4]];
      const transformed = transform(model, Ynew, Xnew);
      
      expect(transformed.length).toBe(1);
      expect(transformed[0].rda1).toBeDefined();
    });

    it('should apply same centering as training', () => {
      const Y = [[10, 20], [11, 21], [12, 22]];
      const X = [[1], [2], [3]];
      
      const model = fit(Y, X);
      
      // Transform data at mean
      const Ynew = [[11, 21]];
      const Xnew = [[2]];
      const transformed = transform(model, Ynew, Xnew);
      
      expect(transformed.length).toBe(1);
      expect(typeof transformed[0].rda1).toBe('number');
    });
  });

  describe('constrained variance', () => {
    it('should be between 0 and 1', () => {
      const Y = [[1, 2], [2, 3], [3, 5]];
      const X = [[1], [2], [3]];
      
      const model = fit(Y, X);
      
      expect(model.constrainedVariance).toBeGreaterThanOrEqual(0);
      expect(model.constrainedVariance).toBeLessThanOrEqual(1);
    });

    it('should be low when Y does not depend on X', () => {
      const Y = [[1, 2], [3, 4], [5, 6], [7, 8]];
      const X = [[1], [1], [1], [1]]; // Constant X
      
      const model = fit(Y, X);
      
      // Should explain little variance
      expect(model.constrainedVariance).toBeLessThan(0.1);
    });
  });
});

describe('RDA - class API', () => {
  it('should fit and transform with class wrapper', () => {
    const Y = [[1, 2], [2, 4], [3, 6]];
    const X = [[1], [2], [3]];

    const estimator = new RDA();
    estimator.fit(Y, X);

    const summary = estimator.summary();
    expect(summary.constrainedVariance).toBeGreaterThanOrEqual(0);

    const transformed = estimator.transform(Y, X);
    expect(transformed.length).toBe(3);
    expect(transformed[0].rda1).toBeDefined();
  });

  it('should keep response names when using declarative fit', () => {
    const data = [
      { y1: 1, y2: 3, x1: 2, x2: 4 },
      { y1: 2, y2: 6, x1: 3, x2: 6 },
      { y1: 3, y2: 9, x1: 4, x2: 8 },
      { y1: 4, y2: 12, x1: 5, x2: 10 }
    ];

    const estimator = new RDA();
    estimator.fit({
      data,
      response: ['y1', 'y2'],
      predictors: ['x1', 'x2']
    });

    expect(estimator.model.responseNames).toEqual(['y1', 'y2']);
    expect(estimator.model.predictorNames).toEqual(['x1', 'x2']);
    expect(estimator.model.canonicalLoadings[0].variable).toBe('y1');
  });

  it('getScores exposes raw and scaled site/loadings/constraints', () => {
    const Y = [
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
    ];
    const X = [
      [1, 0],
      [2, 1],
      [3, 0],
      [4, 1],
    ];

    const estimator = new RDA({ scaling: 1 });
    estimator.fit(Y, X);

    const scaledSites = estimator.getScores('sites');
    const rawSites = estimator.getScores('sites', false);
    expect(scaledSites.length).toBe(rawSites.length);
    expect(Math.abs(scaledSites[0].rda1 - rawSites[0].rda1)).toBeGreaterThan(0);

    const correlationEstimator = new RDA({ scaling: 2 });
    correlationEstimator.fit(Y, X);

    const scaledResponses = correlationEstimator.getScores('responses');
    const rawResponses = correlationEstimator.getScores('responses', false);
    expect(Math.abs(scaledResponses[0].rda1 - rawResponses[0].rda1)).toBeGreaterThan(0);

    const scaledConstraints = correlationEstimator.getScores('constraints');
    const rawConstraints = correlationEstimator.getScores('constraints', false);
    expect(Math.abs(scaledConstraints[0].rda1 - rawConstraints[0].rda1)).toBeGreaterThan(0);
  });
});
