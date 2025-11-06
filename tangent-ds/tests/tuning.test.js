import { describe, it, expect } from 'vitest';
import { GridSearchCV, RandomSearchCV, distributions } from '../src/ml/tuning.js';
import { GLM } from '../src/stats/index.js';
import * as metrics from '../src/ml/metrics.js';

// Helper to create and fit a linear model using GLM
function fitLinearModel(X, y, options = {}) {
  const model = new GLM({ family: 'gaussian', intercept: options.intercept !== false });
  model.fit(X, y);
  return model;
}

describe('Hyperparameter Tuning', () => {
  // Simple dataset
  const X = [
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
  ];
  const y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

  describe('GridSearchCV', () => {
    it('should search over parameter grid', () => {
      const fitFn = (Xtrain, ytrain, params) => {
        // Dummy params, just fit model
        return fitLinearModel(Xtrain, ytrain, { intercept: true });
      };

      const scoreFn = (model, Xtest, ytest) => {
        const yPred = model.predict(Xtest);
        return metrics.r2(ytest, yPred);
      };

      const paramGrid = {
        param1: [1, 2, 3],
        param2: [0.1, 0.2]
      };

      const result = GridSearchCV(fitFn, scoreFn, X, y, paramGrid, {
        k: 3,
        verbose: false
      });

      expect(result.bestParams).toBeDefined();
      expect(result.bestScore).toBeGreaterThan(0);
      expect(result.bestModel).toBeDefined();
      expect(result.results).toHaveLength(6); // 3 * 2 combinations
    });

    it('should find best parameters', () => {
      let callCount = {};

      const fitFn = (Xtrain, ytrain, params) => {
        // Track which params were tried
        const key = JSON.stringify(params);
        callCount[key] = (callCount[key] || 0) + 1;

        return fitLinearModel(Xtrain, ytrain, { intercept: true });
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramGrid = {
        dummyParam: [1, 2]
      };

      const result = GridSearchCV(fitFn, scoreFn, X, y, paramGrid, {
        k: 2,
        verbose: false
      });

      // Should try all combinations
      expect(Object.keys(callCount).length).toBe(2);
      expect(result.bestScore).toBeGreaterThan(0.8); // Good R² for linear data
    });

    it('should return results for all combinations', () => {
      const fitFn = (Xtrain, ytrain, params) => {
        return fitLinearModel(Xtrain, ytrain);
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramGrid = {
        a: [1, 2],
        b: [3, 4]
      };

      const result = GridSearchCV(fitFn, scoreFn, X, y, paramGrid, {
        k: 2,
        verbose: false
      });

      expect(result.results).toHaveLength(4);
      result.results.forEach(r => {
        expect(r).toHaveProperty('params');
        expect(r).toHaveProperty('meanScore');
        expect(r).toHaveProperty('stdScore');
        expect(r).toHaveProperty('scores');
      });
    });

    it('should handle errors gracefully', () => {
      const fitFn = (Xtrain, ytrain, params) => {
        if (params.bad === true) {
          throw new Error('Bad parameter');
        }
        return fitLinearModel(Xtrain, ytrain);
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramGrid = {
        bad: [false, true]
      };

      const result = GridSearchCV(fitFn, scoreFn, X, y, paramGrid, {
        k: 2,
        verbose: false
      });

      // Should have results for both, but one with error
      expect(result.results).toHaveLength(2);
      const errorResult = result.results.find(r => r.params.bad === true);
      expect(errorResult.error).toBeDefined();
      expect(errorResult.meanScore).toBe(-Infinity);
    });
  });

  describe('RandomSearchCV', () => {
    it('should sample random parameters', () => {
      const fitFn = (Xtrain, ytrain, params) => {
        return fitLinearModel(Xtrain, ytrain);
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramDistributions = {
        param1: [1, 2, 3, 4, 5],
        param2: [0.1, 0.2, 0.3]
      };

      const result = RandomSearchCV(fitFn, scoreFn, X, y, paramDistributions, {
        nIter: 5,
        k: 2,
        verbose: false
      });

      expect(result.results).toHaveLength(5);
      expect(result.bestParams).toBeDefined();
      expect(result.bestScore).toBeGreaterThan(0);
    });

    it('should use distribution objects', () => {
      const fitFn = (Xtrain, ytrain, params) => {
        return fitLinearModel(Xtrain, ytrain);
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramDistributions = {
        param1: distributions.uniform(0, 1),
        param2: distributions.choice([1, 2, 3])
      };

      const result = RandomSearchCV(fitFn, scoreFn, X, y, paramDistributions, {
        nIter: 3,
        k: 2,
        verbose: false
      });

      expect(result.results).toHaveLength(3);
      // Check that uniform parameters are between 0 and 1
      result.results.forEach(r => {
        expect(r.params.param1).toBeGreaterThanOrEqual(0);
        expect(r.params.param1).toBeLessThanOrEqual(1);
        expect([1, 2, 3]).toContain(r.params.param2);
      });
    });

    it('should respect nIter parameter', () => {
      const fitFn = (Xtrain, ytrain, params) => {
        return fitLinearModel(Xtrain, ytrain);
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramDistributions = {
        param: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      };

      const result = RandomSearchCV(fitFn, scoreFn, X, y, paramDistributions, {
        nIter: 4,
        k: 2,
        verbose: false
      });

      expect(result.results).toHaveLength(4);
    });
  });

  describe('distributions', () => {
    it('should create uniform distribution', () => {
      const dist = distributions.uniform(0, 10);
      expect(dist.type).toBe('uniform');
      expect(dist.low).toBe(0);
      expect(dist.high).toBe(10);
    });

    it('should create loguniform distribution', () => {
      const dist = distributions.loguniform(0.001, 1);
      expect(dist.type).toBe('loguniform');
      expect(dist.low).toBe(0.001);
      expect(dist.high).toBe(1);
    });

    it('should create randint distribution', () => {
      const dist = distributions.randint(1, 100);
      expect(dist.type).toBe('randint');
      expect(dist.low).toBe(1);
      expect(dist.high).toBe(100);
    });

    it('should create choice distribution', () => {
      const options = ['adam', 'sgd', 'rmsprop'];
      const dist = distributions.choice(options);
      expect(dist.type).toBe('choice');
      expect(dist.options).toEqual(options);
    });
  });

  describe('Integration', () => {
    it('should work end-to-end with real model', () => {
      // Larger dataset
      const XLarge = Array.from({ length: 50 }, (_, i) => [i + 1]);
      const yLarge = XLarge.map(x => 2 * x[0] + 1 + (Math.random() - 0.5));

      const fitFn = (Xtrain, ytrain, params) => {
        // params doesn't affect linear model, but we test the flow
        return fitLinearModel(Xtrain, ytrain, { intercept: true });
      };

      const scoreFn = (model, Xtest, ytest) => {
        return metrics.r2(ytest, model.predict(Xtest));
      };

      const paramGrid = {
        dummy1: [1, 2],
        dummy2: [3, 4]
      };

      const result = GridSearchCV(fitFn, scoreFn, XLarge, yLarge, paramGrid, {
        k: 5,
        verbose: false
      });

      expect(result.bestScore).toBeGreaterThan(0.95); // Should fit very well
      expect(result.bestModel).toBeDefined();
      expect(result.bestModel.coefficients).toBeDefined();
    });
  });
});
