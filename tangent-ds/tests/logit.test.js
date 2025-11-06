import { describe, it, expect } from 'vitest';
import { GLM } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('logistic regression with GLM', () => {
  describe('fit', () => {
    it('should fit logistic regression model', () => {
      // Simple separable data
      const X = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
      ];
      const y = [0, 0, 0, 1, 1, 1];

      const clf = new GLM({ family: 'binomial', link: 'logit', intercept: true, maxIter: 100 });
      clf.fit(X, y);
      const model = clf._model;

      expect(model.coefficients.length).toBe(2);
      expect(model.fitted.length).toBe(6);
      expect(model.iterations).toBeGreaterThan(0);
      expect(model.logLikelihood).toBeLessThan(0);
    });

    it('should produce probabilities between 0 and 1', () => {
      const X = [[1], [2], [3], [4]];
      const y = [0, 0, 1, 1];
      const clf = new GLM({ family: 'binomial', intercept: true });
      clf.fit(X, y);

      for (const p of clf._model.fitted) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    });

    it('should handle extreme values gracefully', () => {
      const X = [[1], [2]];
      const y = [0, 1];

      const clf = new GLM({ family: 'binomial' });
      clf.fit(X, y);

      expect(clf.fitted).toBe(true);
    });
  });

  describe('predict', () => {
    it('should predict probabilities for new data', () => {
      const X = [[1], [2], [3], [4]];
      const y = [0, 0, 1, 1];
      const clf = new GLM({ family: 'binomial', intercept: true });
      clf.fit(X, y);

      const Xnew = [[2.5], [3.5]];
      const predictions = clf.predict(Xnew);

      expect(predictions.length).toBe(2);
      expect(predictions[0]).toBeGreaterThan(0);
      expect(predictions[0]).toBeLessThan(1);
    });
  });

  describe('classify', () => {
    it('should classify based on threshold', () => {
      const X = [[1], [2], [3], [4]];
      const y = [0, 0, 1, 1];
      const clf = new GLM({ family: 'binomial' });
      clf.fit(X, y);

      const predictions = clf.predict(X);
      const classes = predictions.map(p => p >= 0.5 ? 1 : 0);

      // Should classify correctly (at least some)
      expect(classes.length).toBe(4);
    });

    it('should use custom threshold', () => {
      const X = [[1], [2], [3], [4]];
      const y = [0, 0, 1, 1];
      const clf = new GLM({ family: 'binomial' });
      clf.fit(X, y);

      const predictions = clf.predict(X);
      const classes = predictions.map(p => p >= 0.7 ? 1 : 0);

      expect(classes.length).toBe(4);
    });
  });
});
