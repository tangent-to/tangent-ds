import { describe, it, expect } from 'vitest';
import { GLM } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('Linear regression with GLM (backward compatibility)', () => {
  describe('fit', () => {
    it('should fit simple linear model', () => {
      // y = 2x + 1
      const X = [[0], [1], [2], [3], [4]];
      const y = [1, 3, 5, 7, 9];

      const model = new GLM({ family: 'gaussian', intercept: true });
      model.fit(X, y);

      // coefficients: [intercept, slope]
      expect(approxEqual(model._model.coefficients[0], 1, 0.001)).toBe(true); // intercept
      expect(approxEqual(model._model.coefficients[1], 2, 0.001)).toBe(true); // slope
      expect(approxEqual(model._model.pseudoR2, 1, 0.001)).toBe(true); // perfect fit
    });

    it('should fit multiple regression', () => {
      // y = 1 + 2*x1 + 3*x2
      const X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2]
      ];
      const y = [6, 8, 9, 11];

      const model = new GLM({ family: 'gaussian', intercept: true });
      model.fit(X, y);

      expect(model._model.coefficients.length).toBe(3);
      expect(approxEqual(model._model.coefficients[0], 1, 0.001)).toBe(true);
      expect(approxEqual(model._model.coefficients[1], 2, 0.001)).toBe(true);
      expect(approxEqual(model._model.coefficients[2], 3, 0.001)).toBe(true);
    });

    it('should compute residuals and fitted values', () => {
      const X = [[1], [2], [3]];
      const y = [2, 4, 6];

      const model = new GLM({ family: 'gaussian', intercept: true });
      model.fit(X, y);

      expect(model._model.fitted.length).toBe(3);
      expect(model._model.residuals.length).toBe(3);

      // Check residuals are small for good fit
      const maxResid = Math.max(...model._model.residuals.map(Math.abs));
      expect(maxResid).toBeLessThan(0.1);
    });
  });

  describe('predict', () => {
    it('should make predictions', () => {
      const X = [[1], [2], [3]];
      const y = [3, 5, 7];

      const lr = new GLM({ family: 'gaussian', intercept: true });
      lr.fit(X, y);

      const Xnew = [[4], [5]];
      const predictions = lr.predict(Xnew);

      expect(predictions.length).toBe(2);
      expect(approxEqual(predictions[0], 9, 0.001)).toBe(true);
      expect(approxEqual(predictions[1], 11, 0.001)).toBe(true);
    });
  });

  describe('summary', () => {
    it('should provide model summary', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 5, 8];

      const lr = new GLM({ family: 'gaussian', intercept: true });
      lr.fit(X, y);
      const summ = lr.summary();

      expect(typeof summ).toBe('string');
      expect(summ).toContain('Coefficients');
      expect(summ).toContain('AIC');
    });
  });
});
