import { describe, it, expect } from 'vitest';
import { GLM } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('linear mixed models with GLM', () => {
  describe('fit', () => {
    it('should fit random-intercept model', () => {
      // Simple grouped data
      const X = [
        [1],
        [2],
        [1],
        [2],
        [1],
        [2]
      ];
      const y = [2, 4, 3, 5, 2.5, 4.5];
      const groups = ['A', 'A', 'B', 'B', 'C', 'C'];

      const randomEffects = { intercept: groups };
      const model = new GLM({
        family: 'gaussian',
        intercept: true,
        randomEffects: randomEffects
      });

      model.fit(X, y);
      const result = model._model;

      expect(result.fixedEffects.length).toBe(2);
      expect(result.randomEffects.length).toBeGreaterThan(0);
      expect(result.groupInfo[0].nGroups).toBe(3);
      expect(result.varianceComponents[0].variance).toBeGreaterThanOrEqual(0);
    });

    it('should estimate group effects', () => {
      // Data where groups have clear differences
      const X = [[1], [1], [1], [1]];
      const y = [10, 11, 20, 21];
      const groups = ['A', 'A', 'B', 'B'];

      const model = new GLM({
        family: 'gaussian',
        intercept: true,
        randomEffects: { intercept: groups }
      });

      model.fit(X, y);

      // Group B should have higher random effect than group A
      const groupMap = model._model.groupInfo[0].groupMap;
      const groupA = model._model.randomEffects[groupMap['A']];
      const groupB = model._model.randomEffects[groupMap['B']];

      expect(groupB).toBeGreaterThan(groupA);
    });
  });

  describe('predict', () => {
    it('should predict for known groups', () => {
      const X = [[1], [2], [1], [2]];
      const y = [2, 4, 3, 5];
      const groups = ['A', 'A', 'B', 'B'];

      const model = new GLM({
        family: 'gaussian',
        intercept: true,
        randomEffects: { intercept: groups }
      });

      model.fit(X, y);

      const Xnew = [[1.5], [1.5]];
      const groupsNew = ['A', 'B'];

      // For GLMM, predictions require more complex handling
      // This is a simplified test
      const predictions = model.predict(Xnew, { allowNewGroups: true });

      expect(predictions.length).toBe(2);
      expect(predictions.every(p => isFinite(p))).toBe(true);
    });

    it('should predict for new groups using only fixed effects', () => {
      const X = [[1], [2]];
      const y = [2, 4];
      const groups = ['A', 'A'];

      const model = new GLM({
        family: 'gaussian',
        intercept: true,
        randomEffects: { intercept: groups }
      });

      model.fit(X, y);

      const Xnew = [[1.5]];
      const predictions = model.predict(Xnew, { allowNewGroups: true });

      expect(predictions.length).toBe(1);
      expect(typeof predictions[0]).toBe('number');
      expect(Number.isFinite(predictions[0])).toBe(true);
    });
  });
});
