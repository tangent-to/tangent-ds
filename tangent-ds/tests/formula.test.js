import { describe, it, expect } from 'vitest';
import { parseFormula, applyFormula } from '../src/core/index.js';

describe('Formula Parser', () => {
  describe('parseFormula', () => {
    it('should parse simple formula', () => {
      const formula = 'y ~ x1 + x2';
      const parsed = parseFormula(formula);

      expect(parsed.response).toEqual({
        variable: 'y',
        transform: null
      });

      expect(parsed.fixed).toHaveLength(2);
      expect(parsed.fixed[0]).toEqual({
        type: 'variable',
        name: 'x1'
      });
      expect(parsed.fixed[1]).toEqual({
        type: 'variable',
        name: 'x2'
      });

      expect(parsed.random).toBeNull();
    });

    it('should parse formula with transformation on response', () => {
      const formula = 'log(y) ~ x1';
      const parsed = parseFormula(formula);

      expect(parsed.response).toEqual({
        variable: 'y',
        transform: 'log'
      });
    });

    it('should parse formula with transformed predictors', () => {
      const formula = 'y ~ log(x1) + sqrt(x2)';
      const parsed = parseFormula(formula);

      expect(parsed.fixed[0]).toEqual({
        type: 'transform',
        transform: 'log',
        variable: 'x1'
      });

      expect(parsed.fixed[1]).toEqual({
        type: 'transform',
        transform: 'sqrt',
        variable: 'x2'
      });
    });

    it('should parse formula with I() expression', () => {
      const formula = 'y ~ I(x^2)';
      const parsed = parseFormula(formula);

      expect(parsed.fixed[0].type).toBe('expression');
      expect(parsed.fixed[0].expression).toBe('x^2');
    });

    it('should parse formula with polynomial', () => {
      const formula = 'y ~ poly(x, 3)';
      const parsed = parseFormula(formula);

      expect(parsed.fixed[0]).toEqual({
        type: 'polynomial',
        variable: 'x',
        degree: 3
      });
    });

    it('should parse formula with interaction (*)', () => {
      const formula = 'y ~ x1 * x2';
      const parsed = parseFormula(formula);

      // Should expand to x1 + x2 + x1:x2
      expect(parsed.fixed.length).toBeGreaterThanOrEqual(2);

      // Check for interaction term
      const interaction = parsed.fixed.find(t => t.type === 'interaction');
      expect(interaction).toBeDefined();
    });

    it('should parse formula with random intercept', () => {
      const formula = 'y ~ x1 + (1 | group)';
      const parsed = parseFormula(formula);

      expect(parsed.random).toBeDefined();
      expect(parsed.random.intercept).toBe('group');
    });

    it('should parse formula with random slope', () => {
      const formula = 'y ~ x1 + (1 + time | subject)';
      const parsed = parseFormula(formula);

      expect(parsed.random).toBeDefined();
      expect(parsed.random.intercept).toBe('subject');
      expect(parsed.random.slopes).toHaveProperty('time');
      expect(parsed.random.slopes.time).toBe('subject');
    });

    it('should parse formula with multiple random effects', () => {
      const formula = 'y ~ x1 + (1 | group) + (1 | subject)';
      const parsed = parseFormula(formula);

      expect(parsed.random).toBeDefined();
      // Last one wins for intercept
      expect(parsed.random.intercept).toBe('subject');
    });

    it('should handle whitespace correctly', () => {
      const formula = 'y~x1+x2';
      const parsed = parseFormula(formula);

      expect(parsed.response.variable).toBe('y');
      expect(parsed.fixed).toHaveLength(2);
    });
  });

  describe('applyFormula', () => {
    it('should build design matrix from simple formula', () => {
      const formula = 'y ~ x1 + x2';
      const data = [
        { y: 1, x1: 2, x2: 3 },
        { y: 4, x1: 5, x2: 6 },
        { y: 7, x1: 8, x2: 9 }
      ];

      const result = applyFormula(formula, data);

      expect(result.y).toEqual([1, 4, 7]);
      expect(result.X).toHaveLength(3);
      expect(result.X[0]).toEqual([1, 2, 3]); // Intercept, x1, x2
      expect(result.columnNames).toEqual(['(Intercept)', 'x1', 'x2']);
    });

    it('should apply transformations to predictors', () => {
      const formula = 'y ~ log(x)';
      const data = [
        { y: 1, x: Math.E },
        { y: 2, x: Math.E * Math.E }
      ];

      const result = applyFormula(formula, data);

      expect(result.X[0][1]).toBeCloseTo(1, 5); // log(e) = 1
      expect(result.X[1][1]).toBeCloseTo(2, 5); // log(e^2) = 2
      expect(result.columnNames).toContain('log(x)');
    });

    it('should evaluate I() expressions', () => {
      const formula = 'y ~ I(x^2)';
      const data = [
        { y: 1, x: 2 },
        { y: 4, x: 3 }
      ];

      const result = applyFormula(formula, data);

      expect(result.X[0][1]).toBe(4); // 2^2
      expect(result.X[1][1]).toBe(9); // 3^2
    });

    it('should create polynomial terms', () => {
      const formula = 'y ~ poly(x, 3)';
      const data = [
        { y: 1, x: 2 },
        { y: 4, x: 3 }
      ];

      const result = applyFormula(formula, data);

      // Should have intercept + 3 polynomial terms
      expect(result.X[0].length).toBe(4);
      expect(result.X[0][1]).toBe(2); // x
      expect(result.X[0][2]).toBe(4); // x^2
      expect(result.X[0][3]).toBe(8); // x^3
    });

    it('should compute interaction terms', () => {
      const formula = 'y ~ x1 * x2';
      const data = [
        { y: 1, x1: 2, x2: 3 },
        { y: 4, x1: 4, x2: 5 }
      ];

      const result = applyFormula(formula, data);

      // Should include x1, x2, and x1:x2
      expect(result.X[0].length).toBeGreaterThanOrEqual(4); // intercept + x1 + x2 + x1:x2

      // Find the interaction column
      const interactionIdx = result.columnNames.indexOf('x1:x2');
      expect(interactionIdx).toBeGreaterThan(-1);

      expect(result.X[0][interactionIdx]).toBe(6); // 2 * 3
      expect(result.X[1][interactionIdx]).toBe(20); // 4 * 5
    });

    it('should extract random effects grouping', () => {
      const formula = 'y ~ x + (1 | group)';
      const data = [
        { y: 1, x: 2, group: 'A' },
        { y: 3, x: 4, group: 'B' },
        { y: 5, x: 6, group: 'A' }
      ];

      const result = applyFormula(formula, data);

      expect(result.groups).toEqual(['A', 'B', 'A']);
      expect(result.randomEffects).toBeDefined();
      expect(result.randomEffects.intercept).toEqual(['A', 'B', 'A']);
    });

    it('should handle random slopes', () => {
      const formula = 'y ~ x + (1 + time | subject)';
      const data = [
        { y: 1, x: 2, time: 1, subject: 'S1' },
        { y: 3, x: 4, time: 2, subject: 'S1' },
        { y: 5, x: 6, time: 1, subject: 'S2' }
      ];

      const result = applyFormula(formula, data);

      expect(result.randomEffects).toBeDefined();
      expect(result.randomEffects.intercept).toEqual(['S1', 'S1', 'S2']);
      expect(result.randomEffects.slopes).toHaveProperty('time');
      expect(result.randomEffects.slopes.time.groups).toEqual(['S1', 'S1', 'S2']);
      expect(result.randomEffects.slopes.time.values).toEqual([1, 2, 1]);
    });

    it('should support no intercept option', () => {
      const formula = 'y ~ x';
      const data = [
        { y: 1, x: 2 },
        { y: 3, x: 4 }
      ];

      const result = applyFormula(formula, data, { intercept: false });

      expect(result.X[0]).toEqual([2]); // No intercept
      expect(result.columnNames).toEqual(['x']);
    });

    it('should apply transformation to response', () => {
      const formula = 'log(y) ~ x';
      const data = [
        { y: Math.E, x: 1 },
        { y: Math.E * Math.E, x: 2 }
      ];

      const result = applyFormula(formula, data);

      expect(result.y[0]).toBeCloseTo(1, 5); // log(e)
      expect(result.y[1]).toBeCloseTo(2, 5); // log(e^2)
    });
  });

  describe('Formula integration with GLM', () => {
    it('should work with GLM.fit()', async () => {
      const { GLM } = await import('../src/stats/index.js');

      const data = [
        { y: 2, x1: 1, x2: 1 },
        { y: 4, x1: 2, x2: 1 },
        { y: 5, x1: 3, x2: 2 },
        { y: 7, x1: 4, x2: 2 }
      ];

      const model = new GLM({ family: 'gaussian' });
      model.fit('y ~ x1 + x2', data);

      expect(model.fitted).toBe(true);
      expect(model._model.coefficients).toHaveLength(3); // intercept + x1 + x2
      expect(model._formula).toBe('y ~ x1 + x2');
    });

    it('should work with transformations in GLM', async () => {
      const { GLM } = await import('../src/stats/index.js');

      const data = [
        { y: 1, x: 1 },
        { y: 2.718, x: Math.E },
        { y: 7.389, x: Math.E * Math.E }
      ];

      const model = new GLM({ family: 'gaussian' });
      model.fit('y ~ log(x)', data);

      expect(model.fitted).toBe(true);
    });

    it('should work with mixed effects in GLM', async () => {
      const { GLM } = await import('../src/stats/index.js');

      const data = [
        { y: 1, x: 1, group: 'A' },
        { y: 2, x: 2, group: 'A' },
        { y: 3, x: 1, group: 'B' },
        { y: 4, x: 2, group: 'B' }
      ];

      const model = new GLM({ family: 'gaussian' });
      model.fit('y ~ x + (1 | group)', data);

      expect(model.fitted).toBe(true);
      expect(model._model.randomEffects).toBeDefined();
    });
  });
});
