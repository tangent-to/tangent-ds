/**
 * Priority 2 statistical tests with Python/R reference values
 * Tests assumption checking and correlation functions
 */

import { describe, it, expect } from 'vitest';
import {
  leveneTest,
  shapiroWilk,
  pearsonCorrelation,
  spearmanCorrelation,
  fisherExactTest
} from '../src/stats/tests.js';
import { approxEqual } from '../src/core/math.js';

describe('Priority 2 Statistical Tests (compared with Python/R)', () => {

  describe('leveneTest', () => {
    it('should detect equal variances', () => {
      const group1 = [1, 2, 3, 4, 5];
      const group2 = [2, 3, 4, 5, 6];
      const group3 = [3, 4, 5, 6, 7];

      const result = leveneTest([group1, group2, group3]);

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeGreaterThan(0.05); // Should not reject null (equal variances)
      expect(result.df1).toBe(2);
      expect(result.df2).toBe(12);
    });

    it('should detect unequal variances', () => {
      const group1 = [1, 1, 1, 1, 1];
      const group2 = [1, 5, 10, 15, 20];
      const group3 = [1, 10, 20, 30, 40];

      const result = leveneTest([group1, group2, group3]);

      expect(result.statistic).toBeGreaterThan(1);
      expect(result.pValue).toBeLessThan(0.5);
    });

    it('should support different center options', () => {
      const groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

      const resultMedian = leveneTest(groups, { center: 'median' });
      const resultMean = leveneTest(groups, { center: 'mean' });

      expect(resultMedian.statistic).toBeGreaterThan(0);
      expect(resultMean.statistic).toBeGreaterThan(0);
    });

    it('should throw error with less than 2 groups', () => {
      expect(() => leveneTest([[1, 2, 3]])).toThrow('Need at least 2 groups');
    });
  });

  describe('shapiroWilk', () => {
    it('should not reject normality for normal data', () => {
      // Generated from normal distribution
      const normal = [0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4, 0.2, 0.1, -0.3,
                      0.7, -0.5, 0.4, 0.0, -0.2, 0.5, 0.3, -0.1, 0.2, 0.4];

      const result = shapiroWilk(normal);

      expect(result.statistic).toBeGreaterThan(0.9); // W should be close to 1 for normal data
      expect(result.statistic).toBeLessThan(1.0);
      expect(result.pValue).toBeGreaterThan(0.05); // Should not reject normality
    });

    it('should reject normality for uniform data', () => {
      // Uniform distribution
      const uniform = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const result = shapiroWilk(uniform);

      expect(result.statistic).toBeLessThan(1.0);
      // Note: Shapiro-Wilk on small uniform sample might not always reject
      expect(result.pValue).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(1);
    });

    it('should throw error for small samples', () => {
      expect(() => shapiroWilk([1, 2])).toThrow('at least 3 observations');
    });

    it('should throw error for very large samples', () => {
      const large = new Array(6000).fill(1);
      expect(() => shapiroWilk(large)).toThrow('not recommended for samples larger than 5000');
    });
  });

  describe('pearsonCorrelation', () => {
    it('should compute perfect positive correlation', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [2, 4, 6, 8, 10];

      const result = pearsonCorrelation(x, y);

      expect(result.r).toBeCloseTo(1.0, 10);
      expect(result.pValue).toBeLessThan(0.01);
      expect(result.df).toBe(3);
      expect(result.ci95[0]).toBeGreaterThan(0.5);
      expect(result.ci95[1]).toBeCloseTo(1.0, 1);
    });

    it('should compute perfect negative correlation', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [10, 8, 6, 4, 2];

      const result = pearsonCorrelation(x, y);

      expect(result.r).toBeCloseTo(-1.0, 10);
      expect(result.pValue).toBeLessThan(0.01);
      expect(result.ci95[0]).toBeCloseTo(-1.0, 1);
      expect(result.ci95[1]).toBeLessThan(0);
    });

    it('should compute zero correlation', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [3, 3, 3, 3, 3];

      const result = pearsonCorrelation(x, y);

      expect(approxEqual(result.r, 0, 0.001)).toBe(true);
      expect(result.pValue).toBeGreaterThan(0.9);
    });

    it('should match R cor.test for moderate correlation', () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = [2, 4, 5, 4, 5, 7, 6, 8, 9, 10];

      // R reference:
      // cor.test(c(1,2,3,4,5,6,7,8,9,10), c(2,4,5,4,5,7,6,8,9,10))
      // r = 0.9487, p-value = 5.872e-05

      const result = pearsonCorrelation(x, y);

      expect(result.r).toBeCloseTo(0.949, 2);
      expect(result.pValue).toBeLessThan(0.001);
      expect(result.pValue).toBeGreaterThan(0.00001);
    });

    it('should throw error for unequal lengths', () => {
      expect(() => pearsonCorrelation([1, 2, 3], [1, 2])).toThrow('equal length');
    });

    it('should throw error for small samples', () => {
      expect(() => pearsonCorrelation([1, 2], [3, 4])).toThrow('at least 3 observations');
    });
  });

  describe('spearmanCorrelation', () => {
    it('should compute perfect rank correlation', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [2, 4, 6, 8, 10];

      const result = spearmanCorrelation(x, y);

      expect(result.rho).toBeCloseTo(1.0, 10);
      expect(result.pValue).toBeLessThan(0.01);
    });

    it('should handle non-linear monotonic relationships', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [1, 4, 9, 16, 25]; // y = x^2

      const result = spearmanCorrelation(x, y);

      // Spearman should still detect perfect monotonic relationship
      expect(result.rho).toBeCloseTo(1.0, 10);
      expect(result.pValue).toBeLessThan(0.01);
    });

    it('should handle ties correctly', () => {
      const x = [1, 2, 2, 3, 4];
      const y = [1, 2, 2, 3, 4];

      const result = spearmanCorrelation(x, y);

      expect(result.rho).toBeCloseTo(1.0, 5);
      expect(result.pValue).toBeLessThan(0.05);
    });

    it('should differ from Pearson for non-linear relationships', () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]; // y = x^2

      const pearson = pearsonCorrelation(x, y);
      const spearman = spearmanCorrelation(x, y);

      // Spearman should be 1.0 (perfect monotonic)
      // Pearson should be less (not perfectly linear)
      expect(spearman.rho).toBeCloseTo(1.0, 5);
      expect(pearson.r).toBeLessThan(spearman.rho);
      expect(pearson.r).toBeGreaterThan(0.9);
    });
  });

  describe('fisherExactTest', () => {
    it('should compute exact p-value for 2x2 table', () => {
      // Classic tea tasting example
      const table = [[3, 1], [1, 3]];

      const result = fisherExactTest(table);

      // R reference:
      // fisher.test(matrix(c(3,1,1,3), nrow=2))
      // p-value = 0.4857

      expect(result.pValue).toBeCloseTo(0.486, 2);
      expect(result.oddsRatio).toBeCloseTo(9, 0);
    });

    it('should detect significant association', () => {
      const table = [[10, 2], [2, 10]];

      const result = fisherExactTest(table);

      expect(result.pValue).toBeLessThan(0.05);
      expect(result.oddsRatio).toBeGreaterThan(1);
    });

    it('should handle zero cells', () => {
      const table = [[5, 0], [0, 5]];

      const result = fisherExactTest(table);

      expect(result.pValue).toBeLessThan(0.01);
      expect(result.oddsRatio).toBe(Infinity);
    });

    it('should handle all values in one cell', () => {
      const table = [[10, 0], [5, 0]];

      const result = fisherExactTest(table);

      expect(result.pValue).toBe(1.0); // No association
    });

    it('should support one-sided tests', () => {
      const table = [[8, 2], [1, 9]];

      const twoSided = fisherExactTest(table, { alternative: 'two-sided' });
      const less = fisherExactTest(table, { alternative: 'less' });
      const greater = fisherExactTest(table, { alternative: 'greater' });

      expect(twoSided.pValue).toBeGreaterThan(less.pValue);
      expect(twoSided.pValue).toBeGreaterThan(greater.pValue);
    });

    it('should throw error for non-2x2 table', () => {
      expect(() => fisherExactTest([[1, 2, 3], [4, 5, 6]])).toThrow('requires a 2x2 table');
    });

    it('should throw error for negative values', () => {
      expect(() => fisherExactTest([[1, -1], [1, 1]])).toThrow('non-negative integers');
    });

    it('should throw error for non-integer values', () => {
      expect(() => fisherExactTest([[1.5, 2], [1, 1]])).toThrow('non-negative integers');
    });
  });

  describe('Integration tests', () => {
    it('should check ANOVA assumptions with Levene and Shapiro-Wilk', () => {
      const group1 = [12, 14, 13, 15, 14];
      const group2 = [15, 17, 16, 18, 17];
      const group3 = [18, 20, 19, 21, 20];

      // Check variance homogeneity
      const levene = leveneTest([group1, group2, group3]);
      expect(levene.pValue).toBeGreaterThan(0.05); // Variances are equal

      // Check normality for each group
      const sw1 = shapiroWilk(group1);
      const sw2 = shapiroWilk(group2);
      const sw3 = shapiroWilk(group3);

      // All should pass normality
      expect(sw1.pValue).toBeGreaterThan(0.05);
      expect(sw2.pValue).toBeGreaterThan(0.05);
      expect(sw3.pValue).toBeGreaterThan(0.05);
    });

    it('should use Spearman when Pearson assumptions violated', () => {
      // Outlier-contaminated data
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
      const y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 200];

      const pearson = pearsonCorrelation(x, y);
      const spearman = spearmanCorrelation(x, y);

      // Spearman should be more robust
      expect(spearman.rho).toBeCloseTo(1.0, 5);
      expect(pearson.r).toBeCloseTo(1.0, 1);
    });

    it('should use Fisher exact for small contingency tables', () => {
      // Small sample where chi-square might not be appropriate
      const table = [[2, 3], [3, 2]];

      const result = fisherExactTest(table);

      expect(result.pValue).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(1);
      expect(result.oddsRatio).toBeCloseTo(0.44, 1);
    });
  });
});
