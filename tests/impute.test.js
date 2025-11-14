/**
 * Imputation tests with sklearn validation
 * Compare against sklearn.impute.SimpleImputer and KNNImputer
 */

import { describe, it, expect } from 'vitest';
import { SimpleImputer, KNNImputer, IterativeImputer, simpleImpute, knnImpute, iterativeImpute } from '../src/ml/impute.js';

describe('Imputation (compared with sklearn)', () => {

  describe('SimpleImputer', () => {
    it('should impute with mean strategy', () => {
      // Data with missing values (NaN)
      const X = [
        [1, 2, 3],
        [4, NaN, 6],
        [7, 8, NaN],
        [NaN, 11, 12]
      ];

      // sklearn.impute.SimpleImputer(strategy='mean')
      // Expected: [[1, 2, 3], [4, 7, 6], [7, 8, 7], [4, 11, 12]]
      // Mean of col 0: (1+4+7)/3 = 4
      // Mean of col 1: (2+8+11)/3 = 7
      // Mean of col 2: (3+6+12)/3 = 7

      const imputer = new SimpleImputer({ strategy: 'mean' });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[0]).toEqual([1, 2, 3]);
      expect(result[1]).toEqual([4, 7, 6]);
      expect(result[2]).toEqual([7, 8, 7]);
      expect(result[3]).toEqual([4, 11, 12]);
    });

    it('should impute with median strategy', () => {
      const X = [
        [1, 2],
        [3, NaN],
        [5, 6],
        [7, 8],
        [NaN, 10]
      ];

      // sklearn.impute.SimpleImputer(strategy='median')
      // Median of col 0: 4 (values: [1, 3, 5, 7] -> (3+5)/2 = 4)
      // Median of col 1: 7 (values: [2, 6, 8, 10] -> (6+8)/2 = 7)

      const imputer = new SimpleImputer({ strategy: 'median' });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[1][1]).toBe(7);
      expect(result[4][0]).toBe(4);
    });

    it('should impute with most_frequent strategy', () => {
      const X = [
        [1, 10],
        [1, NaN],
        [2, 20],
        [1, 20],
        [NaN, 20]
      ];

      // Mode of col 0: 1 (appears 3 times)
      // Mode of col 1: 20 (appears 3 times)

      const imputer = new SimpleImputer({ strategy: 'most_frequent' });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[1][1]).toBe(20);
      expect(result[4][0]).toBe(1);
    });

    it('should impute with constant strategy', () => {
      const X = [
        [1, 2],
        [NaN, 4],
        [5, NaN]
      ];

      const imputer = new SimpleImputer({ strategy: 'constant', fill_value: -1 });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[1][0]).toBe(-1);
      expect(result[2][1]).toBe(-1);
    });

    it('should handle null values', () => {
      const X = [
        [1, 2, 3],
        [4, null, 6],
        [7, 8, null]
      ];

      const imputer = new SimpleImputer({ strategy: 'mean' });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[1][1]).toBe(5); // mean of [2, 8]
      expect(result[2][2]).toBeCloseTo(4.5, 5); // mean of [3, 6]
    });

    it('should handle undefined values', () => {
      const X = [
        [1, 2],
        [undefined, 4],
        [5, undefined]
      ];

      const imputer = new SimpleImputer({ strategy: 'mean' });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[1][0]).toBe(3); // mean of [1, 5]
      expect(result[2][1]).toBe(3); // mean of [2, 4]
    });

    it('should work with fit_transform', () => {
      const X = [
        [1, NaN],
        [2, 3],
        [NaN, 4]
      ];

      const imputer = new SimpleImputer({ strategy: 'mean' });
      const result = imputer.fit_transform(X);

      expect(result[0][1]).toBeCloseTo(3.5, 5);
      expect(result[2][0]).toBeCloseTo(1.5, 5);
    });

    it('should throw error for invalid strategy', () => {
      expect(() => new SimpleImputer({ strategy: 'invalid' })).toThrow('Invalid strategy');
    });

    it('should throw error for constant without fill_value', () => {
      expect(() => new SimpleImputer({ strategy: 'constant' })).toThrow('fill_value must be provided');
    });

    it('should throw error if not fitted', () => {
      const imputer = new SimpleImputer();
      expect(() => imputer.transform([[1, 2]])).toThrow('must be fitted');
    });

    it('should handle functional interface', () => {
      const X = [
        [1, NaN],
        [NaN, 3]
      ];

      const result = simpleImpute(X, { strategy: 'constant', fill_value: 0 });

      expect(result[0][1]).toBe(0);
      expect(result[1][0]).toBe(0);
    });
  });

  describe('KNNImputer', () => {
    it('should impute using k-nearest neighbors (k=2)', () => {
      // Simple test case
      const X = [
        [1, 2, 3],
        [4, NaN, 6],
        [7, 8, 9],
        [10, 11, 12]
      ];

      // sklearn.impute.KNNImputer(n_neighbors=2)
      // Row 1 missing col 1
      // Nearest neighbors based on cols [0, 2]:
      // - Row 0: distance ~ sqrt((4-1)^2 + (6-3)^2) = sqrt(18) ~ 4.24
      // - Row 2: distance ~ sqrt((4-7)^2 + (6-9)^2) = sqrt(18) ~ 4.24
      // - Row 3: distance ~ sqrt((4-10)^2 + (6-12)^2) = sqrt(72) ~ 8.49
      // Two nearest: Row 0 (col1=2) and Row 2 (col1=8)
      // Imputed value: mean of [2, 8] = 5

      const imputer = new KNNImputer({ n_neighbors: 2 });
      imputer.fit(X);
      const result = imputer.transform(X);

      expect(result[1][1]).toBe(5);
    });

    it('should handle multiple missing values', () => {
      const X = [
        [1, 1, 1],
        [2, NaN, 2],
        [3, 3, NaN],
        [4, 4, 4]
      ];

      const imputer = new KNNImputer({ n_neighbors: 2 });
      imputer.fit(X);
      const result = imputer.transform(X);

      // Row 1 missing col 1: neighbors likely [1,1,1] and [3,3,NaN]
      // Imputed from [1, 3] = 2
      expect(result[1][1]).toBeCloseTo(2, 0);

      // Row 2 missing col 2: neighbors likely [2,NaN,2] and [4,4,4]
      // Imputed from [2, 4] = 3
      expect(result[2][2]).toBeCloseTo(3, 0);
    });

    it('should handle distance weighting', () => {
      const X = [
        [1, 1],
        [2, NaN],
        [10, 10]
      ];

      const imputer = new KNNImputer({ n_neighbors: 2, weights: 'distance' });
      imputer.fit(X);
      const result = imputer.transform(X);

      // Row 1 missing col 1
      // Neighbor 0 (dist ~ 1) has value 1
      // Neighbor 2 (dist ~ 8) has value 10
      // Weighted: closer to 1 than 10
      expect(result[1][1]).toBeLessThan(5.5);
      expect(result[1][1]).toBeGreaterThan(1);
    });

    it('should work with uniform weights (default)', () => {
      const X = [
        [1, 1],
        [2, NaN],
        [3, 3]
      ];

      const imputer = new KNNImputer({ n_neighbors: 2, weights: 'uniform' });
      imputer.fit(X);
      const result = imputer.transform(X);

      // Simple average of nearest neighbors
      expect(result[1][1]).toBeCloseTo(2, 0);
    });

    it('should handle all missing column gracefully', () => {
      const X = [
        [1, NaN],
        [2, NaN],
        [3, NaN]
      ];

      const imputer = new KNNImputer({ n_neighbors: 2 });
      imputer.fit(X);
      const result = imputer.transform(X);

      // When all neighbors are missing, use 0
      expect(result[0][1]).toBe(0);
      expect(result[1][1]).toBe(0);
      expect(result[2][1]).toBe(0);
    });

    it('should work with fit_transform', () => {
      const X = [
        [1, 1],
        [2, NaN],
        [3, 3]
      ];

      const imputer = new KNNImputer({ n_neighbors: 2 });
      const result = imputer.fit_transform(X);

      expect(result[1][1]).toBeGreaterThan(0);
    });

    it('should throw error if not fitted', () => {
      const imputer = new KNNImputer();
      expect(() => imputer.transform([[1, 2]])).toThrow('must be fitted');
    });

    it('should handle functional interface', () => {
      const X = [
        [1, 1],
        [2, NaN],
        [3, 3]
      ];

      const result = knnImpute(X, { n_neighbors: 2 });

      expect(result[1][1]).toBeGreaterThan(0);
    });

    it('should handle completely missing row', () => {
      const X = [
        [1, 2],
        [3, 4],
        [NaN, NaN],
        [5, 6]
      ];

      const imputer = new KNNImputer({ n_neighbors: 2 });
      imputer.fit(X);
      const result = imputer.transform(X);

      // Row 2 has no non-missing features, so any neighbor works
      // Should impute both values
      expect(result[2][0]).toBeGreaterThan(0);
      expect(result[2][1]).toBeGreaterThan(0);
    });
  });

  describe('Edge cases', () => {
    it('should handle single feature', () => {
      const X = [[1], [NaN], [3]];

      const imputer = new SimpleImputer({ strategy: 'mean' });
      const result = imputer.fit_transform(X);

      expect(result[1][0]).toBe(2);
    });

    it('should handle single sample', () => {
      const X = [[1, NaN, 3]];

      const imputer = new SimpleImputer({ strategy: 'mean' });
      const result = imputer.fit_transform(X);

      // Only one sample, so mean = 0 for missing
      expect(result[0][1]).toBe(0);
    });

    it('should handle no missing values', () => {
      const X = [
        [1, 2, 3],
        [4, 5, 6]
      ];

      const imputer = new SimpleImputer({ strategy: 'mean' });
      const result = imputer.fit_transform(X);

      expect(result).toEqual(X);
    });

    it('should preserve copy setting with separate fit/transform', () => {
      const X_train = [[1, 2], [3, 4]];
      const X_test = [[5, NaN]];

      const imputer = new SimpleImputer({ strategy: 'mean', copy: true });
      imputer.fit(X_train);
      const result = imputer.transform(X_test);

      // With copy=true (default), should create new array
      expect(result).not.toBe(X_test);
      expect(result[0][1]).toBe(3); // mean of [2, 4]
    });
  });

  describe('IterativeImputer', () => {
    it('should impute using multivariate approach (MICE)', () => {
      // Data with correlated features
      const X = [
        [1, 2, 3],
        [4, NaN, 6],
        [7, 8, NaN],
        [NaN, 11, 12],
        [10, 11, 12]
      ];

      const imputer = new IterativeImputer({ max_iter: 5, tol: 1e-2 });
      const result = imputer.fit_transform(X);

      // Should fill all missing values
      for (let i = 0; i < result.length; i++) {
        for (let j = 0; j < result[i].length; j++) {
          expect(result[i][j]).not.toBeNaN();
          expect(result[i][j]).not.toBeNull();
        }
      }

      // Check that known values are preserved
      expect(result[0]).toEqual([1, 2, 3]);
      expect(result[1][0]).toBe(4);
      expect(result[1][2]).toBe(6);
    });

    it('should handle initial_strategy parameter', () => {
      const X = [
        [1, NaN],
        [2, 3],
        [NaN, 4]
      ];

      const imputer = new IterativeImputer({ initial_strategy: 'median', max_iter: 3 });
      const result = imputer.fit_transform(X);

      expect(result[0][1]).toBeGreaterThan(0);
      expect(result[2][0]).toBeGreaterThan(0);
    });

    it('should converge within max_iter iterations', () => {
      const X = [
        [1, 2],
        [2, NaN],
        [3, 4],
        [NaN, 5]
      ];

      const imputer = new IterativeImputer({ max_iter: 10, tol: 1e-4 });
      imputer.fit(X);
      const result = imputer.transform(X);

      // Should have converged (n_iter_ <= max_iter)
      expect(imputer.n_iter_).toBeLessThanOrEqual(10);
      expect(imputer.n_iter_).toBeGreaterThan(0);
    });

    it('should respect min_value and max_value bounds', () => {
      const X = [
        [1, 100],
        [2, NaN],
        [3, 120]
      ];

      const imputer = new IterativeImputer({
        max_iter: 5,
        min_value: 0,
        max_value: 150
      });
      const result = imputer.fit_transform(X);

      // Imputed values should be within bounds
      // Row 1 col 1 was imputed (originally NaN)
      expect(result[1][1]).toBeGreaterThanOrEqual(0);
      expect(result[1][1]).toBeLessThanOrEqual(150);

      // Known values are preserved
      expect(result[0]).toEqual([1, 100]);
      expect(result[2]).toEqual([3, 120]);
    });

    it('should work with fit and transform separately', () => {
      const X_train = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ];

      const X_test = [
        [1, NaN, 3],
        [NaN, 5, 6]
      ];

      const imputer = new IterativeImputer({ max_iter: 5 });
      imputer.fit(X_train);
      const result = imputer.transform(X_test);

      expect(result[0][1]).toBeGreaterThan(0);
      expect(result[1][0]).toBeGreaterThan(0);
    });

    it('should handle correlated features better than simple imputation', () => {
      // Create data where feature 2 = 2 * feature 1
      const X = [];
      for (let i = 1; i <= 10; i++) {
        X.push([i, 2 * i]);
      }
      // Add row with missing value in feature 2
      X.push([5, NaN]);

      const imputer = new IterativeImputer({ max_iter: 10 });
      const result = imputer.fit_transform(X);

      // Should impute approximately 10 (2 * 5), allowing some tolerance
      expect(result[10][1]).toBeGreaterThan(8);
      expect(result[10][1]).toBeLessThan(12);
    });

    it('should throw error if not fitted', () => {
      const imputer = new IterativeImputer();
      expect(() => imputer.transform([[1, 2]])).toThrow('must be fitted');
    });

    it('should work with functional interface', () => {
      const X = [
        [1, NaN],
        [2, 3],
        [NaN, 4]
      ];

      const result = iterativeImpute(X, { max_iter: 5 });

      expect(result[0][1]).toBeGreaterThan(0);
      expect(result[2][0]).toBeGreaterThan(0);
    });

    it('should handle all missing column gracefully', () => {
      const X = [
        [1, NaN],
        [2, NaN],
        [3, NaN]
      ];

      const imputer = new IterativeImputer({ max_iter: 3 });
      const result = imputer.fit_transform(X);

      // Should use initial imputation (mean) for all missing column
      const expectedValue = (X[0][0] + X[1][0] + X[2][0]) / 3; // mean of col 0
      // But column 1 is all NaN, so it should be 0 from initial imputation
      expect(result[0][1]).toBe(0);
      expect(result[1][1]).toBe(0);
      expect(result[2][1]).toBe(0);
    });

    it('should handle single missing value', () => {
      const X = [
        [1, 2, 3],
        [4, 5, NaN],
        [7, 8, 9]
      ];

      const imputer = new IterativeImputer({ max_iter: 5 });
      const result = imputer.fit_transform(X);

      // Should impute based on correlation with other features
      expect(result[1][2]).toBeGreaterThan(0);
      expect(result[1][2]).toBeCloseTo(6, 0.5); // Expect close to 6
    });

    it('should handle multiple missing values in same row', () => {
      const X = [
        [1, 2, 3],
        [NaN, NaN, 6],
        [7, 8, 9]
      ];

      const imputer = new IterativeImputer({ max_iter: 5 });
      const result = imputer.fit_transform(X);

      expect(result[1][0]).toBeGreaterThan(0);
      expect(result[1][1]).toBeGreaterThan(0);
    });

    it('should use pseudoinverse for robust regression', () => {
      // Create data with perfectly correlated features (singular covariance)
      const X = [];
      for (let i = 1; i <= 10; i++) {
        X.push([i, 2 * i, 3 * i]);
      }
      // Add row with missing value
      X.push([5, NaN, 15]);

      const imputer = new IterativeImputer({ max_iter: 10 });

      // Should not throw error despite perfect collinearity
      expect(() => imputer.fit_transform(X)).not.toThrow();

      const result = imputer.fit_transform(X);

      // Should impute approximately 10 (2 * 5), allowing some tolerance
      expect(result[10][1]).toBeGreaterThan(8);
      expect(result[10][1]).toBeLessThan(12);
    });

    it('should improve estimates with more iterations', () => {
      const X = [
        [1, 2, 3],
        [2, NaN, 6],
        [3, 6, 9],
        [4, 8, 12]
      ];

      // Run with 1 iteration
      const imputer1 = new IterativeImputer({ max_iter: 1 });
      const result1 = imputer1.fit_transform(X);

      // Run with 10 iterations
      const imputer10 = new IterativeImputer({ max_iter: 10 });
      const result10 = imputer10.fit_transform(X);

      // Both should fill the value, but they may differ
      expect(result1[1][1]).toBeGreaterThan(0);
      expect(result10[1][1]).toBeGreaterThan(0);

      // With more iterations and linear relationship (y = 2x), should get closer to 4
      expect(Math.abs(result10[1][1] - 4)).toBeLessThan(Math.abs(result1[1][1] - 4) + 0.1);
    });

    it('should handle no missing values', () => {
      const X = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ];

      const imputer = new IterativeImputer({ max_iter: 5 });
      const result = imputer.fit_transform(X);

      // Should return data unchanged
      expect(result).toEqual(X);
    });
  });
});
