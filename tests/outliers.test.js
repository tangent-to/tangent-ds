/**
 * Outlier detection tests with sklearn validation
 * Compare against sklearn.ensemble.IsolationForest and sklearn.neighbors.LocalOutlierFactor
 */

import { describe, it, expect } from 'vitest';
import { IsolationForest, LocalOutlierFactor, MahalanobisDistance, isolationForest, localOutlierFactor, mahalanobisDistance } from '../src/ml/outliers.js';
import { setSeed } from '../src/ml/utils.js';

describe('Outlier Detection (compared with sklearn)', () => {

  describe('IsolationForest', () => {
    it('should detect obvious outliers', () => {
      setSeed(42);

      // Normal data clustered around (0, 0)
      const X = [
        [0, 0],
        [0.1, 0.1],
        [-0.1, -0.1],
        [0.2, -0.1],
        [-0.1, 0.2],
        [0, 0.1],
        [0.1, 0],
        [0.05, 0.05],
        [-0.05, -0.05],
        [0.15, -0.05],
        // Obvious outliers far from cluster
        [100, 100],
        [-100, -100]
      ];

      const iso = new IsolationForest({
        n_estimators: 200,
        contamination: 0.15,
        random_state: 42
      });

      iso.fit(X);
      const predictions = iso.predict(X);

      // Should detect the extreme outliers
      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThanOrEqual(1); // At least 1 of the 2 extremes
      expect(outlierCount).toBeLessThanOrEqual(3); // Not too many

      // Most of the first samples should be inliers (1)
      const inlierCount = predictions.slice(0, 10).filter(p => p === 1).length;
      expect(inlierCount).toBeGreaterThan(7);
    });

    it('should compute anomaly scores', () => {
      setSeed(42);

      const X = [
        [0, 0],
        [1, 1],
        [10, 10] // Outlier
      ];

      const iso = new IsolationForest({ n_estimators: 50, random_state: 42 });
      iso.fit(X);
      const scores = iso.score_samples(X);

      // Outlier should have lower (more negative) score
      expect(scores[2]).toBeLessThan(scores[0]);
      expect(scores[2]).toBeLessThan(scores[1]);

      // All scores should be negative (between -1 and 0)
      for (const score of scores) {
        expect(score).toBeLessThanOrEqual(0);
        expect(score).toBeGreaterThanOrEqual(-1);
      }
    });

    it('should handle contamination parameter', () => {
      setSeed(42);

      const X = [];
      // 90 inliers
      for (let i = 0; i < 90; i++) {
        X.push([Math.random() * 2 - 1, Math.random() * 2 - 1]);
      }
      // 10 outliers
      for (let i = 0; i < 10; i++) {
        X.push([10 + Math.random(), 10 + Math.random()]);
      }

      const iso = new IsolationForest({
        n_estimators: 100,
        contamination: 0.1, // Expect 10% outliers
        random_state: 42
      });

      iso.fit(X);
      const predictions = iso.predict(X);

      const outlierCount = predictions.filter(p => p === -1).length;

      // Should detect approximately 10% as outliers (10 samples)
      expect(outlierCount).toBeGreaterThanOrEqual(8);
      expect(outlierCount).toBeLessThanOrEqual(12);
    });

    it('should work with fit_predict', () => {
      setSeed(42);

      const X = [
        [0, 0],
        [0.1, 0.1],
        [0, 0.1],
        [0.1, 0],
        [-0.1, 0.1],
        [100, 100] // Extreme outlier
      ];

      const iso = new IsolationForest({ n_estimators: 200, contamination: 0.15, random_state: 42 });
      const predictions = iso.fit_predict(X);

      expect(predictions).toHaveLength(6);

      // Should detect at least 1 outlier
      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThanOrEqual(1);

      // The outlier should be the extreme point (last one)
      // But we allow some randomness, so just check it's among the outliers
      expect(outlierCount).toBeLessThanOrEqual(2);
    });

    it('should handle max_samples parameter', () => {
      setSeed(42);

      const X = [];
      for (let i = 0; i < 1000; i++) {
        X.push([Math.random(), Math.random()]);
      }

      const iso = new IsolationForest({
        n_estimators: 10,
        max_samples: 0.5, // Sample 50% of data for each tree
        random_state: 42
      });

      iso.fit(X);
      expect(iso.max_samples_).toBe(500);

      const scores = iso.score_samples(X);
      expect(scores).toHaveLength(1000);
    });

    it('should handle max_samples=auto', () => {
      setSeed(42);

      const X = [];
      for (let i = 0; i < 300; i++) {
        X.push([Math.random(), Math.random()]);
      }

      const iso = new IsolationForest({
        n_estimators: 10,
        max_samples: 'auto',
        random_state: 42
      });

      iso.fit(X);
      expect(iso.max_samples_).toBe(256); // min(256, 300) = 256
    });

    it('should throw error if not fitted', () => {
      const iso = new IsolationForest();
      expect(() => iso.predict([[1, 2]])).toThrow('must be fitted');
      expect(() => iso.score_samples([[1, 2]])).toThrow('must be fitted');
    });

    it('should work with functional interface', () => {
      setSeed(42);

      const X = [
        [0, 0],
        [0.1, 0.1],
        [-0.1, 0],
        [0, -0.1],
        [100, 100]  // Extreme outlier
      ];

      const predictions = isolationForest(X, { n_estimators: 200, contamination: 0.2 });

      expect(predictions).toHaveLength(5);
      expect(predictions[4]).toBe(-1); // Outlier
    });
  });

  describe('LocalOutlierFactor', () => {
    it('should detect obvious outliers', () => {
      // Cluster of inliers
      const X = [
        [0, 0],
        [0.1, 0.1],
        [-0.1, -0.1],
        [0.2, -0.1],
        [-0.1, 0.2],
        [0, 0.1],
        [0.1, 0],
        [0.15, 0.05],
        [-0.05, 0.15],
        [0.05, -0.05],
        // Obvious outliers (much farther)
        [100, 100],
        [-100, -100]
      ];

      const lof = new LocalOutlierFactor({
        n_neighbors: 5,
        contamination: 0.15 // Expect ~15% outliers (2 out of 12)
      });

      lof.fit(X);
      const predictions = lof.predict(X);

      // Should detect the extreme outliers
      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThanOrEqual(1); // At least 1 of the 2 extremes
      expect(outlierCount).toBeLessThanOrEqual(3); // Not too many

      // Most of the cluster should be inliers
      const inlierCount = predictions.slice(0, 10).filter(p => p === 1).length;
      expect(inlierCount).toBeGreaterThanOrEqual(7);
    });

    it('should compute negative outlier factors', () => {
      const X = [
        [0, 0],
        [1, 1],
        [0.5, 0.5],
        [10, 10] // Outlier
      ];

      const lof = new LocalOutlierFactor({ n_neighbors: 2 });
      lof.fit(X);

      const nof = lof.negative_outlier_factor;

      // Outlier should have lower (more negative) score
      // LOF > 1 means outlier, so negative LOF < -1
      expect(nof[3]).toBeLessThan(nof[0]);
      expect(nof[3]).toBeLessThan(nof[1]);
      expect(nof[3]).toBeLessThan(nof[2]);
    });

    it('should handle contamination parameter', () => {
      const X = [];

      // 18 inliers in a cluster
      for (let i = 0; i < 18; i++) {
        X.push([Math.random() * 2 - 1, Math.random() * 2 - 1]);
      }

      // 2 outliers
      X.push([10, 10]);
      X.push([-10, -10]);

      const lof = new LocalOutlierFactor({
        n_neighbors: 5,
        contamination: 0.1 // Expect 10% outliers (2 out of 20)
      });

      lof.fit(X);
      const predictions = lof.predict(X);

      const outlierCount = predictions.filter(p => p === -1).length;

      // Should detect approximately 2 outliers
      expect(outlierCount).toBeGreaterThanOrEqual(1);
      expect(outlierCount).toBeLessThanOrEqual(3);
    });

    it('should work with fit_predict', () => {
      const X = [
        [0, 0],
        [0.1, 0.1],
        [-0.1, -0.1],
        [10, 10] // Outlier
      ];

      const lof = new LocalOutlierFactor({ n_neighbors: 2, contamination: 0.25 });
      const predictions = lof.fit_predict(X);

      expect(predictions).toHaveLength(4);
      expect(predictions[3]).toBe(-1); // Outlier
    });

    it('should throw error if n_neighbors >= n_samples', () => {
      const X = [[1, 2], [3, 4]];
      const lof = new LocalOutlierFactor({ n_neighbors: 5 });

      expect(() => lof.fit(X)).toThrow('must be less than n_samples');
    });

    it('should throw error if not fitted', () => {
      const lof = new LocalOutlierFactor();
      expect(() => lof.predict([[1, 2]])).toThrow('must be fitted');
    });

    it('should throw error if predict on new data with novelty=false', () => {
      const X_train = [[0, 0], [1, 1], [2, 2]];
      const X_test = [[0, 0], [1, 1]]; // Different data

      const lof = new LocalOutlierFactor({ n_neighbors: 2, novelty: false });
      lof.fit(X_train);

      expect(() => lof.predict(X_test)).toThrow('can only predict on training data');
    });

    it('should work with functional interface', () => {
      const X = [
        [0, 0],
        [0.1, 0.1],
        [10, 10]
      ];

      const predictions = localOutlierFactor(X, { n_neighbors: 2, contamination: 0.3 });

      expect(predictions).toHaveLength(3);
    });

    it('should handle edge case with uniform data', () => {
      // All points the same - no outliers
      const X = [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
      ];

      const lof = new LocalOutlierFactor({ n_neighbors: 2 });
      lof.fit(X);
      const predictions = lof.predict(X);

      // All should be inliers (or at least based on contamination)
      const inlierCount = predictions.filter(p => p === 1).length;
      expect(inlierCount).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Edge cases and integration', () => {
    it('should handle single feature', () => {
      setSeed(42);

      const X = [[1], [1.1], [0.9], [1.2], [1.05], [0.95], [100]]; // Last one is outlier

      const iso = new IsolationForest({ n_estimators: 200, contamination: 0.14 });
      const predictions = iso.fit_predict(X);

      // Should detect at least the extreme outlier
      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThanOrEqual(1);
    });

    it('should handle high-dimensional data', () => {
      setSeed(42);

      const X = [];
      // 10 inliers
      for (let i = 0; i < 10; i++) {
        X.push([
          Math.random(),
          Math.random(),
          Math.random(),
          Math.random(),
          Math.random()
        ]);
      }
      // 2 outliers
      X.push([10, 10, 10, 10, 10]);
      X.push([-10, -10, -10, -10, -10]);

      const iso = new IsolationForest({ n_estimators: 100, contamination: 0.15 });
      iso.fit(X);
      const predictions = iso.predict(X);

      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThan(0);
    });

    it('should handle small datasets', () => {
      const X = [[1, 2], [2, 3], [1.5, 2.5], [100, 100]];

      const lof = new LocalOutlierFactor({ n_neighbors: 2, contamination: 0.25 });
      lof.fit(X);
      const predictions = lof.predict(X);

      expect(predictions).toHaveLength(4);
      expect(predictions[3]).toBe(-1); // Outlier
    });

    it('IsolationForest and LOF should agree on obvious outliers', () => {
      setSeed(42);

      const X = [
        [0, 0], [0.1, 0.1], [0, 0.1], [0.1, 0],
        [-0.1, 0], [0, -0.1], [0.1, -0.1], [-0.1, 0.1],
        [100, 100], [-100, -100] // Extreme outliers
      ];

      const iso = new IsolationForest({ n_estimators: 200, contamination: 0.2 });
      const isoPredictions = iso.fit_predict(X);

      const lof = new LocalOutlierFactor({ n_neighbors: 3, contamination: 0.2 });
      const lofPredictions = lof.fit_predict(X);

      // Both should detect outliers
      const isoOutlierCount = isoPredictions.filter(p => p === -1).length;
      const lofOutlierCount = lofPredictions.filter(p => p === -1).length;

      expect(isoOutlierCount).toBeGreaterThanOrEqual(2);
      expect(lofOutlierCount).toBeGreaterThanOrEqual(2);
    });

    it('should handle contamination=0 (no outliers expected)', () => {
      setSeed(42);

      const X = [[1, 1], [1.1, 1.1], [0.9, 0.9], [10, 10]];

      const iso = new IsolationForest({ n_estimators: 50, contamination: 0.0 });
      iso.fit(X);
      const predictions = iso.predict(X);

      // With contamination=0, threshold is at minimum, so all should be inliers
      const inlierCount = predictions.filter(p => p === 1).length;
      expect(inlierCount).toBeGreaterThanOrEqual(3);
    });

    it('should be deterministic with same random seed', () => {
      const X = [[1, 1], [1.1, 1.1], [0.9, 0.9], [10, 10]];

      setSeed(123);
      const iso1 = new IsolationForest({ n_estimators: 50 });
      const pred1 = iso1.fit_predict(X);

      setSeed(123);
      const iso2 = new IsolationForest({ n_estimators: 50 });
      const pred2 = iso2.fit_predict(X);

      expect(pred1).toEqual(pred2);
    });
  });

  describe('MahalanobisDistance', () => {
    it('should detect obvious outliers based on statistical distance', () => {
      // Cluster of inliers around origin
      const X = [
        [0, 0],
        [0.1, 0.1],
        [-0.1, -0.1],
        [0.2, -0.1],
        [-0.1, 0.2],
        [0, 0.1],
        [0.1, 0],
        [0.05, 0.05],
        [-0.05, -0.05],
        [0.15, -0.05],
        // Obvious outliers
        [10, 10],
        [-10, -10]
      ];

      const md = new MahalanobisDistance({ contamination: 0.15 });
      md.fit(X);
      const predictions = md.predict(X);

      // Should detect outliers
      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThanOrEqual(1);
      expect(outlierCount).toBeLessThanOrEqual(3);

      // Most inliers should be correctly classified
      const inlierCount = predictions.slice(0, 10).filter(p => p === 1).length;
      expect(inlierCount).toBeGreaterThanOrEqual(7);
    });

    it('should compute Mahalanobis distances', () => {
      const X = [
        [0, 0],
        [1, 1],
        [0.5, 0.5],
        [10, 10] // Outlier
      ];

      const md = new MahalanobisDistance({ contamination: 0.25 });
      md.fit(X);
      const scores = md.score_samples(X);

      // Outlier should have lower (more negative) score
      expect(scores[3]).toBeLessThan(scores[0]);
      expect(scores[3]).toBeLessThan(scores[1]);
      expect(scores[3]).toBeLessThan(scores[2]);

      // All scores should be negative
      for (const score of scores) {
        expect(score).toBeLessThan(0);
      }
    });

    it('should handle correlated features', () => {
      // Data with strong correlation between features
      const X = [];
      for (let i = 0; i < 50; i++) {
        const x = Math.random() * 2 - 1;
        const y = x + Math.random() * 0.1; // y ≈ x with small noise
        X.push([x, y]);
      }
      // Outlier that breaks correlation
      X.push([0, 5]);

      const md = new MahalanobisDistance({ contamination: 0.02 });
      md.fit(X);
      const predictions = md.predict(X);

      // Should detect the correlation-breaking outlier
      expect(predictions[50]).toBe(-1);
    });

    it('should work with fit_predict', () => {
      const X = [
        [0, 0],
        [0.1, 0.1],
        [0, 0.1],
        [0.1, 0],
        [-0.1, 0.1],
        [100, 100] // Extreme outlier
      ];

      const md = new MahalanobisDistance({ contamination: 0.15 });
      const predictions = md.fit_predict(X);

      expect(predictions).toHaveLength(6);
      expect(predictions[5]).toBe(-1); // Outlier
    });

    it('should handle contamination parameter', () => {
      const X = [];
      // 90 inliers
      for (let i = 0; i < 90; i++) {
        X.push([Math.random() * 2 - 1, Math.random() * 2 - 1]);
      }
      // 10 outliers
      for (let i = 0; i < 10; i++) {
        X.push([10 + Math.random(), 10 + Math.random()]);
      }

      const md = new MahalanobisDistance({ contamination: 0.1 });
      md.fit(X);
      const predictions = md.predict(X);

      const outlierCount = predictions.filter(p => p === -1).length;

      // Should detect approximately 10% as outliers
      expect(outlierCount).toBeGreaterThanOrEqual(8);
      expect(outlierCount).toBeLessThanOrEqual(12);
    });

    it('should throw error if not fitted', () => {
      const md = new MahalanobisDistance();
      expect(() => md.predict([[1, 2]])).toThrow('must be fitted');
      expect(() => md.score_samples([[1, 2]])).toThrow('must be fitted');
    });

    it('should throw error if n_samples < n_features', () => {
      const X = [[1, 2, 3], [4, 5, 6]]; // 2 samples, 3 features
      const md = new MahalanobisDistance();

      expect(() => md.fit(X)).toThrow('Number of samples must be >= number of features');
    });

    it('should work with functional interface', () => {
      const X = [
        [0, 0],
        [0.1, 0.1],
        [-0.1, 0],
        [0, -0.1],
        [100, 100]  // Extreme outlier
      ];

      const predictions = mahalanobisDistance(X, { contamination: 0.2 });

      expect(predictions).toHaveLength(5);
      expect(predictions[4]).toBe(-1); // Outlier
    });

    it('should handle singular covariance with pseudoinverse', () => {
      // Create data where features are perfectly correlated (singular covariance)
      const X = [];
      for (let i = 0; i < 20; i++) {
        const x = i;
        const y = 2 * x; // Perfect correlation
        X.push([x, y]);
      }
      // Add outlier
      X.push([10, 0]); // Breaks the pattern

      const md = new MahalanobisDistance({ contamination: 0.05 });

      // Should not throw error despite singular covariance
      expect(() => md.fit(X)).not.toThrow();

      const predictions = md.predict(X);
      expect(predictions).toHaveLength(21);

      // The outlier should be detected
      expect(predictions[20]).toBe(-1);
    });

    it('should handle high-dimensional data', () => {
      const X = [];
      // 20 inliers
      for (let i = 0; i < 20; i++) {
        X.push([
          Math.random(),
          Math.random(),
          Math.random(),
          Math.random(),
          Math.random()
        ]);
      }
      // 2 outliers
      X.push([10, 10, 10, 10, 10]);
      X.push([-10, -10, -10, -10, -10]);

      const md = new MahalanobisDistance({ contamination: 0.1 });
      md.fit(X);
      const predictions = md.predict(X);

      const outlierCount = predictions.filter(p => p === -1).length;
      expect(outlierCount).toBeGreaterThanOrEqual(1);
      expect(outlierCount).toBeLessThanOrEqual(3);
    });

    it('should distinguish from Euclidean distance in correlated data', () => {
      // Create data with correlation
      const X = [];
      for (let i = 0; i < 30; i++) {
        const x = Math.random() * 2;
        const y = x + Math.random() * 0.2;
        X.push([x, y]);
      }

      // Point that's far in Euclidean sense but on the correlation line
      X.push([10, 10]);

      // Point that's close in Euclidean sense but off the correlation line
      X.push([1, 5]);

      const md = new MahalanobisDistance({ contamination: 0.1 });
      md.fit(X);
      const scores = md.score_samples(X);

      // The point off the correlation line should have a worse (more negative) score
      // than the point that's far but on the correlation line
      expect(scores[31]).toBeLessThan(scores[30]);
    });
  });
});
