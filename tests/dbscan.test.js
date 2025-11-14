import { describe, it, expect } from 'vitest';
import { fit as dbfit, predict as dbpredict } from '../src/ml/dbscan.js';
import { DBSCAN } from '../src/ml/index.js';

describe('DBSCAN clustering', () => {
  describe('functional API', () => {
    it('should cluster simple 2D data with two clusters', () => {
      // Create two well-separated dense clusters
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],      // Cluster 1
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5] // Cluster 2
      ];

      const model = dbfit(data, { eps: 1.0, minSamples: 3 });

      expect(model.labels.length).toBe(8);
      expect(model.nClusters).toBe(2);
      expect(model.nNoise).toBe(0);
      expect(model.coreSampleIndices.length).toBeGreaterThan(0);
    });

    it('should identify noise points', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5],       // Dense cluster
        [10, 10], [10.5, 10], [10, 10.5], // Dense cluster
        [5, 5]                             // Isolated noise point
      ];

      const model = dbfit(data, { eps: 1.0, minSamples: 3 });

      expect(model.nClusters).toBe(2);
      expect(model.nNoise).toBe(1);
      // Noise points have label -1
      expect(model.labels).toContain(-1);
    });

    it('should handle single cluster', () => {
      const data = [
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5], [0.25, 0.25], [0.75, 0.75]
      ];

      const model = dbfit(data, { eps: 1.5, minSamples: 3 });

      expect(model.nClusters).toBe(1);
      expect(model.nNoise).toBe(0);
    });

    it('should detect all points as noise with high eps', () => {
      const data = [
        [0, 0], [10, 10], [20, 20], [30, 30]
      ];

      const model = dbfit(data, { eps: 1.0, minSamples: 3 });

      expect(model.nClusters).toBe(0);
      expect(model.nNoise).toBe(4);
    });

    it('should handle 1D data', () => {
      const data = [
        [1], [2], [3], [10], [11], [12]
      ];

      const model = dbfit(data, { eps: 1.5, minSamples: 2 });

      expect(model.labels.length).toBe(6);
      expect(model.nClusters).toBe(2);
    });

    it('should find arbitrarily shaped clusters', () => {
      // Create a circular cluster and linear cluster
      const data = [
        // Circular cluster
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [0.7, 0.7], [-0.7, 0.7], [-0.7, -0.7], [0.7, -0.7],
        // Linear cluster
        [10, 0], [11, 0], [12, 0], [13, 0], [14, 0]
      ];

      const model = dbfit(data, { eps: 1.5, minSamples: 3 });

      expect(model.nClusters).toBe(2);
    });

    it('should predict cluster labels for new points', () => {
      const trainData = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5]
      ];

      const model = dbfit(trainData, { eps: 1.0, minSamples: 3 });

      const newData = [
        [0.25, 0.25],  // Near cluster 1
        [10.25, 10.25], // Near cluster 2
        [5, 5]          // Far from both (should be noise)
      ];

      const labels = dbpredict(model, newData, trainData, 1.0);

      expect(labels.length).toBe(3);
      // First two should be assigned to clusters
      expect(labels[0]).toBeGreaterThanOrEqual(1);
      expect(labels[1]).toBeGreaterThanOrEqual(1);
      // Third should be noise
      expect(labels[2]).toBe(-1);
    });

    it('should handle varying density with appropriate minSamples', () => {
      const data = [
        // Dense cluster (5 points)
        [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.05],
        // Sparse cluster (3 points)
        [10, 10], [11, 10], [10, 11]
      ];

      // With minSamples=3, both should be detected
      const model1 = dbfit(data, { eps: 1.0, minSamples: 3 });
      expect(model1.nClusters).toBeGreaterThan(0);

      // With minSamples=5, only dense cluster
      const model2 = dbfit(data, { eps: 0.5, minSamples: 5 });
      expect(model2.nClusters).toBeGreaterThanOrEqual(1);
    });
  });

  describe('class-based API (DBSCAN)', () => {
    it('should fit and predict with class interface', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
        [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5]
      ];

      const dbscan = new DBSCAN({ eps: 1.0, minSamples: 3 });
      dbscan.fit(data);

      expect(dbscan.fitted).toBe(true);
      expect(dbscan.labels.length).toBe(8);
      expect(dbscan.nClusters).toBe(2);

      const newData = [[0.25, 0.25]];
      const labels = dbscan.predict(newData);
      expect(labels.length).toBe(1);
    });

    it('should provide core sample information', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5],
        [10, 10], [10.5, 10], [10, 10.5]
      ];

      const dbscan = new DBSCAN({ eps: 1.0, minSamples: 2 });
      dbscan.fit(data);

      expect(dbscan.coreSampleMask.length).toBe(6);
      expect(dbscan.components.length).toBeGreaterThan(0);
      expect(dbscan.coreSampleMask.filter(x => x).length).toBe(dbscan.components.length);
    });

    it('should provide summary statistics', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
        [10, 10],  // Noise point
        [20, 20], [20.5, 20], [20, 20.5], [20.5, 20.5]
      ];

      const dbscan = new DBSCAN({ eps: 1.0, minSamples: 3 });
      dbscan.fit(data);

      const summary = dbscan.summary();

      expect(summary.eps).toBe(1.0);
      expect(summary.minSamples).toBe(3);
      expect(summary.nClusters).toBe(2);
      expect(summary.nNoise).toBe(1);
      expect(summary.nSamples).toBe(9);
      expect(summary.noiseRatio).toBeCloseTo(1 / 9);
      expect(summary.coreRatio).toBeGreaterThan(0);
    });

    it('should throw error when predicting before fitting', () => {
      const dbscan = new DBSCAN();
      expect(() => dbscan.predict([[1, 2]])).toThrow();
    });

    it('should handle declarative table-style input', () => {
      const data = [
        { x: 0, y: 0 },
        { x: 0.5, y: 0 },
        { x: 0, y: 0.5 },
        { x: 10, y: 10 },
        { x: 10.5, y: 10 },
        { x: 10, y: 10.5 }
      ];

      const dbscan = new DBSCAN({ eps: 1.0, minSamples: 2 });
      dbscan.fit({ data, columns: ['x', 'y'] });

      expect(dbscan.fitted).toBe(true);
      expect(dbscan.labels.length).toBe(6);
      expect(dbscan.nClusters).toBe(2);
    });

    it('should serialize and deserialize', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5],
        [10, 10], [10.5, 10], [10, 10.5]
      ];

      const dbscan1 = new DBSCAN({ eps: 1.0, minSamples: 2 });
      dbscan1.fit(data);

      const json = dbscan1.toJSON();
      const dbscan2 = DBSCAN.fromJSON(json);

      expect(dbscan2.fitted).toBe(true);
      expect(dbscan2.labels).toEqual(dbscan1.labels);
      expect(dbscan2.nClusters).toBe(dbscan1.nClusters);
      expect(dbscan2.eps).toBe(1.0);
      expect(dbscan2.minSamples).toBe(2);
    });
  });

  describe('edge cases', () => {
    it('should handle empty clusters gracefully', () => {
      const data = [
        [0, 0], [100, 100], [200, 200]
      ];

      const model = dbfit(data, { eps: 1.0, minSamples: 2 });

      expect(model.nClusters).toBe(0);
      expect(model.nNoise).toBe(3);
    });

    it('should handle duplicate points', () => {
      const data = [
        [0, 0], [0, 0], [0, 0], [0, 0],
        [10, 10], [10, 10], [10, 10], [10, 10]
      ];

      const model = dbfit(data, { eps: 0.1, minSamples: 3 });

      expect(model.nClusters).toBe(2);
      expect(model.nNoise).toBe(0);
    });

    it('should handle high-dimensional data', () => {
      const data = [
        [0, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.1, 0, 0],
        [10, 10, 10, 10], [10.1, 10, 10, 10], [10, 10.1, 10, 10]
      ];

      const model = dbfit(data, { eps: 0.5, minSamples: 2 });

      expect(model.labels.length).toBe(6);
      expect(model.nClusters).toBeGreaterThan(0);
    });
  });
});
