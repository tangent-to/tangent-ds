import { describe, it, expect } from 'vitest';
import { pca as pcaFns, PCA } from '../src/mva/index.js';
import { approxEqual } from '../src/core/math.js';

function scoreObjectsToMatrix(scores) {
  if (!scores.length) return [];
  const componentKeys = Object.keys(scores[0])
    .filter((key) => key.startsWith('pc'))
    .sort((a, b) => {
      const ai = Number(a.slice(2));
      const bi = Number(b.slice(2));
      return ai - bi;
    });
  return scores.map((score) => componentKeys.map((key) => score[key]));
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function pairwiseDistances(matrix) {
  const n = matrix.length;
  const distances = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      distances.push(euclideanDistance(matrix[i], matrix[j]));
    }
  }
  return distances;
}

function correlation(a, b) {
  const n = a.length;
  const meanA = a.reduce((sum, val) => sum + val, 0) / n;
  const meanB = b.reduce((sum, val) => sum + val, 0) / n;

  let numerator = 0;
  let denomA = 0;
  let denomB = 0;

  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    numerator += da * db;
    denomA += da * da;
    denomB += db * db;
  }

  if (denomA === 0 || denomB === 0) return 0;
  return numerator / Math.sqrt(denomA * denomB);
}

describe('PCA - Principal Component Analysis (class API)', () => {
  describe('fit', () => {
    it('should fit PCA on simple 2D data', () => {
      const X = [
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8]
      ];

      const p = new PCA();
      p.fit(X);
      const model = p.model;

      expect(model.scores.length).toBe(4);
      expect(model.loadings.length).toBe(2);
      expect(model.eigenvalues.length).toBe(2);
      expect(model.varianceExplained.length).toBe(2);
    });

    it('should explain 100% variance with all components', () => {
      const X = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
      ];

      const p = new PCA();
      p.fit(X);
      const totalVariance = p.model.varianceExplained.reduce((a, b) => a + b, 0);

      expect(approxEqual(totalVariance, 1, 0.001)).toBe(true);
    });

    it('should center data by default', () => {
      const X = [[10, 20], [11, 21], [12, 22]];
      const p = new PCA({ center: true });
      p.fit(X);

      expect(p.model.means).toBeTruthy();
      expect(p.model.means.length).toBe(2);
      expect(approxEqual(p.model.means[0], 11, 0.001)).toBe(true);
      expect(approxEqual(p.model.means[1], 21, 0.001)).toBe(true);
    });

    it('should scale data when requested', () => {
      const X = [[1, 10], [2, 20], [3, 30]];
      const p = new PCA({ scale: true });
      p.fit(X);

      expect(p.model.sds).toBeTruthy();
      expect(p.model.sds.length).toBe(2);
    });

    it('should sort components by variance', () => {
      const X = [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0.1, 0],
        [4, 0, 0.1]
      ];

      const p = new PCA();
      p.fit(X);
      const model = p.model;

      // First eigenvalue should be largest
      for (let i = 0; i < model.eigenvalues.length - 1; i++) {
        expect(model.eigenvalues[i]).toBeGreaterThanOrEqual(model.eigenvalues[i + 1]);
      }
    });

    it('should retain original feature names when provided via columns', () => {
      const data = [
        { sepal_length: 5.1, sepal_width: 3.5 },
        { sepal_length: 4.9, sepal_width: 3.0 },
        { sepal_length: 4.7, sepal_width: 3.2 },
        { sepal_length: 4.6, sepal_width: 3.1 }
      ];

      const p = new PCA();
      p.fit({ data, columns: ['sepal_length', 'sepal_width'] });
      const { loadings, featureNames } = p.model;

      expect(loadings[0].variable).toBe('sepal_length');
      expect(loadings[1].variable).toBe('sepal_width');
      expect(featureNames).toEqual(['sepal_length', 'sepal_width']);
    });

    it('scaling=1 preserves Euclidean distances of standardized samples', () => {
      const X = [
        [2, 4, 6],
        [3, 5, 7],
        [4, 6, 8],
        [5, 7, 9]
      ];

      const p = new PCA({ center: true, scale: true, scaling: 1 });
      p.fit(X);

      const means = p.model.means;
      const sds = p.model.sds;
      const standardized = X.map((row) =>
        row.map((val, j) => (val - means[j]) / sds[j])
      );

      const scoresMatrix = scoreObjectsToMatrix(p.model.scores);

      const distStd = pairwiseDistances(standardized);
      const distScores = pairwiseDistances(scoresMatrix);

      for (let i = 0; i < distStd.length; i++) {
        expect(distScores[i]).toBeCloseTo(distStd[i], 6);
      }
    });

    it('scaling=2 produces loadings equal to correlations with components', () => {
      const X = [
        [10, 20, 30],
        [20, 25, 35],
        [30, 15, 20],
        [40, 30, 10],
        [50, 35, 25],
      ];

      const p = new PCA({ center: true, scale: true, scaling: 2 });
      p.fit(X);

      const means = p.model.means;
      const sds = p.model.sds;
      const standardized = X.map((row) =>
        row.map((val, j) => (val - means[j]) / sds[j])
      );

      const scoresMatrix = scoreObjectsToMatrix(p.model.scores);

      p.model.loadings.forEach((loading, varIdx) => {
        const variableValues = standardized.map((row) => row[varIdx]);
        const componentKeys = Object.keys(loading)
          .filter((key) => key.startsWith('pc'))
          .sort((a, b) => Number(a.slice(2)) - Number(b.slice(2)));
        const vector = componentKeys.map((key) => loading[key]);
        const length = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        componentKeys.forEach((key, compIdx) => {
          const compValues = scoresMatrix.map((row) => row[compIdx]);
          const corr = correlation(variableValues, compValues);
          const cosine = length > 0 ? vector[compIdx] / length : 0;
          expect(corr).toBeCloseTo(cosine, 6);
        });
      });
    });

    it('getScores returns raw and scaled coordinates consistently', () => {
      const X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
      ];

      const p = new PCA({ center: true, scaling: 1 });
      p.fit(X);

      const raw = p.getScores('sites', false);
      const scaled = p.getScores('sites', true);

      expect(raw.length).toBe(scaled.length);
      expect(raw[0]).toHaveProperty('pc1');
      expect(scaled[0]).toHaveProperty('pc1');
      // Raw coordinates should differ from scaled for scaling=1
      expect(Math.abs(raw[0].pc1 - scaled[0].pc1)).toBeGreaterThan(0);
    });
  });

  describe('transform', () => {
    it('should transform new data', () => {
      const X = [[1, 2], [2, 4], [3, 6]];
      const p = new PCA();
      p.fit(X);
      const model = p.model;

      const Xnew = [[4, 8], [5, 10]];
      const transformed = p.transform(Xnew);

      expect(transformed.length).toBe(2);
      expect(transformed[0].pc1).toBeDefined();
      expect(transformed[0].pc2).toBeDefined();
    });

    it('should apply same standardization as training', () => {
      const X = [[10, 20], [11, 21], [12, 22]];
      const p = new PCA({ center: true, scale: true });
      p.fit(X);

      const Xnew = [[11, 21]]; // Same as mean
      const transformed = p.transform(Xnew);

      // Should be close to origin in PC space
      expect(Math.abs(transformed[0].pc1)).toBeLessThan(1);
    });
  });

  describe('cumulativeVariance', () => {
    it('should compute cumulative variance explained', () => {
      const X = [[1, 0], [2, 0], [3, 0.1]];
      const p = new PCA();
      p.fit(X);
      const model = p.model;

      const cumulative = pcaFns.cumulativeVariance(model);

      expect(cumulative.length).toBe(model.varianceExplained.length);
      expect(cumulative[cumulative.length - 1]).toBeLessThanOrEqual(1.001);

      // Should be increasing
      for (let i = 0; i < cumulative.length - 1; i++) {
        expect(cumulative[i + 1]).toBeGreaterThanOrEqual(cumulative[i]);
      }
    });
  });

  describe('edge cases', () => {
    it('should throw error for insufficient samples', () => {
      const X = [[1, 2]];
      const p = new PCA();
      expect(() => p.fit(X)).toThrow();
    });

    it('should handle 1D data', () => {
      const X = [[1], [2], [3]];
      const p = new PCA();
      p.fit(X);
      const model = p.model;

      expect(model.scores.length).toBe(3);
      expect(model.eigenvalues.length).toBe(1);
    });
  });
});
