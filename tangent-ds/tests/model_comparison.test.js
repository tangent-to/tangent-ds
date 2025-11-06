import { describe, it, expect } from 'vitest';
import { GLM } from '../src/stats/index.js';
import {
  compareModels,
  likelihoodRatioTest,
  pairwiseLRT,
  modelSelectionPlot,
  aicWeightPlot,
  coefficientComparisonPlot,
  crossValidate
} from '../src/stats/index.js';

describe('Model Comparison', () => {
  describe('compareModels', () => {
    it('should compare multiple models by AIC', () => {
      // Create some test data: y = 1 + 2*x1 + 3*x2 + noise
      const X1 = [[1], [2], [3], [4], [5]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2], [5, 3]];
      const y = [6.1, 8.0, 10.9, 13.1, 16.0];

      // Fit two models
      const model1 = new GLM({ family: 'gaussian' });
      model1.fit(X1, y);
      model1._name = 'Simple';

      const model2 = new GLM({ family: 'gaussian' });
      model2.fit(X2, y);
      model2._name = 'Multiple';

      const comparison = compareModels([model1, model2], { criterion: 'aic' });

      expect(comparison.comparison).toHaveLength(2);
      expect(comparison.comparison[0]).toHaveProperty('name');
      expect(comparison.comparison[0]).toHaveProperty('aic');
      expect(comparison.comparison[0]).toHaveProperty('deltaAIC');
      expect(comparison.comparison[0]).toHaveProperty('aicWeight');
      expect(comparison.best).toHaveProperty('name');
      expect(comparison.table).toHaveLength(2);

      // Model 2 should be better (lower AIC)
      expect(comparison.best.name).toBe('Multiple');
    });

    it('should sort models by criterion', () => {
      const X1 = [[1], [2], [3], [4]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2]];
      const X3 = [[1, 1, 1], [2, 1, 0], [3, 2, 1], [4, 2, 0]];
      const y = [6, 8, 11, 13];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);

      const m3 = new GLM({ family: 'gaussian' });
      m3.fit(X3, y);

      const comparison = compareModels([m1, m2, m3], { sort: true });

      // Models should be sorted by AIC (ascending)
      for (let i = 1; i < comparison.comparison.length; i++) {
        expect(comparison.comparison[i].aic).toBeGreaterThanOrEqual(
          comparison.comparison[i - 1].aic
        );
      }
    });

    it('should compute AIC weights', () => {
      const X1 = [[1], [2], [3], [4]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2]];
      const y = [6, 8, 11, 13];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);

      const comparison = compareModels([m1, m2]);

      // AIC weights should sum to approximately 1 after normalization
      const totalWeight = comparison.comparison.reduce((sum, m) => sum + m.aicWeight, 0);
      expect(totalWeight).toBeGreaterThan(0);

      // Best model should have weight = 1 (before normalization)
      expect(comparison.comparison[0].aicWeight).toBe(1);
    });
  });

  describe('likelihoodRatioTest', () => {
    it('should perform LRT for nested models', () => {
      // Nested models: y ~ x1 vs y ~ x1 + x2
      const X1 = [[1], [2], [3], [4], [5]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2], [5, 3]];
      const y = [6, 8, 11, 13, 16];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);

      const lrt = likelihoodRatioTest(m1, m2);

      expect(lrt).toHaveProperty('statistic');
      expect(lrt).toHaveProperty('df');
      expect(lrt).toHaveProperty('pValue');
      expect(lrt).toHaveProperty('significant');
      expect(lrt).toHaveProperty('summary');

      expect(lrt.statistic).toBeGreaterThanOrEqual(0);
      expect(lrt.df).toBe(1); // One additional parameter
      expect(lrt.pValue).toBeGreaterThanOrEqual(0);
      expect(lrt.pValue).toBeLessThanOrEqual(1);
    });

    it('should throw error if models not nested', () => {
      const X = [[1], [2], [3], [4]];
      const y = [6, 8, 10, 12];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X, y);

      // Same model, same df - not nested
      expect(() => likelihoodRatioTest(m1, m2)).toThrow();
    });

    it('should detect significant improvement', () => {
      // Create data where x2 is clearly important
      const X1 = [[1], [2], [3], [4], [5], [6]];
      const X2 = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]];
      const y = [3, 6, 9, 12, 15, 18]; // y = 3 * x2

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);

      const lrt = likelihoodRatioTest(m1, m2);

      // Should detect significant improvement
      expect(lrt.pValue).toBeLessThan(0.05);
      expect(lrt.significant).toBe(true);
    });
  });

  describe('pairwiseLRT', () => {
    it('should perform pairwise LRT for multiple models', () => {
      const X1 = [[1], [2], [3], [4]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2]];
      const X3 = [[1, 1, 1], [2, 1, 0], [3, 2, 1], [4, 2, 0]];
      const y = [6, 8, 11, 13];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);

      const m3 = new GLM({ family: 'gaussian' });
      m3.fit(X3, y);

      const pairwise = pairwiseLRT([m1, m2, m3]);

      expect(pairwise.results).toBeDefined();
      expect(pairwise.table).toBeDefined();
      expect(Array.isArray(pairwise.results)).toBe(true);

      // Should have comparisons for nested models
      expect(pairwise.results.length).toBeGreaterThan(0);

      pairwise.results.forEach(result => {
        expect(result).toHaveProperty('statistic');
        expect(result).toHaveProperty('pValue');
        expect(result).toHaveProperty('nested');
        expect(result).toHaveProperty('full');
      });
    });
  });

  describe('modelSelectionPlot', () => {
    it('should generate plot specification for model selection', () => {
      const X1 = [[1], [2], [3], [4]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2]];
      const y = [6, 8, 11, 13];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);
      m1._name = 'Model 1';

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);
      m2._name = 'Model 2';

      const plotSpec = modelSelectionPlot([m1, m2]);

      expect(plotSpec).toHaveProperty('criterion');
      expect(plotSpec).toHaveProperty('delta');
      expect(plotSpec).toHaveProperty('data');
      expect(plotSpec).toHaveProperty('best');

      // Check criterion plot structure
      expect(plotSpec.criterion).toHaveProperty('marks');
      expect(plotSpec.criterion).toHaveProperty('x');
      expect(plotSpec.criterion).toHaveProperty('y');

      // Check delta plot structure
      expect(plotSpec.delta).toHaveProperty('marks');
      expect(plotSpec.delta).toHaveProperty('x');
      expect(plotSpec.delta).toHaveProperty('y');

      // Check data
      expect(Array.isArray(plotSpec.data)).toBe(true);
      expect(plotSpec.data.length).toBe(2);
    });
  });

  describe('aicWeightPlot', () => {
    it('should generate AIC weight plot specification', () => {
      const X1 = [[1], [2], [3], [4]];
      const X2 = [[1, 1], [2, 1], [3, 2], [4, 2]];
      const y = [6, 8, 11, 13];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X1, y);

      const m2 = new GLM({ family: 'gaussian' });
      m2.fit(X2, y);

      const plotSpec = aicWeightPlot([m1, m2]);

      expect(plotSpec).toHaveProperty('marks');
      expect(plotSpec).toHaveProperty('x');
      expect(plotSpec).toHaveProperty('y');
      expect(Array.isArray(plotSpec.marks)).toBe(true);
      expect(plotSpec.marks.length).toBeGreaterThan(0);
    });
  });

  describe('coefficientComparisonPlot', () => {
    it('should generate coefficient comparison plot', () => {
      const X = [[1, 1], [2, 1], [3, 2], [4, 2]];
      const y = [6, 8, 11, 13];

      const m1 = new GLM({ family: 'gaussian' });
      m1.fit(X, y);
      m1._name = 'Gaussian';

      const m2 = new GLM({ family: 'poisson' });
      m2.fit(X, y);
      m2._name = 'Poisson';

      const plotSpec = coefficientComparisonPlot([m1, m2]);

      expect(plotSpec).toHaveProperty('marks');
      expect(plotSpec).toHaveProperty('x');
      expect(plotSpec).toHaveProperty('y');
      expect(Array.isArray(plotSpec.marks)).toBe(true);
    });
  });

  describe('crossValidate', () => {
    it('should perform k-fold cross-validation', () => {
      const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
      const y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

      const modelFactory = () => new GLM({ family: 'gaussian' });

      const cv = crossValidate(modelFactory, X, y, { k: 5, metric: 'mse' });

      expect(cv).toHaveProperty('scores');
      expect(cv).toHaveProperty('mean');
      expect(cv).toHaveProperty('std');
      expect(cv).toHaveProperty('metric');
      expect(cv).toHaveProperty('k');

      expect(cv.scores).toHaveLength(5);
      expect(cv.k).toBe(5);
      expect(cv.metric).toBe('mse');

      // MSE should be low for this perfect linear relationship
      expect(cv.mean).toBeLessThan(1);
    });

    it('should shuffle data if requested', () => {
      const X = [[1], [2], [3], [4], [5], [6]];
      const y = [2, 4, 6, 8, 10, 12];

      const modelFactory = () => new GLM({ family: 'gaussian' });

      const cv1 = crossValidate(modelFactory, X, y, { k: 3, shuffle: false });
      const cv2 = crossValidate(modelFactory, X, y, { k: 3, shuffle: true, seed: 42 });

      // Both should complete successfully
      expect(cv1.scores).toHaveLength(3);
      expect(cv2.scores).toHaveLength(3);
    });

    it('should support different metrics', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];

      const modelFactory = () => new GLM({ family: 'gaussian' });

      const cvMSE = crossValidate(modelFactory, X, y, { k: 2, metric: 'mse' });
      const cvMAE = crossValidate(modelFactory, X, y, { k: 2, metric: 'mae' });
      const cvRMSE = crossValidate(modelFactory, X, y, { k: 2, metric: 'rmse' });

      expect(cvMSE.metric).toBe('mse');
      expect(cvMAE.metric).toBe('mae');
      expect(cvRMSE.metric).toBe('rmse');

      expect(cvMSE.mean).toBeGreaterThanOrEqual(0);
      expect(cvMAE.mean).toBeGreaterThanOrEqual(0);
      expect(cvRMSE.mean).toBeGreaterThanOrEqual(0);
    });
  });
});
