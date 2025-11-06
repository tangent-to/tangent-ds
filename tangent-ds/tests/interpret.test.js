import { describe, it, expect } from 'vitest';
import * as interpret from '../src/ml/interpret.js';
import { GLM } from '../src/stats/index.js';
import * as metrics from '../src/ml/metrics.js';

// Helper to create and fit a linear model using GLM
function fitLinearModel(X, y, options = {}) {
  const model = new GLM({ family: 'gaussian', intercept: options.intercept !== false });
  model.fit(X, y);
  return model;
}

describe('Model Interpretation', () => {
  // Simple linear regression dataset
  const X = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10]
  ];
  const y = [10, 15, 20, 25, 30, 35, 40, 45]; // y = 5*x1 + noise
  
  describe('featureImportance', () => {
    it('should compute permutation importance', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const importance = interpret.featureImportance(
        model,
        X,
        y,
        (yTrue, yPred) => metrics.r2(yTrue, yPred),
        { nRepeats: 5, seed: 42 }
      );
      
      expect(importance).toHaveLength(3);
      expect(importance[0]).toHaveProperty('feature');
      expect(importance[0]).toHaveProperty('importance');
      expect(importance[0]).toHaveProperty('std');
      
      // First feature should be most important (highest importance)
      expect(importance[0].importance).toBeGreaterThan(0);
    });
    
    it('should return features sorted by importance', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const importance = interpret.featureImportance(
        model,
        X,
        y,
        (yTrue, yPred) => metrics.r2(yTrue, yPred),
        { nRepeats: 3 }
      );
      
      // Check descending order
      for (let i = 0; i < importance.length - 1; i++) {
        expect(importance[i].importance).toBeGreaterThanOrEqual(importance[i + 1].importance);
      }
    });
  });
  
  describe('coefficientImportance', () => {
    it('should compute importance from linear model coefficients', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const importance = interpret.coefficientImportance(model);
      
      // Model has intercept, so coefficients include it
      // But coefficientImportance only returns feature importances (not intercept)
      expect(importance.length).toBeGreaterThan(0);
      expect(importance[0]).toHaveProperty('feature');
      expect(importance[0]).toHaveProperty('importance');
      expect(importance[0]).toHaveProperty('coefficient');
      
      // Should be sorted by absolute value
      for (let i = 0; i < importance.length - 1; i++) {
        expect(importance[i].importance).toBeGreaterThanOrEqual(importance[i + 1].importance);
      }
    });
    
    it('should use feature names if provided', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      model.intercept = true; // Mark that this model has an intercept
      const featureNames = ['age', 'income', 'experience'];
      
      const importance = interpret.coefficientImportance(model, featureNames);
      
      // Should have 3 features (intercept excluded)
      expect(importance).toHaveLength(3);
      expect(importance.every(imp => featureNames.includes(imp.feature))).toBe(true);
    });
    
    it('should throw error if model has no coefficients', () => {
      const badModel = { predict: () => [] };
      
      expect(() => {
        interpret.coefficientImportance(badModel);
      }).toThrow('Model must have coefficients property');
    });
  });
  
  describe('partialDependence', () => {
    it('should compute partial dependence for a feature', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const pd = interpret.partialDependence(model, X, 0, { gridSize: 10 });
      
      expect(pd).toHaveProperty('feature', 0);
      expect(pd).toHaveProperty('values');
      expect(pd).toHaveProperty('predictions');
      expect(pd).toHaveProperty('range');
      
      expect(pd.values).toHaveLength(10);
      expect(pd.predictions).toHaveLength(10);
      expect(pd.range).toHaveLength(2);
      
      // Values should be ascending
      for (let i = 0; i < pd.values.length - 1; i++) {
        expect(pd.values[i]).toBeLessThanOrEqual(pd.values[i + 1]);
      }
    });
    
    it('should respect percentile bounds', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const pd = interpret.partialDependence(model, X, 0, { 
        gridSize: 10, 
        percentiles: [0.1, 0.9] 
      });
      
      const featureValues = X.map(row => row[0]).sort((a, b) => a - b);
      const min = featureValues[Math.floor(X.length * 0.1)];
      const max = featureValues[Math.floor(X.length * 0.9)];
      
      expect(pd.range[0]).toBeCloseTo(min, 5);
      expect(pd.range[1]).toBeCloseTo(max, 5);
    });
  });
  
  describe('residualPlotData', () => {
    it('should compute residual statistics', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const residuals = interpret.residualPlotData(model, X, y);
      
      expect(residuals).toHaveProperty('fitted');
      expect(residuals).toHaveProperty('residuals');
      expect(residuals).toHaveProperty('standardized');
      expect(residuals).toHaveProperty('yTrue', y);
      
      expect(residuals.fitted).toHaveLength(y.length);
      expect(residuals.residuals).toHaveLength(y.length);
      expect(residuals.standardized).toHaveLength(y.length);
    });
    
    it('should compute residuals correctly', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const residuals = interpret.residualPlotData(model, X, y);
      
      // residual = y - yhat
      for (let i = 0; i < y.length; i++) {
        expect(residuals.residuals[i]).toBeCloseTo(y[i] - residuals.fitted[i], 5);
      }
    });
    
    it('should compute standardized residuals', () => {
      const model = fitLinearModel(X, y, { intercept: true });
      
      const residuals = interpret.residualPlotData(model, X, y);
      
      // Mean of standardized residuals should be ~0
      const mean = residuals.standardized.reduce((a, b) => a + b, 0) / residuals.standardized.length;
      expect(Math.abs(mean)).toBeLessThan(1e-10);
      
      // Std of standardized residuals should be ~1
      const variance = residuals.standardized.reduce((sum, r) => sum + r * r, 0) / residuals.standardized.length;
      expect(Math.sqrt(variance)).toBeCloseTo(1, 5);
    });
  });
  
  describe('correlationMatrix', () => {
    it('should compute correlation matrix', () => {
      const corrResult = interpret.correlationMatrix(X);
      
      expect(corrResult).toHaveProperty('matrix');
      expect(corrResult).toHaveProperty('features');
      
      expect(corrResult.matrix).toHaveLength(3);
      expect(corrResult.matrix[0]).toHaveLength(3);
      expect(corrResult.features).toHaveLength(3);
    });
    
    it('should have 1s on diagonal', () => {
      const corrResult = interpret.correlationMatrix(X);
      
      for (let i = 0; i < 3; i++) {
        expect(corrResult.matrix[i][i]).toBeCloseTo(1, 10);
      }
    });
    
    it('should be symmetric', () => {
      const corrResult = interpret.correlationMatrix(X);
      
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(corrResult.matrix[i][j]).toBeCloseTo(corrResult.matrix[j][i], 10);
        }
      }
    });
    
    it('should use custom feature names', () => {
      const featureNames = ['A', 'B', 'C'];
      const corrResult = interpret.correlationMatrix(X, featureNames);
      
      expect(corrResult.features).toEqual(featureNames);
    });
    
    it('should compute valid correlation values', () => {
      const corrResult = interpret.correlationMatrix(X);
      
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(corrResult.matrix[i][j]).toBeGreaterThanOrEqual(-1);
          expect(corrResult.matrix[i][j]).toBeLessThanOrEqual(1.01); // Allow small float error
        }
      }
    });
  });
  
  describe('learningCurve', () => {
    it('should compute learning curve', () => {
      // Use larger training sizes to avoid singular matrices
      const result = interpret.learningCurve(
        (Xtrain, ytrain) => fitLinearModel(Xtrain, ytrain, { intercept: true }),
        (yTrue, yPred) => metrics.r2(yTrue, yPred),
        X,
        y,
        { trainSizes: [0.5, 0.7, 0.9], cv: 2 }
      );
      
      expect(result).toHaveProperty('trainSizes');
      expect(result).toHaveProperty('trainScores');
      expect(result).toHaveProperty('testScores');
      
      expect(result.trainSizes).toHaveLength(3);
      expect(result.trainScores).toHaveLength(3);
      expect(result.testScores).toHaveLength(3);
    });
    
    it('should show increasing training sizes', () => {
      const result = interpret.learningCurve(
        (Xtrain, ytrain) => fitLinearModel(Xtrain, ytrain, { intercept: true }),
        (yTrue, yPred) => metrics.r2(yTrue, yPred),
        X,
        y,
        { trainSizes: [0.5, 0.7, 0.9], cv: 2 }
      );
      
      for (let i = 0; i < result.trainSizes.length - 1; i++) {
        expect(result.trainSizes[i]).toBeLessThan(result.trainSizes[i + 1]);
      }
    });
    
    it('should return valid scores', () => {
      const result = interpret.learningCurve(
        (Xtrain, ytrain) => fitLinearModel(Xtrain, ytrain, { intercept: true }),
        (yTrue, yPred) => metrics.r2(yTrue, yPred),
        X,
        y,
        { trainSizes: [0.6, 0.8], cv: 2 }
      );
      
      result.trainScores.forEach(score => {
        // Scores can be negative for poor fits, but should be numeric
        expect(typeof score).toBe('number');
        expect(isNaN(score)).toBe(false);
      });
      
      result.testScores.forEach(score => {
        expect(typeof score).toBe('number');
        expect(isNaN(score)).toBe(false);
      });
    });
  });
});
