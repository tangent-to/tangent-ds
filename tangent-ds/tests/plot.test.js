import { describe, it, expect } from 'vitest';
import * as plot from '../src/plot/index.js';
import * as mva from '../src/mva/index.js';
import * as ml from '../src/ml/index.js';
import * as interpret from '../src/ml/interpret.js';
import { GLM } from '../src/stats/index.js';

// Helper to create and fit a linear model using GLM
function fitLinearModel(X, y, options = {}) {
  const model = new GLM({ family: 'gaussian', intercept: options.intercept !== false });
  model.fit(X, y);
  return model;
}

describe('Plot Configuration Generators', () => {
  describe('plotPCA', () => {
    const data = [
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
      [1.9, 2.2],
      [3.1, 3.0]
    ];
    
    it('should generate PCA plot config', () => {
      const pcaResult = mva.pca.fit(data, { nComponents: 2 });
      const config = plot.plotPCA(pcaResult);
      
      expect(config).toHaveProperty('type', 'pca');
      expect(config).toHaveProperty('width');
      expect(config).toHaveProperty('height');
      expect(config).toHaveProperty('data');
      expect(config).toHaveProperty('axes');
      expect(config).toHaveProperty('marks');
      
      expect(config.data.scores).toBeDefined();
      expect(config.marks.length).toBeGreaterThan(0);
    });
    
    it('should include loadings when requested', () => {
      const pcaResult = mva.pca.fit(data, { nComponents: 2 });
      const config = plot.plotPCA(pcaResult, { showLoadings: true });
      
      expect(config.data.loadings).toBeDefined();
      expect(config.marks.length).toBeGreaterThanOrEqual(2);
    });
    
    it('should handle color grouping', () => {
      const pcaResult = mva.pca.fit(data, { nComponents: 2 });
      const colorBy = [0, 0, 1, 1, 1];
      const config = plot.plotPCA(pcaResult, { colorBy });
      
      config.data.scores.forEach((point, i) => {
        expect(point.color).toBe(colorBy[i]);
      });
    });
    
    it('should support custom dimensions', () => {
      const pcaResult = mva.pca.fit(data, { nComponents: 2 });
      const config = plot.plotPCA(pcaResult, { width: 800, height: 600 });
      
      expect(config.width).toBe(800);
      expect(config.height).toBe(600);
    });
  });
  
  describe('plotScree', () => {
    it('should generate scree plot config', () => {
      const data = [
        [2.5, 2.4, 1.0],
        [0.5, 0.7, 0.3],
        [2.2, 2.9, 1.2],
        [1.9, 2.2, 0.9],
        [3.1, 3.0, 1.5]
      ];
      
      const pcaResult = mva.pca.fit(data, { nComponents: 3 });
      const config = plot.plotScree(pcaResult);
      
      expect(config).toHaveProperty('type', 'scree');
      expect(config).toHaveProperty('data');
      expect(config.data.components).toHaveLength(3);
      
      config.data.components.forEach((comp, i) => {
        expect(comp).toHaveProperty('component', i + 1);
        expect(comp).toHaveProperty('variance');
        expect(comp.variance).toBeGreaterThan(0);
      });
    });
  });
  
  describe('plotLDA', () => {
    const X = [
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [5.5, 8.2],
      [9, 5],
      [9.2, 5.5]
    ];
    const y = [0, 0, 1, 1, 2, 2];
    
    it('should generate LDA plot config', () => {
      const ldaResult = mva.lda.fit(X, y);
      const config = plot.plotLDA(ldaResult);
      
      expect(config).toHaveProperty('type', 'lda');
      expect(config).toHaveProperty('data');
      expect(config.data.scores).toBeDefined();
      expect(config.data.scores.length).toBe(X.length);
      
      config.data.scores.forEach(point => {
        expect(point).toHaveProperty('x');
        expect(point).toHaveProperty('y');
        expect(point).toHaveProperty('class');
      });
    });
    
    it('should handle 2D discriminants', () => {
      const ldaResult = mva.lda.fit(X, y);
      const config = plot.plotLDA(ldaResult, { ldX: 1, ldY: 2 });
      
      expect(config.axes.x.label).toBe('LD1');
      expect(config.axes.y.label).toBe('LD2');
    });
  });
  
  describe('plotHCA', () => {
    const data = [
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [5.5, 8.2]
    ];
    
    it('should generate dendrogram config', () => {
      const hcaResult = ml.hca.fit(data);
      const config = plot.plotHCA(hcaResult);
      
      expect(config).toHaveProperty('type', 'dendrogram');
      expect(config).toHaveProperty('data');
      expect(config).toHaveProperty('config');
      
      expect(config.data.nodes).toBeDefined();
      expect(config.data.merges).toBeDefined();
      expect(config.data.nodes.length).toBe(data.length);
    });
    
    it('should include linkage method', () => {
      const hcaResult = ml.hca.fit(data, { linkage: 'complete' });
      const config = plot.plotHCA(hcaResult);
      
      expect(config.config.linkage).toBe('complete');
    });
  });
  
  describe('dendrogramLayout', () => {
    it('should generate layout coordinates', () => {
      const data = [
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [5.5, 8.2]
      ];
      
      const hcaResult = ml.hca.fit(data);
      const dendroConfig = plot.plotHCA(hcaResult);
      const layout = plot.dendrogramLayout(dendroConfig, { width: 640, height: 400 });
      
      expect(layout).toHaveProperty('type', 'dendrogramLayout');
      expect(layout).toHaveProperty('data');
      expect(layout.data).toHaveProperty('nodes');
      expect(layout.data).toHaveProperty('links');
      
      // Check that nodes have coordinates
      layout.data.nodes.forEach(node => {
        expect(node).toHaveProperty('x');
        expect(node).toHaveProperty('y');
        expect(node).toHaveProperty('isLeaf');
      });
      
      // Check links
      layout.data.links.forEach(link => {
        expect(link).toHaveProperty('source');
        expect(link).toHaveProperty('target');
      });
    });
  });
  
  describe('plotRDA', () => {
    const Y = [
      [2.0, 1.5],
      [3.5, 2.0],
      [1.0, 0.5],
      [4.0, 3.0]
    ];
    
    const X = [
      [1.0, 2.0],
      [2.0, 3.0],
      [0.5, 1.0],
      [3.0, 4.0]
    ];
    
    it('should generate RDA triplot config', () => {
      const rdaResult = mva.rda.fit(Y, X, { nComponents: 2 });
      const config = plot.plotRDA(rdaResult);
      
      expect(config).toHaveProperty('type', 'rda');
      expect(config).toHaveProperty('data');
      expect(config.data).toHaveProperty('sites');
      expect(config.data).toHaveProperty('species');
      
      expect(config.data.sites.length).toBe(Y.length);
      expect(config.data.species.length).toBe(Y[0].length);
    });
  });

  describe('plotSilhouette', () => {
    it('should plot silhouette scores from raw data', () => {
      const data = [
        [0, 0],
        [0, 1],
        [5, 5],
        [6, 5],
        [5, 6]
      ];
      const labels = [0, 0, 1, 1, 1];

      const config = plot.plotSilhouette({ data, labels });

      expect(config.type).toBe('silhouette');
      expect(config.data.values.length).toBe(data.length);
      expect(config.axes.x.label).toBe('Silhouette score');
      expect(typeof config.show).toBe('function');
    });

    it('should accept precomputed samples', () => {
      const samples = [
        { index: 0, cluster: '0', silhouette: 0.8, a: 0.1, b: 0.5 },
        { index: 1, cluster: '1', silhouette: 0.2, a: 0.4, b: 0.5 }
      ];

      const config = plot.plotSilhouette({ samples });

      expect(config.data.values[0].silhouette).toBe(0.8);
      expect(config.data.values[1].cluster).toBe('1');
    });
  });
  
  describe('plotFeatureImportance', () => {
    it('should generate feature importance plot config', () => {
      const importances = [
        { feature: 0, importance: 0.5, std: 0.1 },
        { feature: 1, importance: 0.3, std: 0.05 },
        { feature: 2, importance: 0.15, std: 0.03 },
        { feature: 3, importance: 0.05, std: 0.01 }
      ];
      
      const config = plot.plotFeatureImportance(importances);
      
      expect(config).toHaveProperty('type', 'featureImportance');
      expect(config).toHaveProperty('data');
      expect(config.data.features).toBeDefined();
      expect(config.data.features.length).toBeLessThanOrEqual(10);
    });
    
    it('should respect topN parameter', () => {
      const importances = Array.from({ length: 20 }, (_, i) => ({
        feature: i,
        importance: 1 - i * 0.05,
        std: 0.1
      }));
      
      const config = plot.plotFeatureImportance(importances, { topN: 5 });
      
      expect(config.data.features).toHaveLength(5);
    });
  });
  
  describe('plotPartialDependence', () => {
    it('should generate PD plot config', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [2, 4, 6, 8, 10];
      const model = fitLinearModel(X, y, { intercept: true });
      
      const pd = interpret.partialDependence(model, X, 0, { gridSize: 10 });
      const config = plot.plotPartialDependence(pd);
      
      expect(config).toHaveProperty('type', 'partialDependence');
      expect(config).toHaveProperty('data');
      expect(config.data.points).toHaveLength(10);
      
      config.data.points.forEach(point => {
        expect(point).toHaveProperty('value');
        expect(point).toHaveProperty('prediction');
      });
    });
    
    it('should use custom feature name', () => {
      const X = [[1], [2], [3]];
      const y = [2, 4, 6];
      const model = fitLinearModel(X, y, { intercept: true });
      
      const pd = interpret.partialDependence(model, X, 0);
      const config = plot.plotPartialDependence(pd, { featureName: 'Temperature' });
      
      expect(config.axes.x.label).toBe('Temperature');
    });
  });
  
  describe('plotCorrelationMatrix', () => {
    it('should generate correlation heatmap config', () => {
      const X = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
      ];
      
      const corrResult = interpret.correlationMatrix(X, ['A', 'B', 'C']);
      const config = plot.plotCorrelationMatrix(corrResult);
      
      expect(config).toHaveProperty('type', 'correlationMatrix');
      expect(config).toHaveProperty('data');
      expect(config.data.cells).toHaveLength(9); // 3x3 matrix
      
      config.data.cells.forEach(cell => {
        expect(cell).toHaveProperty('feature1');
        expect(cell).toHaveProperty('feature2');
        expect(cell).toHaveProperty('correlation');
        expect(cell.correlation).toBeGreaterThanOrEqual(-1);
        expect(cell.correlation).toBeLessThanOrEqual(1.01); // Allow small float error
      });
    });
  });
  
  describe('plotResiduals', () => {
    it('should generate residual plot config', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [2, 4, 6, 8, 10];
      const model = fitLinearModel(X, y, { intercept: true });
      
      const residuals = interpret.residualPlotData(model, X, y);
      const config = plot.plotResiduals(residuals);
      
      expect(config).toHaveProperty('type', 'residuals');
      expect(config).toHaveProperty('data');
      expect(config.data.points).toHaveLength(y.length);
      
      config.data.points.forEach(point => {
        expect(point).toHaveProperty('fitted');
        expect(point).toHaveProperty('residual');
      });
    });
    
    it('should support standardized residuals', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];
      const model = fitLinearModel(X, y, { intercept: true });
      
      const residuals = interpret.residualPlotData(model, X, y);
      const config = plot.plotResiduals(residuals, { standardized: true });
      
      expect(config.axes.y.label).toBe('Standardized Residuals');
    });
  });
  
  describe('plotQQ', () => {
    it('should generate Q-Q plot config', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [2, 4, 6, 8, 10];
      const model = fitLinearModel(X, y, { intercept: true });
      
      const residuals = interpret.residualPlotData(model, X, y);
      const config = plot.plotQQ(residuals);
      
      expect(config).toHaveProperty('type', 'qq');
      expect(config).toHaveProperty('data');
      expect(config.data.points).toBeDefined();
      expect(config.data.reference).toBeDefined();
      
      config.data.points.forEach(point => {
        expect(point).toHaveProperty('theoretical');
        expect(point).toHaveProperty('observed');
      });
    });
  });
  
  describe('plotLearningCurve', () => {
    it('should generate learning curve plot config', () => {
      const lcResult = {
        trainSizes: [10, 20, 30],
        trainScores: [0.7, 0.8, 0.85],
        testScores: [0.65, 0.75, 0.8]
      };
      
      const config = plot.plotLearningCurve(lcResult);
      
      expect(config).toHaveProperty('type', 'learningCurve');
      expect(config).toHaveProperty('data');
      expect(config.data.points).toHaveLength(6); // 3 train + 3 test
      
      const trainPoints = config.data.points.filter(p => p.type === 'train');
      const testPoints = config.data.points.filter(p => p.type === 'test');
      
      expect(trainPoints).toHaveLength(3);
      expect(testPoints).toHaveLength(3);
    });
  });
  
  describe('Plot Config Structure', () => {
    it('should return objects, not DOM elements', () => {
      const data = [[1, 2], [2, 3], [3, 4]];
      const pcaResult = mva.pca.fit(data);
      const config = plot.plotPCA(pcaResult);
      
      // Ensure it's a plain object, not a DOM element
      expect(typeof config).toBe('object');
      expect(config.constructor.name).toBe('Object');
      expect(config.nodeType).toBeUndefined();
    });
    
    it('should have consistent structure across plot types', () => {
      const data = [[1, 2], [2, 3], [3, 4]];
      const pcaResult = mva.pca.fit(data);
      const config = plot.plotPCA(pcaResult);
      
      // All plot configs should have these properties
      expect(config).toHaveProperty('type');
      expect(config).toHaveProperty('width');
      expect(config).toHaveProperty('height');
      expect(config).toHaveProperty('data');
      expect(config).toHaveProperty('marks');
    });
  });
});
