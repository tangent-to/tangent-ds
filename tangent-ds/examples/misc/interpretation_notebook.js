// ---
// title: Interpretation
// id: interpretation
// ---

// %% [markdown]
/*
 * Model Interpretation Example
 * Demonstrates feature importance, partial dependence, and residual analysis
 * Using Tangent Notebook format
*/

// %% [javascript]
import { ml, plot, stats } from '@tangent.to/ds';

// Generate synthetic regression dataset
function generateRegressionData(n = 100) {
  const X = [];
  const y = [];
  
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 10;
    const x2 = Math.random() * 5;
    const x3 = Math.random() * 3;
    const x4 = Math.random() * 2; // Less important feature
    
    // y = 3*x1 + 2*x2 - 1.5*x3 + noise
    const noise = (Math.random() - 0.5) * 2;
    const target = 3 * x1 + 2 * x2 - 1.5 * x3 + noise;
    
    X.push([x1, x2, x3, x4]);
    y.push(target);
  }
  
  return { X, y };
}

// ## Generate Data
const { X, y } = generateRegressionData(100);
console.log('Dataset shape:', X.length, 'x', X[0].length);

// %% [javascript]
// ## Train Linear Model
const linearModel = new stats.GLM({ family: 'gaussian' }{ intercept: true });
linearModel.fit(X, y);
const modelSummary = linearModel.summary();
const model = linearModel;
console.log('\nModel trained');
console.log('R²:', modelSummary.rSquared.toFixed(4));
console.log('Number of coefficients:', linearModel.coef.length);

// %% [javascript]
// ## Feature Importance (Permutation-based)
const importance = ml.interpret.featureImportance(
  model,
  X,
  y,
  (yTrue, yPred) => ml.metrics.r2(yTrue, yPred),
  { nRepeats: 5 }
);

console.log('\n=== Feature Importance ===');
importance.forEach((imp, i) => {
  console.log(`Feature ${imp.feature}: ${imp.importance.toFixed(4)} ± ${imp.std.toFixed(4)}`);
});

// %% [javascript]
// ## Generate Feature Importance Plot
const importancePlot = plot.plotFeatureImportance(importance, { 
  topN: 4,
  width: 600,
  height: 300
});
console.log('\n[Plot Config] Feature Importance:', importancePlot.type);

// %% [javascript]
// ## Partial Dependence for Feature 0
const pd0 = ml.interpret.partialDependence(model, X, 0, { gridSize: 20 });
console.log('\n=== Partial Dependence (Feature 0) ===');
console.log('Range:', pd0.range);
console.log('Sample values:', pd0.values.slice(0, 5).map(v => v.toFixed(2)));
console.log('Sample predictions:', pd0.predictions.slice(0, 5).map(p => p.toFixed(2)));

const pdPlot = plot.plotPartialDependence(pd0, {
  featureName: 'Feature 0',
  width: 600,
  height: 400
});
console.log('\n[Plot Config] Partial Dependence:', pdPlot.type);

// %% [javascript]
// ## Residual Analysis
const residuals = ml.interpret.residualPlotData(model, X, y);
console.log('\n=== Residual Analysis ===');
console.log('Mean residual:', 
  residuals.residuals.reduce((a, b) => a + b, 0) / residuals.residuals.length
);
console.log('Residual std:', 
  Math.sqrt(
    residuals.residuals.reduce((sum, r) => sum + r * r, 0) / residuals.residuals.length
  ).toFixed(4)
);

const residualPlot = plot.plotResiduals(residuals, {
  standardized: false,
  width: 600,
  height: 400
});
console.log('\n[Plot Config] Residuals:', residualPlot.type);

const qqPlot = plot.plotQQ(residuals, {
  width: 400,
  height: 400
});
console.log('[Plot Config] Q-Q Plot:', qqPlot.type);

// %% [javascript]
// ## Correlation Matrix
const corrMatrix = ml.interpret.correlationMatrix(X, ['X1', 'X2', 'X3', 'X4']);
console.log('\n=== Correlation Matrix ===');
corrMatrix.features.forEach((feat, i) => {
  const row = corrMatrix.matrix[i].map(v => v.toFixed(2)).join('  ');
  console.log(`${feat}: [${row}]`);
});

const corrPlot = plot.plotCorrelationMatrix(corrMatrix, {
  width: 600,
  height: 600
});
console.log('\n[Plot Config] Correlation Matrix:', corrPlot.type);

// %% [javascript]
// ## Learning Curve
console.log('\n=== Learning Curve ===');
const learningCurve = ml.interpret.learningCurve(
  (Xtrain, ytrain) => {
    const learner = new stats.GLM({ family: 'gaussian' }{ intercept: true });
    learner.fit(Xtrain, ytrain);
    return learner;
  },
  (yTrue, yPred) => ml.metrics.r2(yTrue, yPred),
  X,
  y,
  { trainSizes: [0.3, 0.5, 0.7, 0.9], cv: 3 }
);

console.log('Training sizes:', learningCurve.trainSizes);
console.log('Train scores:', learningCurve.trainScores.map(s => s.toFixed(3)));
console.log('Test scores:', learningCurve.testScores.map(s => s.toFixed(3)));

const lcPlot = plot.plotLearningCurve(learningCurve, {
  width: 600,
  height: 400
});
console.log('\n[Plot Config] Learning Curve:', lcPlot.type);

console.log('\n✓ All interpretation tools executed successfully');
console.log('Note: Plot configs can be rendered with Observable Plot in browser/notebook');
