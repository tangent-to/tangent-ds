/**
 * Model Diagnostics Example
 * Complete workflow for model evaluation and interpretation
 * Using Tangent Notebook format
 */

import { ml, plot, stats } from '@tangent.to/ds';

// ## Generate Dataset with Known Structure
function generateDiagnosticData(n = 150) {
  const X = [];
  const y = [];
  
  for (let i = 0; i < n; i++) {
    const age = 20 + Math.random() * 60;
    const income = 30000 + Math.random() * 100000;
    const experience = Math.random() * 40;
    const education = Math.floor(Math.random() * 5); // 0-4 levels
    const location = Math.random() * 100;
    
    // Target: price prediction
    // Strong dependencies on income and experience
    const noise = (Math.random() - 0.5) * 10000;
    const price = 50000 + 
                  income * 0.3 + 
                  experience * 2000 - 
                  age * 100 +
                  education * 5000 +
                  noise;
    
    X.push([age, income, experience, education, location]);
    y.push(price);
  }
  
  return { 
    X, 
    y,
    featureNames: ['age', 'income', 'experience', 'education', 'location']
  };
}

console.log('=== Model Diagnostics Workflow ===\n');

const { X, y, featureNames } = generateDiagnosticData(150);
console.log('Dataset generated');
console.log('Samples:', X.length);
console.log('Features:', featureNames.join(', '));

// ## Split Data
const split = ml.validation.trainTestSplit(X, y, { testSize: 0.3, shuffle: true });
console.log('\nTrain/Test split:');
console.log('Train size:', split.XTrain.length);
console.log('Test size:', split.XTest.length);

// ## Train Model
console.log('\n--- Training Linear Model ---');
const model = new stats.GLM({ family: 'gaussian' }{ intercept: true });
model.fit(split.XTrain, split.yTrain);
const lmSummary = model.summary();
console.log('Model trained');

// ## Evaluate Performance
const trainPred = model.predict(split.XTrain);
const testPred = model.predict(split.XTest);

const trainR2 = ml.metrics.r2(split.yTrain, trainPred);
const testR2 = ml.metrics.r2(split.yTest, testPred);
const trainRMSE = ml.metrics.rmse(split.yTrain, trainPred);
const testRMSE = ml.metrics.rmse(split.yTest, testPred);

console.log('\n=== Model Performance ===');
console.log(`Train R²: ${trainR2.toFixed(4)}`);
console.log(`Test R²:  ${testR2.toFixed(4)}`);
console.log(`Train RMSE: ${trainRMSE.toFixed(2)}`);
console.log(`Test RMSE:  ${testRMSE.toFixed(2)}`);

// ## Feature Importance Analysis
console.log('\n--- Feature Importance ---');
const importance = ml.interpret.featureImportance(
  model,
  split.XTest,
  split.yTest,
  (yTrue, yPred) => ml.metrics.r2(yTrue, yPred),
  { nRepeats: 10, seed: 42 }
);

console.log('\nTop features:');
importance.slice(0, 5).forEach((imp, i) => {
  const featName = featureNames[imp.feature];
  console.log(`${i + 1}. ${featName}: ${imp.importance.toFixed(4)} ± ${imp.std.toFixed(4)}`);
});

const importancePlot = plot.plotFeatureImportance(
  importance.map(imp => ({
    ...imp,
    feature: featureNames[imp.feature]
  })),
  { topN: 5, width: 600, height: 300 }
);
console.log('\n[Plot] Feature importance chart generated');

// ## Partial Dependence for Top Feature
const topFeatureIdx = importance[0].feature;
const topFeatureName = featureNames[topFeatureIdx];

console.log(`\n--- Partial Dependence: ${topFeatureName} ---`);
const pd = ml.interpret.partialDependence(model, split.XTest, topFeatureIdx, { 
  gridSize: 25 
});

console.log(`Range: [${pd.range[0].toFixed(2)}, ${pd.range[1].toFixed(2)}]`);
console.log(`Prediction range: [${Math.min(...pd.predictions).toFixed(2)}, ${Math.max(...pd.predictions).toFixed(2)}]`);

const pdPlot = plot.plotPartialDependence(pd, {
  featureName: topFeatureName,
  width: 600,
  height: 400
});
console.log('[Plot] Partial dependence curve generated');

// ## Residual Diagnostics
console.log('\n--- Residual Diagnostics ---');
const residuals = ml.interpret.residualPlotData(model, split.XTest, split.yTest);

const meanResidual = residuals.residuals.reduce((a, b) => a + b, 0) / residuals.residuals.length;
const maxResidual = Math.max(...residuals.residuals.map(Math.abs));

console.log(`Mean residual: ${meanResidual.toFixed(2)} (should be ~0)`);
console.log(`Max |residual|: ${maxResidual.toFixed(2)}`);

// Check for outliers (standardized residuals > 2.5)
const outliers = residuals.standardized.filter(r => Math.abs(r) > 2.5);
console.log(`Outliers (|z| > 2.5): ${outliers.length} / ${residuals.standardized.length}`);

const residualPlot = plot.plotResiduals(residuals, {
  standardized: false,
  width: 600,
  height: 400
});
console.log('\n[Plot] Residual vs fitted values generated');

const qqPlot = plot.plotQQ(residuals, { width: 400, height: 400 });
console.log('[Plot] Q-Q normality plot generated');

// ## Correlation Analysis
console.log('\n--- Feature Correlation ---');
const corrMatrix = ml.interpret.correlationMatrix(split.XTrain, featureNames);

console.log('\nCorrelation matrix:');
console.log('       ', featureNames.map(n => n.slice(0, 7).padEnd(7)).join(' '));
featureNames.forEach((feat, i) => {
  const row = corrMatrix.matrix[i].map(v => v.toFixed(2).padStart(6)).join(' ');
  console.log(`${feat.slice(0, 7).padEnd(7)} [${row}]`);
});

// Find high correlations (excluding diagonal)
const highCorr = [];
for (let i = 0; i < featureNames.length; i++) {
  for (let j = i + 1; j < featureNames.length; j++) {
    const corr = corrMatrix.matrix[i][j];
    if (Math.abs(corr) > 0.7) {
      highCorr.push({ feat1: featureNames[i], feat2: featureNames[j], corr });
    }
  }
}

if (highCorr.length > 0) {
  console.log('\nHigh correlations (|r| > 0.7):');
  highCorr.forEach(({ feat1, feat2, corr }) => {
    console.log(`  ${feat1} <-> ${feat2}: ${corr.toFixed(3)}`);
  });
} else {
  console.log('\nNo high correlations detected');
}

const corrPlot = plot.plotCorrelationMatrix(corrMatrix, {
  width: 600,
  height: 600
});
console.log('\n[Plot] Correlation heatmap generated');

// ## Cross-Validation
console.log('\n--- Cross-Validation ---');
const folds = ml.validation.kFold(X, y, 5, true);
const cvResults = ml.validation.crossValidate(
  (Xtrain, ytrain) => {
    const estimator = new stats.GLM({ family: 'gaussian' }{ intercept: true });
    estimator.fit(Xtrain, ytrain);
    return estimator;
  },
  (model, Xtest, ytest) => ml.metrics.r2(ytest, model.predict(Xtest)),
  X,
  y,
  folds
);

console.log(`5-Fold CV R² scores: ${cvResults.scores.map(s => s.toFixed(3)).join(', ')}`);
console.log(`Mean CV R²: ${cvResults.meanScore.toFixed(4)} ± ${cvResults.stdScore.toFixed(4)}`);

// ## Model Summary
console.log('\n=== Diagnostic Summary ===');
console.log(`✓ Model Performance: Test R² = ${testR2.toFixed(3)}, RMSE = ${testRMSE.toFixed(1)}`);
console.log(`✓ Most Important: ${featureNames[importance[0].feature]} (importance: ${importance[0].importance.toFixed(3)})`);
console.log(`✓ Residuals: Mean = ${meanResidual.toFixed(2)}, Outliers = ${outliers.length}`);
console.log(`✓ CV Performance: ${cvResults.meanScore.toFixed(3)} ± ${cvResults.stdScore.toFixed(3)}`);

console.log('\n✓ Complete diagnostics workflow executed');
console.log('All plot configs ready for Observable Plot rendering');
