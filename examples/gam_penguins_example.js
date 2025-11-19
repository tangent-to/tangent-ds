/**
 * Example: GAM with Penguins Dataset
 *
 * This example demonstrates:
 * 1. Using the Recipe API to preprocess categorical variables
 * 2. GAMRegressor for predicting continuous outcomes (body mass)
 * 3. GAMClassifier for multi-class classification (species)
 * 4. Model summaries and evaluation
 */

import * as ds from '../src/index.js';

// Fetch penguins data
const penguinsResponse = await fetch(
  'https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json',
);
const penguinsDataRaw = await penguinsResponse.json();

// Clean the data - remove rows with "." in Sex field and filter out nulls
const penguinsData = penguinsDataRaw
  .map((row) => (row.Sex === '.' ? { ...row, Sex: null } : row))
  .filter((row) => row.Sex && row.Species); // Keep only rows with valid Sex and Species

console.log('=== GAM with Penguins Dataset ===\n');
console.log(`Total samples after cleaning: ${penguinsData.length}\n`);

// ============================================================================
// Example 1: GAMRegressor - Predict Body Mass
// ============================================================================

console.log('--- Example 1: GAMRegressor (Predict Body Mass) ---\n');

// Create recipe for regression
// We'll use numeric features + one-hot encoded Sex to predict body_mass_g
const regressionRecipe = ds.ml
  .recipe({
    data: penguinsData,
    X: ['Bill Length (mm)', 'Bill Depth (mm)', 'Flipper Length (mm)', 'Sex'],
    y: 'Body Mass (g)',
  })
  .oneHot(['Sex']) // One-hot encode Sex (male/female)
  .split({ ratio: 0.7, shuffle: true, seed: 42 });

const regressionPrepped = regressionRecipe.prep();

console.log('Preprocessing:');
console.log('- One-hot encoded Sex into binary features');
console.log('- Train/test split: 70/30');
console.log(`- Training samples: ${regressionPrepped.train.data.length}`);
console.log(`- Test samples: ${regressionPrepped.test.data.length}`);
console.log(`- Features: ${regressionPrepped.train.X.join(', ')}\n`);

// Fit GAM regressor
const gamRegressor = new ds.ml.GAMRegressor({
  nSplines: 8,
  basis: 'cr',
  smoothMethod: 'GCV',
  lambdaMin: 1e-6,
  lambdaMax: 1e3,
  nSteps: 25,
});

gamRegressor.fit({
  data: regressionPrepped.train.data,
  X: regressionPrepped.train.X,
  y: regressionPrepped.train.y,
});

// Display summary
console.log('Model Summary:');
const regSummary = gamRegressor.summary();
console.log(`- ${regSummary.call}`);
console.log(`- R²: ${regSummary.rSquared.toFixed(4)}`);
console.log(`- Residual Std Error: ${regSummary.residualStdError.toFixed(2)}g`);
console.log(`- EDF: ${regSummary.edf.toFixed(2)}`);
console.log(`- Sample size: ${regSummary.n}\n`);

console.log('Smooth Terms:');
for (const term of regSummary.smoothTerms) {
  console.log(
    `  ${term.term}: EDF=${term.edf.toFixed(2)}, p-value=${term.pValue.toFixed(4)}`,
  );
}

// Evaluate on test set
const testPredictions = gamRegressor.predict({
  data: regressionPrepped.test.data,
  X: regressionPrepped.test.X,
});

const testActual = regressionPrepped.test.data.map((row) => row['Body Mass (g)']);
const testR2 = ds.ml.metrics.r2(testActual, testPredictions);
const testRMSE = Math.sqrt(ds.ml.metrics.mse(testActual, testPredictions));

console.log(`\nTest Set Performance:`);
console.log(`- R²: ${testR2.toFixed(4)}`);
console.log(`- RMSE: ${testRMSE.toFixed(2)}g`);

// ============================================================================
// Example 2: GAMClassifier - Predict Species (Multi-class)
// ============================================================================

console.log('\n\n--- Example 2: GAMClassifier (Predict Species) ---\n');

// Create recipe for classification
// Use numeric features + one-hot encoded Sex to predict Species
const classificationRecipe = ds.ml
  .recipe({
    data: penguinsData,
    X: ['Bill Length (mm)', 'Bill Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex'],
    y: 'Species',
  })
  .oneHot(['Sex']) // One-hot encode Sex
  .split({ ratio: 0.7, shuffle: true, seed: 42 });

const classificationPrepped = classificationRecipe.prep();

console.log('Preprocessing:');
console.log('- One-hot encoded Sex into binary features');
console.log('- Train/test split: 70/30');
console.log(`- Training samples: ${classificationPrepped.train.data.length}`);
console.log(`- Test samples: ${classificationPrepped.test.data.length}`);
console.log(`- Features: ${classificationPrepped.train.X.join(', ')}\n`);

// Fit GAM classifier (multi-class!)
const gamClassifier = new ds.ml.GAMClassifier({
  nSplines: 6,
  basis: 'cr',
  lambda: 0.1, // Fixed smoothing parameter
});

// IMPORTANT: Pass encoders to ensure species labels are decoded properly
// The Recipe encodes categorical y to numbers [0,1,2] during preprocessing
// The encoders parameter tells GAM to decode them back to ["Adelie", "Chinstrap", "Gentoo"]
// This ensures predictions return actual species names, not numbers
gamClassifier.fit({
  data: classificationPrepped.train.data,
  X: classificationPrepped.train.X,
  y: classificationPrepped.train.y,
  encoders: classificationPrepped.train.metadata.encoders,
});

// Display summary
console.log('Model Summary:');
const clsSummary = gamClassifier.summary();
console.log(`- ${clsSummary.call}`);
console.log(`- Family: ${clsSummary.family}`);
console.log(`- Link: ${clsSummary.link}`);
console.log(`- Number of classes: ${clsSummary.nClasses}`);
console.log(`- Classes: ${clsSummary.classes.join(', ')}`);
console.log(`- Coefficient vectors: ${clsSummary.nCoefficients} (K-1 for multinomial)`);
console.log(`- Training accuracy: ${(clsSummary.trainingAccuracy * 100).toFixed(1)}%\n`);

console.log('Per-class Accuracy (Training):');
for (const [cls, acc] of Object.entries(clsSummary.perClassAccuracy)) {
  console.log(`  ${cls}: ${(acc * 100).toFixed(1)}%`);
}

// Evaluate on test set
const testClassPredictions = gamClassifier.predict({
  data: classificationPrepped.test.data,
  X: classificationPrepped.test.X,
});

const testClassActual = classificationPrepped.test.data.map((row) => row.Species);
const testAccuracy = ds.ml.metrics.accuracy(testClassActual, testClassPredictions);

console.log(`\nTest Set Performance:`);
console.log(`- Accuracy: ${(testAccuracy * 100).toFixed(1)}%`);

// Show confusion matrix
const confMatrix = {};
const classes = clsSummary.classes;
for (const cls of classes) {
  confMatrix[cls] = {};
  for (const predCls of classes) {
    confMatrix[cls][predCls] = 0;
  }
}

for (let i = 0; i < testClassActual.length; i++) {
  confMatrix[testClassActual[i]][testClassPredictions[i]]++;
}

console.log('\nConfusion Matrix (Test Set):');
console.log('         ', classes.map((c) => c.padEnd(10)).join(' '));
for (const actualCls of classes) {
  const row = classes.map((predCls) => String(confMatrix[actualCls][predCls]).padEnd(10)).join(' ');
  console.log(`${actualCls.padEnd(10)} ${row}`);
}

// ============================================================================
// Example 3: Predictions with Probabilities
// ============================================================================

console.log('\n\n--- Example 3: Prediction Probabilities ---\n');

// Take first 3 test samples
const sampleData = classificationPrepped.test.data.slice(0, 3);
const sampleProbs = gamClassifier.predictProba({
  data: sampleData,
  X: classificationPrepped.test.X,
});
const samplePreds = gamClassifier.predict({
  data: sampleData,
  X: classificationPrepped.test.X,
});

for (let i = 0; i < sampleData.length; i++) {
  console.log(`Sample ${i + 1}:`);
  console.log(`  Bill Length: ${sampleData[i]['Bill Length (mm)']}mm`);
  console.log(`  Bill Depth: ${sampleData[i]['Bill Depth (mm)']}mm`);
  console.log(`  Flipper Length: ${sampleData[i]['Flipper Length (mm)']}mm`);
  console.log(`  Actual: ${sampleData[i].Species}`);
  console.log(`  Predicted: ${samplePreds[i]}`);
  console.log('  Probabilities:');
  for (const [cls, prob] of Object.entries(sampleProbs[i])) {
    console.log(`    ${cls}: ${(prob * 100).toFixed(1)}%`);
  }
  console.log();
}

console.log('=== Example Complete ===');
