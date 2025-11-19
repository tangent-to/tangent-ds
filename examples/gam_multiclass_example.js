/**
 * Example: Multi-class GAM Classification and GCV Parameter Control
 *
 * This example demonstrates:
 * 1. Controlling GCV/REML smoothing parameter search
 * 2. Multi-class classification with GAMClassifier (>2 classes)
 */

import { GAMClassifier } from '../src/ml/estimators/GAM.js';

console.log('=== GAM Multi-class Classification Example ===\n');

// Example 1: Controlling GCV Parameters
console.log('1. GCV Parameter Control');
console.log('   You can now control the GCV/REML smoothing parameter search:');
console.log('   - lambdaMin: Minimum smoothing parameter (default: 1e-8)');
console.log('   - lambdaMax: Maximum smoothing parameter (default: 1e4)');
console.log('   - nSteps: Number of grid points in log-space (default: 20)\n');

const gamWithCustomGCV = new GAMClassifier({
  nSplines: 10,
  basis: 'cr',
  smoothMethod: 'GCV',
  lambdaMin: 1e-6,  // Custom minimum
  lambdaMax: 1e3,   // Custom maximum
  nSteps: 30        // More grid points for finer search
});

console.log('   Created GAM with custom GCV parameters:');
console.log('   - lambdaMin: 1e-6');
console.log('   - lambdaMax: 1e3');
console.log('   - nSteps: 30\n');

// Example 2: Multi-class Classification (3+ classes)
console.log('2. Multi-class GAM Classification\n');

// Create a 3-class dataset (Iris-style)
const X_train = [
  // Class "setosa" - small values
  [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
  [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],

  // Class "versicolor" - medium values
  [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [5.5, 2.3], [6.5, 2.8],
  [5.7, 2.8], [6.3, 3.3], [4.9, 2.4], [6.6, 2.9], [5.2, 2.7],

  // Class "virginica" - large values
  [6.3, 3.3], [5.8, 2.7], [7.1, 3.0], [6.3, 2.9], [6.5, 3.0],
  [7.6, 3.0], [4.9, 2.5], [7.3, 2.9], [6.7, 2.5], [7.2, 3.6]
];

const y_train = [
  'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
  'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
  'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
  'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
  'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
  'virginica', 'virginica', 'virginica', 'virginica', 'virginica'
];

console.log('   Training data: 30 samples, 3 classes');
console.log('   Classes: setosa, versicolor, virginica');
console.log('   Features: 2 continuous variables\n');

// Fit multi-class GAM
const gam = new GAMClassifier({
  nSplines: 5,
  basis: 'cr',
  lambda: 0.1  // Fixed smoothing (can also use smoothMethod: 'GCV')
});

console.log('   Fitting multi-class GAM...');
gam.fit(X_train, y_train);

console.log('   ✓ Fit successful!');
console.log(`   - Number of classes: ${gam.gam.K}`);
console.log(`   - Class names: ${gam.gam.classes.join(', ')}`);
console.log(`   - Coefficient vectors fitted: ${gam.gam.coef.length} (K-1 for multinomial)`);

// Display summary
console.log('\n   Model Summary:');
const summary = gam.summary();
console.log(`   - Call: ${summary.call}`);
console.log(`   - Family: ${summary.family}`);
console.log(`   - Link: ${summary.link}`);
console.log(`   - Training accuracy: ${(summary.trainingAccuracy * 100).toFixed(1)}%`);
console.log('   - Per-class accuracy:');
for (const [cls, acc] of Object.entries(summary.perClassAccuracy)) {
  console.log(`     ${cls}: ${(acc * 100).toFixed(1)}%`);
}

// Make predictions
const X_test = [
  [5.0, 3.5],  // Should predict setosa
  [6.5, 3.0],  // Should predict versicolor
  [7.0, 3.0]   // Should predict virginica
];

console.log('\n   Making predictions on test data:');
const predictions = gam.predict(X_test);
const probabilities = gam.predictProba(X_test);

for (let i = 0; i < X_test.length; i++) {
  console.log(`\n   Sample ${i + 1}: [${X_test[i].join(', ')}]`);
  console.log(`   Predicted: ${predictions[i]}`);
  console.log('   Probabilities:');
  for (const [cls, prob] of Object.entries(probabilities[i])) {
    console.log(`     ${cls}: ${(prob * 100).toFixed(1)}%`);
  }
}

// Example 3: Binary classification still works
console.log('\n\n3. Binary Classification (backward compatible)\n');

const X_binary = [
  [1, 2], [1.5, 1.8], [1.2, 2.1],  // Class 0
  [5, 6], [5.5, 5.8], [5.2, 6.1]   // Class 1
];
const y_binary = [0, 0, 0, 1, 1, 1];

const gamBinary = new GAMClassifier({ nSplines: 3 });
gamBinary.fit(X_binary, y_binary);

console.log('   Binary GAM fitted successfully');
console.log(`   Number of classes: ${gamBinary.gam.K}`);
console.log(`   Coefficient vectors: ${gamBinary.gam.coef.length}`);

const probsBinary = gamBinary.predictProba([[1, 2], [5, 6]]);
console.log('\n   Predictions:');
console.log('   [1, 2]:', probsBinary[0]);
console.log('   [5, 6]:', probsBinary[1]);

console.log('\n=== Example Complete ===');
