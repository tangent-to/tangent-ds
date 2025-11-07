/**
 * Hyperparameter Tuning Example
 * Demonstrates GridSearchCV and RandomSearchCV
 * Using Tangent Notebook format
 */

import { ml, stats } from '@tangent.to/ds';

console.log('=== Hyperparameter Tuning Demo ===\n');

// ## Generate Polynomial Data

function generatePolyData(n = 100, degree = 2, noise = 0.5) {
  const X = [];
  const y = [];
  
  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 6; // -3 to 3
    
    // y = x^2 - 2x + 1 + noise
    let target = 0;
    for (let d = 0; d <= degree; d++) {
      target += [1, -2, 1][d] * Math.pow(x, d);
    }
    target += (Math.random() - 0.5) * noise;
    
    X.push([x]);
    y.push(target);
  }
  
  return { X, y };
}

const { X, y } = generatePolyData(150, 2, 1.0);
console.log(`Generated polynomial dataset: ${X.length} samples\n`);

// ## Grid Search for Polynomial Degree

console.log('=== GridSearchCV: Finding Best Polynomial Degree ===\n');

// Create polynomial features manually for different degrees
function createPolyFeatures(X, degree) {
  return X.map(row => {
    const features = [];
    for (let d = 1; d <= degree; d++) {
      features.push(Math.pow(row[0], d));
    }
    return features;
  });
}

const fitFn = (Xtrain, ytrain, params) => {
  const degree = params.degree;
  const XPoly = createPolyFeatures(Xtrain, degree);
  const model = new stats.GLM({ family: 'gaussian' }{ intercept: true });
  model.fit(XPoly, ytrain);
  model._polyDegree = degree;
  return model;
};

const scoreFn = (model, Xtest, ytest) => {
  const degree = model._polyDegree ?? (model.coef.length - 1);
  const XPolyTest = createPolyFeatures(Xtest, degree);
  const yPred = model.predict(XPolyTest);
  return ml.metrics.r2(ytest, yPred);
};

const paramGrid = {
  degree: [1, 2, 3, 4, 5]
};

const gridResult = ml.tuning.GridSearchCV(
  fitFn,
  scoreFn,
  X,
  y,
  paramGrid,
  { k: 5, verbose: false }
);

console.log('Grid Search Results:\n');
gridResult.results.forEach(result => {
  console.log(`Degree ${result.params.degree}: R² = ${result.meanScore.toFixed(4)} ± ${result.stdScore.toFixed(4)}`);
});

console.log(`\nBest parameters: degree = ${gridResult.bestParams.degree}`);
console.log(`Best CV score: ${gridResult.bestScore.toFixed(4)}\n`);

// ## Random Search with Distributions

console.log('=== RandomSearchCV: Exploring Parameter Space ===\n');

// For demonstration, we'll search over a made-up parameter space
const fitFnRandom = (Xtrain, ytrain, params) => {
  // Just use linear model for simplicity
  // In practice, params would control model complexity
  const model = new stats.GLM({ family: 'gaussian' }{ intercept: true });
  model.fit(Xtrain, ytrain);
  return model;
};

const scoreFnRandom = (model, Xtest, ytest) => {
  const yPred = model.predict(Xtest);
  return ml.metrics.r2(ytest, yPred);
};

const paramDistributions = {
  param1: ml.tuning.distributions.uniform(0, 1),
  param2: ml.tuning.distributions.loguniform(0.001, 1.0),
  param3: ml.tuning.distributions.choice(['a', 'b', 'c'])
};

const randomResult = ml.tuning.RandomSearchCV(
  fitFnRandom,
  scoreFnRandom,
  X,
  y,
  paramDistributions,
  { nIter: 10, k: 3, verbose: false }
);

console.log('Random Search Results (first 5):\n');
randomResult.results.slice(0, 5).forEach((result, i) => {
  console.log(`Trial ${i + 1}:`);
  console.log(`  param1: ${result.params.param1.toFixed(3)}`);
  console.log(`  param2: ${result.params.param2.toFixed(3)}`);
  console.log(`  param3: ${result.params.param3}`);
  console.log(`  Score: ${result.meanScore.toFixed(4)}\n`);
});

console.log(`Best parameters:`, randomResult.bestParams);
console.log(`Best score: ${randomResult.bestScore.toFixed(4)}\n`);

// ## GridSearch vs RandomSearch Comparison

console.log('=== Comparison: Grid vs Random Search ===\n');

// Simple grid
const simpleGrid = {
  degree: [1, 2, 3]
};

const gridResult2 = ml.tuning.GridSearchCV(
  fitFn,
  scoreFn,
  X,
  y,
  simpleGrid,
  { k: 3, verbose: false }
);

// Random search with degree
const randomResult2 = ml.tuning.RandomSearchCV(
  fitFn,
  scoreFn,
  X,
  y,
  { degree: [1, 2, 3, 4, 5] },
  { nIter: 3, k: 3, verbose: false }
);

console.log('Grid Search (3 combinations):');
console.log(`  Best degree: ${gridResult2.bestParams.degree}`);
console.log(`  Best score: ${gridResult2.bestScore.toFixed(4)}`);
console.log(`  Total fits: ${gridResult2.results.length * 3}`); // 3 folds

console.log('\nRandom Search (3 random trials):');
console.log(`  Best degree: ${randomResult2.bestParams.degree}`);
console.log(`  Best score: ${randomResult2.bestScore.toFixed(4)}`);
console.log(`  Total fits: ${randomResult2.results.length * 3}\n`);

// ## Analyzing Search Results

console.log('=== Detailed Analysis ===\n');

console.log('Score distribution from Grid Search:');
const scores = gridResult.results.map(r => r.meanScore);
console.log(`  Min: ${Math.min(...scores).toFixed(4)}`);
console.log(`  Max: ${Math.max(...scores).toFixed(4)}`);
console.log(`  Mean: ${(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(4)}`);
console.log(`  Std: ${Math.sqrt(scores.reduce((sum, s) => sum + (s - scores.reduce((a,b) => a+b, 0)/scores.length)**2, 0) / scores.length).toFixed(4)}\n`);

// Find degree that balances complexity and performance
console.log('Model Complexity vs Performance:');
gridResult.results.forEach(result => {
  const complexity = result.params.degree;
  const performance = result.meanScore;
  const variance = result.stdScore;
  
  console.log(`  Degree ${complexity}: R²=${performance.toFixed(4)}, var=${variance.toFixed(4)}, complexity=${complexity}`);
});

// ## Best Practice Insights

console.log('\n=== Best Practices ===\n');

console.log('Grid Search:');
console.log('  ✓ Exhaustive search over discrete values');
console.log('  ✓ Guaranteed to find best combination');
console.log('  ✗ Expensive for large grids');
console.log('  ✗ Exponential growth with parameters\n');

console.log('Random Search:');
console.log('  ✓ Efficient for continuous parameters');
console.log('  ✓ Scales well with parameters');
console.log('  ✓ Can sample more important ranges');
console.log('  ✗ No guarantee of finding optimum\n');

console.log('When to use each:');
console.log('  - Grid: Small discrete parameter space');
console.log('  - Random: Large continuous space or many parameters');
console.log('  - Random: Initial exploration, then Grid for refinement\n');

console.log('✓ Hyperparameter tuning demo complete');
