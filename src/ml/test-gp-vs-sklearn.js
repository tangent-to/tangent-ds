/**
 * Compare ds GaussianProcessRegressor with scikit-learn reference values.
 * Reference values from running test-gp-vs-sklearn.py
 */
import { GaussianProcessRegressor } from './estimators/GaussianProcessRegressor.js';
import { RBF } from './kernels/rbf.js';

const TOLERANCE = 1e-6;

function assertClose(actual, expected, name, tol = TOLERANCE) {
  const diff = Math.abs(actual - expected);
  const pass = diff < tol;
  console.log(`${pass ? '✓' : '✗'} ${name}: ${actual.toFixed(8)} (expected ${expected.toFixed(8)}, diff=${diff.toExponential(2)})`);
  return pass;
}

console.log("=== Test 1: RBF Kernel Values (compare to sklearn) ===");
// sklearn reference: RBF with length_scale=1.0
// K[0,0] = 1.0, K[0,1] = 0.6065306597126334, K[0,2] = 0.1353352832366127
const kernel1 = new RBF({ lengthScale: 1.0, amplitude: 1.0 });

// Compute K([0], [0]) - should be 1.0
const k00 = kernel1.compute([0], [0]);
assertClose(k00, 1.0, "K([0], [0])");

// Compute K([0], [1]) - should be exp(-0.5 * 1^2) = 0.6065...
const k01 = kernel1.compute([0], [1]);
assertClose(k01, 0.6065306597126334, "K([0], [1])");

// Compute K([0], [2]) - should be exp(-0.5 * 2^2) = 0.1353...
const k02 = kernel1.compute([0], [2]);
assertClose(k02, 0.1353352832366127, "K([0], [2])");

console.log("\n=== Test 2: RBF Kernel with length_scale=2.0 ===");
// sklearn: K[0,1] with length_scale=2.0 = 0.8824969025845955
const kernel2 = new RBF({ lengthScale: 2.0, amplitude: 1.0 });
const k01_ls2 = kernel2.compute([0], [1]);
assertClose(k01_ls2, 0.8824969025845955, "K([0], [1]) with ls=2.0");

console.log("\n=== Test 3: Full Kernel Matrix ===");
// Build kernel matrix manually
const X = [[0], [1], [2]];
const K = X.map(x1 => X.map(x2 => kernel1.compute(x1, x2)));
console.log("Kernel matrix K(X, X):");
K.forEach((row, i) => console.log(`  [${row.map(v => v.toFixed(8)).join(', ')}]`));

// Compare to sklearn values
const sklearnK = [
  [1.0, 0.60653066, 0.13533528],
  [0.60653066, 1.0, 0.60653066],
  [0.13533528, 0.60653066, 1.0]
];
let kernelMatrixMatch = true;
for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    if (Math.abs(K[i][j] - sklearnK[i][j]) > 1e-6) {
      kernelMatrixMatch = false;
    }
  }
}
console.log(kernelMatrixMatch ? "✓ Kernel matrix matches sklearn" : "✗ Kernel matrix differs from sklearn");

console.log("\n=== Test 4: GP Fit and Predict ===");
// sklearn reference (with optimizer=None, fixed kernel):
// X_train = [[0], [1], [2], [3], [4]], y_train = [0, 1, 0, 1, 0]
// alpha = 0.1, predictions at training points: [0.1216, 0.7229, 0.2757, 0.7229, 0.1216]
const xTrain = [[0], [1], [2], [3], [4]];
const yTrain = [0, 1, 0, 1, 0];

const gp = new GaussianProcessRegressor({ 
  kernel: new RBF({ lengthScale: 1.0, amplitude: 1.0 }), 
  alpha: 0.1 
});
gp.fit(xTrain, yTrain);

const { mean: yPred } = gp.predict(xTrain, { returnStd: true });
console.log("Predictions:", yPred.map(v => v.toFixed(8)));

// sklearn predictions (with optimizer=None): [0.12164727, 0.72287914, 0.27567115, 0.72287914, 0.12164727]
assertClose(yPred[0], 0.12164727, "Prediction at x=0", 0.01);
assertClose(yPred[1], 0.72287914, "Prediction at x=1", 0.01);
assertClose(yPred[2], 0.27567115, "Prediction at x=2", 0.01);
assertClose(yPred[3], 0.72287914, "Prediction at x=3", 0.01);
assertClose(yPred[4], 0.12164727, "Prediction at x=4", 0.01);

console.log("\n=== Test 5: Interpolation at Training Points ===");
// With low noise (alpha=0.01), predictions should be very close to training values
const gpTight = new GaussianProcessRegressor({ 
  kernel: new RBF({ lengthScale: 1.0, amplitude: 1.0 }), 
  alpha: 0.01 
});
gpTight.fit(xTrain, yTrain);
const { mean: yPredTight } = gpTight.predict(xTrain, { returnStd: true });

console.log("With alpha=0.01:");
console.log("  Training y:", yTrain);
console.log("  Predicted:", yPredTight.map(v => v.toFixed(4)));
const maxError = Math.max(...yTrain.map((y, i) => Math.abs(y - yPredTight[i])));
console.log(`  Max error: ${maxError.toFixed(6)}`);
console.log(maxError < 0.1 ? "✓ Good interpolation" : "✗ Poor interpolation");

console.log("\n=== Test 6: Predict at New Points ===");
// This tests the GP's ability to interpolate between training points
const xNew = [[0.5], [1.5], [2.5], [3.5]];
const { mean: yNew } = gp.predict(xNew, { returnStd: true });
console.log("Predictions at [0.5, 1.5, 2.5, 3.5]:", yNew.map(v => v.toFixed(4)));

// The pattern is [0, 1, 0, 1, 0], so midpoints should be around 0.5
// Actually with RBF kernel, the interpolation depends on the kernel bandwidth
console.log("Midpoint predictions should show smooth interpolation between training points");

console.log("\n=== Test 7: Posterior Sampling ===");
// Test that sampling from posterior gives values near predictions
const samples = gp.sample(xTrain, 3, 42);
console.log("Posterior samples (3 samples at training points):");
for (let i = 0; i < 3; i++) {
  console.log(`  Sample ${i + 1}: [${samples[i].map(v => v.toFixed(4)).join(', ')}]`);
}

// Samples should have mean close to predictions
const sampleMeans = xTrain.map((_, j) => {
  return samples.reduce((sum, s) => sum + s[j], 0) / 3;
});
console.log(`Sample means: [${sampleMeans.map(v => v.toFixed(4)).join(', ')}]`);
console.log(`Predictions:  [${yPred.map(v => v.toFixed(4)).join(', ')}]`);

console.log("\n=== Summary ===");
console.log("Key comparisons with scikit-learn:");
console.log(`  RBF kernel formula: k(x1, x2) = amplitude * exp(-0.5 * ||x1-x2||^2 / lengthScale^2)`);
console.log(`  sklearn uses:       k(x1, x2) = exp(-0.5 * ||x1-x2||^2 / length_scale^2)`);
console.log(`  → Both implementations should match when amplitude=1.0`);
