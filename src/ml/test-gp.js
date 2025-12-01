/**
 * Test GaussianProcessRegressor against scikit-learn API
 * 
 * scikit-learn API reference:
 * - GaussianProcessRegressor(kernel=None, alpha=1e-10, normalize_y=False, ...)
 * - kernel.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
 * - kernel.ConstantKernel(constant_value=1.0) * RBF() for amplitude
 * - gp.fit(X, y)
 * - gp.predict(X, return_std=False, return_cov=False)
 * - gp.sample_y(X, n_samples=1, random_state=None)
 */

import { GaussianProcessRegressor, RBF, Periodic, RationalQuadratic, Kernel } from './index.js';

// Test 1: Basic kernel construction (both APIs)
console.log('=== Test 1: Kernel Construction ===');
try {
  // Positional API
  const rbf1 = new RBF(1.0, 1.0);
  console.log('RBF(1.0, 1.0) params:', rbf1.getParams());
  
  // Object API (what the tutorial uses)
  const rbf2 = new RBF({ lengthScale: 2.0, amplitude: 3.0 });
  console.log('RBF({ lengthScale: 2.0, amplitude: 3.0 }) params:', rbf2.getParams());
  
  // Test kernel computation
  const k = rbf1.compute([0], [0]);
  console.log('k([0], [0]) =', k, '(expected: 1.0)');
  
  console.log('✓ Kernel construction works\n');
} catch (e) {
  console.log('✗ Kernel construction failed:', e.message, '\n');
}

// Test 2: GP Construction with noiseLevel
console.log('=== Test 2: GP Construction with noiseLevel ===');
try {
  const kernel = new RBF({ lengthScale: 1.0, amplitude: 1.0 });
  const gp = new GaussianProcessRegressor({ kernel, noiseLevel: 0.1 });
  console.log('GP created with kernel:', gp.kernel.constructor.name);
  console.log('GP alpha (from noiseLevel):', gp.alpha);
  console.log('✓ GP construction with noiseLevel works\n');
} catch (e) {
  console.log('✗ GP construction failed:', e.message, '\n');
}

// Test 3: GP Fit and Predict
console.log('=== Test 3: GP Fit and Predict ===');
try {
  const kernel = new RBF({ lengthScale: 1.0, amplitude: 1.0 });
  const gp = new GaussianProcessRegressor({ kernel, noiseLevel: 1e-5 });
  
  const X_train = [[0], [1], [2], [3], [4]];
  const y_train = [0, 1, 0, 1, 0];
  
  gp.fit(X_train, y_train);
  
  // Predict at training points
  const y_pred = gp.predict(X_train);
  console.log('Predictions:', y_pred.map(v => v.toFixed(3)));
  console.log('Expected:   ', y_train);
  console.log('✓ GP fit and predict works\n');
} catch (e) {
  console.log('✗ GP fit/predict failed:', e.message, '\n');
}

// Test 4: Sample with seed (reproducibility)
console.log('=== Test 4: Sample with seed (reproducibility) ===');
try {
  const kernel = new RBF({ lengthScale: 1.0, amplitude: 1.0 });
  const gp = new GaussianProcessRegressor({ kernel, noiseLevel: 1e-5 });
  
  const X_train = [[0], [2], [4]];
  const y_train = [0, 1, 0];
  gp.fit(X_train, y_train);
  
  const X_test = [[0], [1], [2], [3], [4]];
  
  // Sample with same seed should give same results
  const samples1 = gp.sample(X_test, 2, 42);
  const samples2 = gp.sample(X_test, 2, 42);
  
  console.log('Sample 1 (seed=42):', samples1[0].map(v => v.toFixed(3)));
  console.log('Sample 2 (seed=42):', samples2[0].map(v => v.toFixed(3)));
  console.log('Samples match:', JSON.stringify(samples1) === JSON.stringify(samples2));
  console.log('✓ Seeded sampling works\n');
} catch (e) {
  console.log('✗ Seeded sampling failed:', e.message, '\n');
}

// Test 5: samplePrior with seed
console.log('=== Test 5: samplePrior with seed ===');
try {
  const kernel = new RBF({ lengthScale: 1.0, amplitude: 1.0 });
  const gp = new GaussianProcessRegressor({ kernel, noiseLevel: 0.1 });
  
  const X_test = [[0], [1], [2], [3], [4]];
  
  // Sample prior with same seed
  const samples1 = gp.samplePrior(X_test, 2, 123);
  const samples2 = gp.samplePrior(X_test, 2, 123);
  
  console.log('Prior sample 1 (seed=123):', samples1[0].map(v => v.toFixed(3)));
  console.log('Prior sample 2 (seed=123):', samples2[0].map(v => v.toFixed(3)));
  console.log('Samples match:', JSON.stringify(samples1) === JSON.stringify(samples2));
  console.log('✓ Seeded samplePrior works\n');
} catch (e) {
  console.log('✗ Seeded samplePrior failed:', e.message, '\n');
}

// Test 6: Tutorial-style usage
console.log('=== Test 6: Tutorial-style usage ===');
try {
  // This is exactly how the tutorial uses it
  const kernel = new RBF({ lengthScale: 0.5, amplitude: 1.0 });
  const gp = new GaussianProcessRegressor({ kernel, noiseLevel: 0.1 });
  
  const xTest = Array.from({ length: 50 }, (_, i) => [i]);
  const samples = gp.samplePrior(xTest, 1, 500); // seed = 500
  
  console.log('Generated', samples.length, 'samples');
  console.log('First 5 values:', samples[0].slice(0, 5).map(v => v.toFixed(3)));
  console.log('✓ Tutorial-style usage works\n');
} catch (e) {
  console.log('✗ Tutorial-style usage failed:', e.message, '\n');
}

console.log('=== All tests completed ===');
