#!/usr/bin/env node

/**
 * Compare tangent-ds GLM/GLMM implementations with R reference
 *
 * This script:
 * 1. Runs the R script to generate reference results
 * 2. Runs the same tests using tangent-ds GLM
 * 3. Compares coefficients, fitted values, and other statistics
 */

import { spawn } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { GLM } from '../tangent-ds/src/stats/estimators/GLM.js';

// ==============================================================================
// Utility Functions
// ==============================================================================

function maxAbsDiff(arr1, arr2) {
  if (arr1.length !== arr2.length) {
    throw new Error(`Array length mismatch: ${arr1.length} vs ${arr2.length}`);
  }
  let maxDiff = 0;
  for (let i = 0; i < arr1.length; i++) {
    const diff = Math.abs(arr1[i] - arr2[i]);
    maxDiff = Math.max(maxDiff, diff);
  }
  return maxDiff;
}

function runRScript() {
  return new Promise((resolve, reject) => {
    console.log('Running R reference script...');
    const rProcess = spawn('Rscript', ['glm_reference.R'], {
      cwd: process.cwd()
    });

    let stdout = '';
    let stderr = '';

    rProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    rProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    rProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('R script stderr:', stderr);
        reject(new Error(`R script exited with code ${code}`));
      } else {
        console.log(stdout);
        resolve();
      }
    });
  });
}

// ==============================================================================
// Test Implementations (matching R datasets)
// ==============================================================================

function testGaussianGLM() {
  const X = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 6],
    [7, 5],
    [8, 7],
    [9, 8],
    [10, 9]
  ];
  const y = [3, 5, 7, 6, 9, 11, 10, 13, 15, 17];

  const model = new GLM({ family: 'gaussian', link: 'identity' });
  model.fit(X, y);

  return {
    coefficients: model._model.coefficients,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    standardErrors: model._model.standardErrors,
    deviance: model._model.deviance,
    nullDeviance: model._model.nullDeviance,
    aic: model._model.aic,
    bic: model._model.bic,
    dfResidual: model._model.dfResidual,
    converged: model._model.converged
  };
}

function testBinomialGLM() {
  const X = [
    [1, 2],
    [2, 3],
    [3, 2],
    [4, 4],
    [5, 3],
    [6, 5],
    [7, 4],
    [8, 6],
    [9, 5],
    [10, 7]
  ];
  const y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];

  const model = new GLM({ family: 'binomial', link: 'logit' });
  model.fit(X, y);

  return {
    coefficients: model._model.coefficients,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    standardErrors: model._model.standardErrors,
    deviance: model._model.deviance,
    nullDeviance: model._model.nullDeviance,
    aic: model._model.aic,
    bic: model._model.bic,
    dfResidual: model._model.dfResidual,
    converged: model._model.converged,
    iterations: model._model.iterations
  };
}

function testPoissonGLM() {
  const X = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 6],
    [7, 5],
    [8, 7],
    [9, 8],
    [10, 9]
  ];
  const y = [2, 3, 5, 4, 7, 9, 8, 11, 13, 15];

  const model = new GLM({ family: 'poisson', link: 'log' });
  model.fit(X, y);

  return {
    coefficients: model._model.coefficients,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    standardErrors: model._model.standardErrors,
    deviance: model._model.deviance,
    nullDeviance: model._model.nullDeviance,
    aic: model._model.aic,
    bic: model._model.bic,
    dfResidual: model._model.dfResidual,
    converged: model._model.converged
  };
}

function testGammaGLM() {
  const X = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 6],
    [7, 5],
    [8, 7],
    [9, 8],
    [10, 9]
  ];
  const y = [1.2, 2.3, 3.1, 2.8, 4.5, 5.9, 5.2, 7.3, 8.7, 10.1];

  const model = new GLM({ family: 'gamma', link: 'inverse' });
  model.fit(X, y);

  return {
    coefficients: model._model.coefficients,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    standardErrors: model._model.standardErrors,
    deviance: model._model.deviance,
    nullDeviance: model._model.nullDeviance,
    aic: model._model.aic,
    bic: model._model.bic,
    dfResidual: model._model.dfResidual,
    converged: model._model.converged
  };
}

function testInverseGaussianGLM() {
  const X = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 6],
    [7, 5],
    [8, 7],
    [9, 8],
    [10, 9]
  ];
  const y = [0.8, 1.5, 2.2, 1.9, 3.1, 3.8, 3.5, 4.9, 5.7, 6.5];

  const model = new GLM({ family: 'inverse_gaussian', link: 'inverse_squared' });
  model.fit(X, y);

  return {
    coefficients: model._model.coefficients,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    standardErrors: model._model.standardErrors,
    deviance: model._model.deviance,
    nullDeviance: model._model.nullDeviance,
    aic: model._model.aic,
    bic: model._model.bic,
    dfResidual: model._model.dfResidual,
    converged: model._model.converged
  };
}

function testGaussianGLMM() {
  const X = [];
  const y = [];
  const groups = [];

  // Build data: rep(1:10, each = 3)
  const X1 = [1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9, 10,10,10];
  const X2 = [2,3,4, 3,4,5, 4,5,6, 3,4,5, 5,6,7, 6,7,8, 5,6,7, 7,8,9, 8,9,10, 9,10,11];
  const yData = [3,4,5, 5,6,7, 7,8,9, 6,7,8, 9,10,11, 11,12,13, 10,11,12, 13,14,15, 15,16,17, 17,18,19];
  const groupData = ['A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C'];

  for (let i = 0; i < X1.length; i++) {
    X.push([X1[i], X2[i]]);
    y.push(yData[i]);
    groups.push(groupData[i]);
  }

  const randomEffects = {
    intercept: groups
  };

  const model = new GLM({
    family: 'gaussian',
    link: 'identity',
    randomEffects: randomEffects
  });

  model.fit(X, y);

  return {
    fixedEffects: model._model.fixedEffects,
    randomEffects: model._model.randomEffects,
    varianceComponents: model._model.varianceComponents,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    logLikelihood: model._model.logLikelihood,
    aic: model._model.aic,
    bic: model._model.bic,
    ngroups: model._model.groupInfo[0].nGroups,
    converged: model._model.converged
  };
}

function testBinomialGLMM() {
  const X = [];
  const y = [];
  const groups = [];

  const X1 = [1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9, 10,10,10];
  const X2 = [2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4];
  const yData = [0,0,1, 0,0,1, 0,1,1, 0,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1];
  const groupData = ['A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C'];

  for (let i = 0; i < X1.length; i++) {
    X.push([X1[i], X2[i]]);
    y.push(yData[i]);
    groups.push(groupData[i]);
  }

  const randomEffects = {
    intercept: groups
  };

  const model = new GLM({
    family: 'binomial',
    link: 'logit',
    randomEffects: randomEffects
  });

  model.fit(X, y);

  return {
    fixedEffects: model._model.fixedEffects,
    randomEffects: model._model.randomEffects,
    varianceComponents: model._model.varianceComponents,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    logLikelihood: model._model.logLikelihood,
    aic: model._model.aic,
    bic: model._model.bic,
    ngroups: model._model.groupInfo[0].nGroups,
    converged: model._model.converged
  };
}

function testPoissonGLMM() {
  const X = [];
  const y = [];
  const groups = [];

  const X1 = [1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9, 10,10,10];
  const X2 = [2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4, 2,3,4];
  const yData = [2,3,4, 3,4,5, 5,6,7, 4,5,6, 7,8,9, 9,10,11, 8,9,10, 11,12,13, 13,14,15, 15,16,17];
  const groupData = ['A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C', 'A','B','C'];

  for (let i = 0; i < X1.length; i++) {
    X.push([X1[i], X2[i]]);
    y.push(yData[i]);
    groups.push(groupData[i]);
  }

  const randomEffects = {
    intercept: groups
  };

  const model = new GLM({
    family: 'poisson',
    link: 'log',
    randomEffects: randomEffects
  });

  model.fit(X, y);

  return {
    fixedEffects: model._model.fixedEffects,
    randomEffects: model._model.randomEffects,
    varianceComponents: model._model.varianceComponents,
    fitted: model._model.fitted,
    residuals: model._model.residuals,
    logLikelihood: model._model.logLikelihood,
    aic: model._model.aic,
    bic: model._model.bic,
    ngroups: model._model.groupInfo[0].nGroups,
    converged: model._model.converged
  };
}

// ==============================================================================
// Comparison Logic
// ==============================================================================

function compareResults(testName, tangentResult, rResult) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Test: ${testName}`);
  console.log('='.repeat(70));

  const isGLMM = testName.includes('glmm');
  const coeffField = isGLMM ? 'fixedEffects' : 'coefficients';
  const rCoeffField = isGLMM ? 'fixed_effects' : 'coefficients';

  // Compare coefficients
  const coefDiff = maxAbsDiff(tangentResult[coeffField], rResult[rCoeffField]);
  console.log(`Coefficients max abs diff: ${coefDiff.toExponential(4)}`);
  console.log(`  tangent: [${tangentResult[coeffField].map(x => x.toFixed(6)).join(', ')}]`);
  console.log(`  R:       [${rResult[rCoeffField].map(x => x.toFixed(6)).join(', ')}]`);

  // Compare fitted values
  const fittedDiff = maxAbsDiff(tangentResult.fitted, rResult.fitted_values);
  console.log(`Fitted values max abs diff: ${fittedDiff.toExponential(4)}`);

  // Compare standard errors (if GLM)
  if (!isGLMM && tangentResult.standardErrors && rResult.standard_errors) {
    const seDiff = maxAbsDiff(tangentResult.standardErrors, rResult.standard_errors);
    console.log(`Standard errors max abs diff: ${seDiff.toExponential(4)}`);
  }

  // Compare deviance (if GLM)
  if (!isGLMM) {
    const devDiff = Math.abs(tangentResult.deviance - rResult.deviance);
    console.log(`Deviance diff: ${devDiff.toExponential(4)} (tangent: ${tangentResult.deviance.toFixed(6)}, R: ${rResult.deviance.toFixed(6)})`);
  }

  // Compare AIC
  const aicDiff = Math.abs(tangentResult.aic - rResult.aic);
  console.log(`AIC diff: ${aicDiff.toExponential(4)} (tangent: ${tangentResult.aic.toFixed(4)}, R: ${rResult.aic.toFixed(4)})`);

  // Compare BIC
  const bicDiff = Math.abs(tangentResult.bic - rResult.bic);
  console.log(`BIC diff: ${bicDiff.toExponential(4)} (tangent: ${tangentResult.bic.toFixed(4)}, R: ${rResult.bic.toFixed(4)})`);

  // Compare log-likelihood (if GLMM)
  if (isGLMM) {
    const llDiff = Math.abs(tangentResult.logLikelihood - rResult.logLik);
    console.log(`Log-likelihood diff: ${llDiff.toExponential(4)} (tangent: ${tangentResult.logLikelihood.toFixed(4)}, R: ${rResult.logLik.toFixed(4)})`);

    // Compare variance components
    if (tangentResult.varianceComponents && rResult.variance_components) {
      console.log('\nVariance Components:');
      const vc = tangentResult.varianceComponents[0];
      const rVc = rResult.variance_components;

      if (vc && rVc.group_intercept !== undefined) {
        const vcDiff = Math.abs(vc.variance - rVc.group_intercept);
        console.log(`  Group intercept variance diff: ${vcDiff.toExponential(4)} (tangent: ${vc.variance.toFixed(6)}, R: ${rVc.group_intercept.toFixed(6)})`);
      }
    }

    // Compare random effects
    if (tangentResult.randomEffects && rResult.random_effects) {
      const reDiff = maxAbsDiff(tangentResult.randomEffects.slice(0, rResult.random_effects.length), rResult.random_effects);
      console.log(`Random effects max abs diff: ${reDiff.toExponential(4)}`);
    }
  }

  console.log(`Converged: tangent=${tangentResult.converged}, R=${rResult.converged}`);

  // Determine pass/fail
  const tolerance = isGLMM ? 0.1 : 0.01; // GLMM tolerances are looser due to different algorithms
  const passed = coefDiff < tolerance && fittedDiff < tolerance;

  console.log(`\nResult: ${passed ? '✓ PASS' : '✗ FAIL'}`);

  return passed;
}

// ==============================================================================
// Main Test Runner
// ==============================================================================

async function main() {
  try {
    // Run R script to generate reference
    await runRScript();

    // Load R results
    if (!existsSync('glm_reference_results.json')) {
      throw new Error('R reference results file not found');
    }

    const rResults = JSON.parse(readFileSync('glm_reference_results.json', 'utf-8'));

    // Run tangent-ds tests
    console.log('\n' + '='.repeat(70));
    console.log('Running tangent-ds GLM tests and comparing with R...');
    console.log('='.repeat(70));

    const tests = [
      { name: 'gaussian_glm', fn: testGaussianGLM },
      { name: 'binomial_glm', fn: testBinomialGLM },
      { name: 'poisson_glm', fn: testPoissonGLM },
      { name: 'gamma_glm', fn: testGammaGLM },
      { name: 'inverse_gaussian_glm', fn: testInverseGaussianGLM },
      { name: 'gaussian_glmm', fn: testGaussianGLMM },
      { name: 'binomial_glmm', fn: testBinomialGLMM },
      { name: 'poisson_glmm', fn: testPoissonGLMM }
    ];

    const results = [];

    for (const test of tests) {
      try {
        const tangentResult = test.fn();
        const rResult = rResults[test.name];

        if (!rResult) {
          console.log(`\nSkipping ${test.name}: No R reference found`);
          continue;
        }

        const passed = compareResults(test.name, tangentResult, rResult);
        results.push({ name: test.name, passed });
      } catch (error) {
        console.error(`\nError in ${test.name}:`, error.message);
        results.push({ name: test.name, passed: false, error: error.message });
      }
    }

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));

    const passed = results.filter(r => r.passed).length;
    const total = results.length;

    results.forEach(r => {
      const status = r.passed ? '✓' : '✗';
      const error = r.error ? ` (${r.error})` : '';
      console.log(`${status} ${r.name}${error}`);
    });

    console.log(`\nPassed: ${passed}/${total}`);

    if (passed === total) {
      console.log('\n✓ All tests passed!');
      process.exit(0);
    } else {
      console.log('\n✗ Some tests failed');
      process.exit(1);
    }
  } catch (error) {
    console.error('Error running tests:', error);
    process.exit(1);
  }
}

main();
