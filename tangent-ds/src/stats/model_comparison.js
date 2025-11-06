/**
 * Model Comparison Utilities
 *
 * Functions for comparing multiple GLM/GLMM models
 * Returns Observable-friendly tables and specifications
 */

/**
 * Compare multiple models with information criteria
 * @param {Array<Object>} models - Array of fitted GLM models with optional names
 * @param {Object} options - Comparison options
 * @returns {Object} Comparison table and best model info
 */
export function compareModels(models, options = {}) {
  const {
    criterion = 'aic', // 'aic', 'bic', or 'both'
    sort = true
  } = options;

  if (!Array.isArray(models) || models.length === 0) {
    throw new Error('compareModels requires an array of fitted models');
  }

  // Extract model statistics
  const comparison = models.map((model, i) => {
    if (!model.fitted) {
      throw new Error(`Model ${i} must be fitted before comparison`);
    }

    const m = model._model;
    const name = model._name || model._formula || `Model ${i + 1}`;

    return {
      name,
      formula: model._formula || 'N/A',
      family: model.params.family,
      n: m.n,
      df: m.df,
      logLik: m.logLikelihood,
      aic: m.aic,
      bic: m.bic,
      deviance: m.deviance,
      nullDeviance: m.nullDeviance,
      pseudoR2: m.pseudoR2,
      model // Keep reference for downstream use
    };
  });

  // Sort by criterion if requested
  if (sort) {
    const sortKey = criterion === 'both' ? 'aic' : criterion;
    comparison.sort((a, b) => a[sortKey] - b[sortKey]);
  }

  // Compute delta values (difference from best model)
  const bestAIC = Math.min(...comparison.map(m => m.aic));
  const bestBIC = Math.min(...comparison.map(m => m.bic));

  comparison.forEach(m => {
    m.deltaAIC = m.aic - bestAIC;
    m.deltaBIC = m.bic - bestBIC;
    m.aicWeight = computeAICWeight(m.deltaAIC);
  });

  // Identify best model
  const bestModel = criterion === 'bic'
    ? comparison.find(m => m.bic === bestBIC)
    : comparison.find(m => m.aic === bestAIC);

  return {
    comparison,
    best: {
      name: bestModel.name,
      criterion,
      value: bestModel[criterion]
    },
    table: formatComparisonTable(comparison, criterion)
  };
}

/**
 * Compute Akaike weight for model selection
 * Higher weights indicate better support for the model
 */
function computeAICWeight(deltaAIC) {
  return Math.exp(-0.5 * deltaAIC);
}

/**
 * Format comparison results as a table object for Observable
 */
function formatComparisonTable(comparison, criterion) {
  return comparison.map(m => ({
    Model: m.name,
    Formula: m.formula,
    Family: m.family,
    'Log-Lik': m.logLik.toFixed(2),
    'AIC': m.aic.toFixed(2),
    'Δ AIC': m.deltaAIC.toFixed(2),
    'AIC Weight': m.aicWeight.toFixed(3),
    'BIC': m.bic.toFixed(2),
    'Δ BIC': m.deltaBIC.toFixed(2),
    'Pseudo-R²': m.pseudoR2.toFixed(3),
    'df': m.df
  }));
}

/**
 * Perform likelihood ratio test for nested models
 * @param {Object} model1 - Smaller (nested) model
 * @param {Object} model2 - Larger model
 * @param {Object} options - Test options
 * @returns {Object} Test results
 */
export function likelihoodRatioTest(model1, model2, options = {}) {
  if (!model1.fitted || !model2.fitted) {
    throw new Error('Both models must be fitted before comparison');
  }

  const m1 = model1._model;
  const m2 = model2._model;

  // Check if models are nested (based on df)
  if (m1.df >= m2.df) {
    throw new Error('Model 1 should be nested in Model 2 (fewer parameters)');
  }

  // Compute likelihood ratio test statistic
  const lrStatistic = 2 * (m2.logLikelihood - m1.logLikelihood);
  const dfDiff = m2.df - m1.df;

  // Compute p-value from chi-square distribution
  const pValue = 1 - chiSquareCDF(lrStatistic, dfDiff);

  return {
    statistic: lrStatistic,
    df: dfDiff,
    pValue,
    significant: pValue < 0.05,
    model1: {
      name: model1._name || model1._formula || 'Model 1',
      logLik: m1.logLikelihood,
      df: m1.df
    },
    model2: {
      name: model2._name || model2._formula || 'Model 2',
      logLik: m2.logLikelihood,
      df: m2.df
    },
    summary: formatLRTSummary(lrStatistic, dfDiff, pValue)
  };
}

/**
 * Format LRT results as string
 */
function formatLRTSummary(statistic, df, pValue) {
  const sigCode = pValue < 0.001 ? '***' : pValue < 0.01 ? '**' : pValue < 0.05 ? '*' : pValue < 0.1 ? '.' : '';

  return `
Likelihood Ratio Test
─────────────────────
LR statistic: ${statistic.toFixed(4)}
df: ${df}
p-value: ${pValue.toExponential(4)} ${sigCode}

Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
`.trim();
}

/**
 * Compare multiple models and perform pairwise LRT
 * @param {Array<Object>} models - Array of fitted models (should be nested)
 * @param {Object} options - Options
 * @returns {Object} Matrix of pairwise comparisons
 */
export function pairwiseLRT(models, options = {}) {
  if (!Array.isArray(models) || models.length < 2) {
    throw new Error('pairwiseLRT requires at least 2 models');
  }

  const n = models.length;
  const results = [];

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const m1 = models[i];
      const m2 = models[j];

      // Determine which model is nested (has fewer parameters)
      const df1 = m1._model.df;
      const df2 = m2._model.df;

      let nested, full;
      if (df1 < df2) {
        nested = m1;
        full = m2;
      } else {
        nested = m2;
        full = m1;
      }

      try {
        const test = likelihoodRatioTest(nested, full);
        results.push({
          model1: i,
          model2: j,
          nested: nested === m1 ? i : j,
          full: full === m1 ? i : j,
          ...test
        });
      } catch (e) {
        // Models not nested, skip
        continue;
      }
    }
  }

  return {
    results,
    table: formatPairwiseTable(results, models)
  };
}

/**
 * Format pairwise LRT results as table
 */
function formatPairwiseTable(results, models) {
  return results.map(r => ({
    'Nested Model': models[r.nested]._name || models[r.nested]._formula || `Model ${r.nested + 1}`,
    'Full Model': models[r.full]._name || models[r.full]._formula || `Model ${r.full + 1}`,
    'LR Statistic': r.statistic.toFixed(4),
    'df': r.df,
    'p-value': r.pValue.toExponential(4),
    'Significant': r.significant ? 'Yes' : 'No'
  }));
}

/**
 * Generate model selection plot specification
 * @param {Array<Object>} models - Array of fitted models
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function modelSelectionPlot(models, options = {}) {
  const {
    criterion = 'aic',
    width = 640,
    height = 400
  } = options;

  const comparison = compareModels(models, { criterion, sort: true });

  const data = comparison.comparison.map((m, i) => ({
    model: m.name,
    value: m[criterion],
    delta: criterion === 'aic' ? m.deltaAIC : m.deltaBIC,
    weight: m.aicWeight,
    rank: i + 1
  }));

  // Main criterion plot
  const criterionPlot = {
    marks: [
      {
        mark: "barY",
        data,
        x: "model",
        y: "value",
        fill: "steelblue",
        tip: true
      },
      {
        mark: "ruleY",
        y: [Math.min(...data.map(d => d.value))],
        stroke: "red",
        strokeDasharray: "4,4"
      }
    ],
    x: { label: "Model", tickRotate: -45 },
    y: { label: criterion.toUpperCase(), grid: true },
    marginBottom: 80,
    width,
    height: height / 2,
    title: `Model Comparison by ${criterion.toUpperCase()}`,
    subtitle: "Lower values indicate better fit"
  };

  // Delta plot (difference from best)
  const deltaPlot = {
    marks: [
      {
        mark: "barY",
        data,
        x: "model",
        y: "delta",
        fill: d => d.delta < 2 ? "green" : d.delta < 7 ? "orange" : "red",
        tip: true
      },
      {
        mark: "ruleY",
        y: [0, 2, 7],
        stroke: ["red", "orange", "gold"],
        strokeDasharray: "4,4"
      }
    ],
    x: { label: "Model", tickRotate: -45 },
    y: { label: `Δ ${criterion.toUpperCase()}`, grid: true },
    marginBottom: 80,
    width,
    height: height / 2,
    title: `Model Uncertainty`,
    subtitle: "Δ < 2: substantial support; 2-7: some support; > 7: little support"
  };

  return {
    criterion: criterionPlot,
    delta: deltaPlot,
    data,
    best: comparison.best
  };
}

/**
 * Generate AIC weight visualization
 * @param {Array<Object>} models - Array of fitted models
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function aicWeightPlot(models, options = {}) {
  const { width = 640, height = 300 } = options;

  const comparison = compareModels(models, { sort: true });
  const data = comparison.comparison.map(m => ({
    model: m.name,
    weight: m.aicWeight,
    normalizedWeight: m.aicWeight / comparison.comparison.reduce((sum, c) => sum + c.aicWeight, 0)
  }));

  return {
    marks: [
      {
        mark: "barY",
        data,
        x: "model",
        y: "normalizedWeight",
        fill: "steelblue",
        tip: true
      }
    ],
    x: { label: "Model", tickRotate: -45 },
    y: { label: "Normalized AIC Weight", grid: true, domain: [0, 1] },
    marginBottom: 80,
    width,
    height,
    title: "Model Evidence (AIC Weights)",
    subtitle: "Relative likelihood each model is the best model"
  };
}

/**
 * Generate coefficient comparison plot across models
 * @param {Array<Object>} models - Array of fitted models
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function coefficientComparisonPlot(models, options = {}) {
  const { width = 640, height = 400 } = options;

  // Collect all coefficients across models
  const allData = [];

  models.forEach((model, modelIdx) => {
    const m = model._model;
    const name = model._name || model._formula || `Model ${modelIdx + 1}`;

    if (!m.coefficientNames || !m.coefficients) return;

    m.coefficientNames.forEach((coefName, i) => {
      const ci = m.confidenceIntervals ? m.confidenceIntervals[i] : null;

      allData.push({
        model: name,
        coefficient: coefName,
        estimate: m.coefficients[i],
        lower: ci ? ci.lower : null,
        upper: ci ? ci.upper : null,
        se: m.standardErrors ? m.standardErrors[i] : null
      });
    });
  });

  return {
    marks: [
      {
        mark: "ruleY",
        y: [0],
        stroke: "red",
        strokeDasharray: "4,4"
      },
      {
        mark: "errorbarX",
        data: allData.filter(d => d.lower !== null),
        x1: "lower",
        x2: "upper",
        y: "coefficient",
        stroke: "model",
        strokeWidth: 2
      },
      {
        mark: "dot",
        data: allData,
        x: "estimate",
        y: "coefficient",
        fill: "model",
        r: 4,
        tip: true
      }
    ],
    x: { label: "Coefficient estimate", grid: true },
    y: { label: "Coefficient" },
    color: { legend: true },
    width,
    height,
    title: "Coefficient Comparison Across Models",
    subtitle: "Points show estimates, bars show 95% confidence intervals"
  };
}

/**
 * K-fold cross-validation for model selection
 * @param {Function} modelFactory - Function that creates and returns a model
 * @param {Array} X - Predictor matrix
 * @param {Array} y - Response variable
 * @param {Object} options - CV options
 * @returns {Object} Cross-validation results
 */
export function crossValidate(modelFactory, X, y, options = {}) {
  const {
    k = 5,
    shuffle = true,
    metric = 'mse', // 'mse', 'mae', 'deviance'
    seed = null
  } = options;

  const n = y.length;
  const foldSize = Math.floor(n / k);

  // Create fold indices
  let indices = Array.from({ length: n }, (_, i) => i);
  if (shuffle) {
    indices = shuffleArray(indices, seed);
  }

  const folds = [];
  for (let i = 0; i < k; i++) {
    const start = i * foldSize;
    const end = i === k - 1 ? n : (i + 1) * foldSize;
    folds.push(indices.slice(start, end));
  }

  // Perform cross-validation
  const scores = [];

  for (let i = 0; i < k; i++) {
    const testIndices = folds[i];
    const trainIndices = indices.filter(idx => !testIndices.includes(idx));

    // Split data
    const Xtrain = trainIndices.map(idx => X[idx]);
    const ytrain = trainIndices.map(idx => y[idx]);
    const Xtest = testIndices.map(idx => X[idx]);
    const ytest = testIndices.map(idx => y[idx]);

    // Fit model
    const model = modelFactory();
    model.fit(Xtrain, ytrain);

    // Predict
    const predictions = model.predict(Xtest);

    // Compute score
    const score = computeMetric(ytest, predictions, metric);
    scores.push(score);
  }

  return {
    scores,
    mean: scores.reduce((a, b) => a + b, 0) / scores.length,
    std: Math.sqrt(scores.reduce((sum, s) => sum + Math.pow(s - scores.reduce((a, b) => a + b, 0) / scores.length, 2), 0) / scores.length),
    metric,
    k
  };
}

/**
 * Compare models using cross-validation
 * @param {Array<Function>} modelFactories - Array of model factory functions
 * @param {Array} X - Predictor matrix
 * @param {Array} y - Response variable
 * @param {Object} options - CV options
 * @returns {Object} CV comparison results
 */
export function crossValidateModels(modelFactories, X, y, options = {}) {
  const results = modelFactories.map((factory, i) => {
    const cv = crossValidate(factory, X, y, options);
    return {
      model: `Model ${i + 1}`,
      ...cv
    };
  });

  // Sort by mean score (lower is better for MSE/MAE)
  results.sort((a, b) => a.mean - b.mean);

  return {
    results,
    best: results[0],
    table: results.map(r => ({
      Model: r.model,
      'Mean Score': r.mean.toFixed(4),
      'Std Dev': r.std.toFixed(4),
      'Min': Math.min(...r.scores).toFixed(4),
      'Max': Math.max(...r.scores).toFixed(4)
    }))
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Chi-square CDF approximation
 */
function chiSquareCDF(x, df) {
  if (x <= 0) return 0;

  // Use gamma function approximation
  const k = df / 2;
  const z = x / 2;

  return lowerIncompleteGamma(k, z) / gamma(k);
}

/**
 * Gamma function (Stirling's approximation)
 */
function gamma(z) {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }

  z -= 1;
  const g = 7;
  const coefficients = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  let x = coefficients[0];
  for (let i = 1; i < g + 2; i++) {
    x += coefficients[i] / (z + i);
  }

  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

/**
 * Lower incomplete gamma function (series approximation)
 */
function lowerIncompleteGamma(s, x) {
  if (x === 0) return 0;
  if (x < 0 || s <= 0) return NaN;

  let sum = 0;
  let term = 1 / s;
  let n = 0;

  while (Math.abs(term) > 1e-10 && n < 100) {
    sum += term;
    n++;
    term *= x / (s + n);
  }

  return Math.pow(x, s) * Math.exp(-x) * sum;
}

/**
 * Shuffle array with optional seed
 */
function shuffleArray(array, seed = null) {
  const arr = [...array];
  let random = seed !== null ? seededRandom(seed) : Math.random;

  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }

  return arr;
}

/**
 * Seeded random number generator (simple LCG)
 */
function seededRandom(seed) {
  let state = seed;
  return function() {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

/**
 * Compute prediction metric
 */
function computeMetric(y, yhat, metric) {
  const n = y.length;

  switch (metric) {
    case 'mse':
      return y.reduce((sum, yi, i) => sum + Math.pow(yi - yhat[i], 2), 0) / n;

    case 'mae':
      return y.reduce((sum, yi, i) => sum + Math.abs(yi - yhat[i]), 0) / n;

    case 'rmse':
      return Math.sqrt(y.reduce((sum, yi, i) => sum + Math.pow(yi - yhat[i], 2), 0) / n);

    case 'deviance':
      // For Gaussian, deviance = sum of squared residuals
      return y.reduce((sum, yi, i) => sum + Math.pow(yi - yhat[i], 2), 0);

    default:
      throw new Error(`Unknown metric: ${metric}`);
  }
}
