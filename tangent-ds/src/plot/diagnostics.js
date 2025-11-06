/**
 * Diagnostic Plots for GLM/GLMM
 *
 * Returns Observable Plot specifications for model diagnostics
 * These can be used directly in Observable notebooks with Plot.plot(spec)
 */

/**
 * Generate residual vs fitted plot
 * @param {Object} model - Fitted GLM model
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function residualPlot(model, options = {}) {
  if (!model.fitted) {
    throw new Error('Model must be fitted before creating diagnostic plots');
  }

  const m = model._model;
  const data = m.fitted.map((fit, i) => ({
    fitted: fit,
    residual: m.residuals[i],
    index: i
  }));

  return {
    marks: [
      {mark: "dot", data, x: "fitted", y: "residual", fill: "steelblue", fillOpacity: 0.6},
      {mark: "ruleY", y: [0], stroke: "red", strokeDasharray: "4,4"},
      {mark: "linearRegressionY", data, x: "fitted", y: "residual", stroke: "orange"}
    ],
    x: {label: "Fitted values"},
    y: {label: "Residuals"},
    grid: true,
    title: "Residuals vs Fitted",
    subtitle: "Should show random scatter around zero",
    ...options
  };
}

/**
 * Generate scale-location plot (sqrt of standardized residuals vs fitted)
 * @param {Object} model - Fitted GLM model
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function scaleLocationPlot(model, options = {}) {
  if (!model.fitted) {
    throw new Error('Model must be fitted before creating diagnostic plots');
  }

  const m = model._model;

  // Compute standardized residuals
  const stdResiduals = m.residuals.map((r, i) => {
    const se = Math.sqrt(m.dispersion || 1);
    return r / se;
  });

  const data = m.fitted.map((fit, i) => ({
    fitted: fit,
    sqrtAbsStdResid: Math.sqrt(Math.abs(stdResiduals[i])),
    index: i
  }));

  return {
    marks: [
      {mark: "dot", data, x: "fitted", y: "sqrtAbsStdResid", fill: "steelblue", fillOpacity: 0.6},
      {mark: "linearRegressionY", data, x: "fitted", y: "sqrtAbsStdResid", stroke: "red"}
    ],
    x: {label: "Fitted values"},
    y: {label: "√|Standardized residuals|"},
    grid: true,
    title: "Scale-Location",
    subtitle: "Check homoscedasticity - line should be roughly horizontal",
    ...options
  };
}

/**
 * Generate Q-Q plot for normality check
 * @param {Object} model - Fitted GLM model
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function qqPlot(model, options = {}) {
  if (!model.fitted) {
    throw new Error('Model must be fitted before creating diagnostic plots');
  }

  const m = model._model;

  // Compute standardized residuals
  const stdResiduals = m.residuals.map((r, i) => {
    const se = Math.sqrt(m.dispersion || 1);
    return r / se;
  }).sort((a, b) => a - b);

  // Generate theoretical quantiles
  const n = stdResiduals.length;
  const theoretical = stdResiduals.map((_, i) => {
    const p = (i + 0.5) / n;
    return qnorm(p);
  });

  const data = theoretical.map((t, i) => ({
    theoretical: t,
    sample: stdResiduals[i],
    index: i
  }));

  // Find range for diagonal line
  const minVal = Math.min(...theoretical, ...stdResiduals);
  const maxVal = Math.max(...theoretical, ...stdResiduals);

  return {
    marks: [
      {mark: "dot", data, x: "theoretical", y: "sample", fill: "steelblue", fillOpacity: 0.6},
      {
        mark: "line",
        data: [{x: minVal, y: minVal}, {x: maxVal, y: maxVal}],
        x: "x",
        y: "y",
        stroke: "red",
        strokeDasharray: "4,4"
      }
    ],
    x: {label: "Theoretical quantiles"},
    y: {label: "Sample quantiles"},
    grid: true,
    title: "Normal Q-Q",
    subtitle: "Points should follow the diagonal line if residuals are normal",
    ...options
  };
}

/**
 * Generate residuals vs leverage plot (Cook's distance)
 * @param {Object} model - Fitted GLM model
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function residualsLeveragePlot(model, options = {}) {
  if (!model.fitted) {
    throw new Error('Model must be fitted before creating diagnostic plots');
  }

  const m = model._model;

  // This is a simplified version - true Cook's distance requires hat matrix
  // For now, use simple metrics
  const data = m.residuals.map((r, i) => ({
    index: i,
    residual: r,
    fitted: m.fitted[i],
    absResidual: Math.abs(r)
  }));

  // Sort by absolute residual to identify potential outliers
  const sorted = [...data].sort((a, b) => b.absResidual - a.absResidual);
  const top5 = sorted.slice(0, 5);

  return {
    marks: [
      {mark: "dot", data, x: "index", y: "residual", fill: "steelblue", fillOpacity: 0.6},
      {mark: "ruleY", y: [0], stroke: "red", strokeDasharray: "4,4"},
      {
        mark: "text",
        data: top5,
        x: "index",
        y: "residual",
        text: d => d.index.toString(),
        dy: -10,
        fill: "red"
      }
    ],
    x: {label: "Observation index"},
    y: {label: "Residuals"},
    grid: true,
    title: "Residuals vs Leverage",
    subtitle: "Labeled points are potential outliers",
    ...options
  };
}

/**
 * Generate all diagnostic plots in a dashboard
 * @param {Object} model - Fitted GLM model
 * @param {Object} options - Options for individual plots
 * @returns {Array<Object>} Array of Plot specifications
 */
export function diagnosticDashboard(model, options = {}) {
  return [
    residualPlot(model, { width: options.width, height: options.height }),
    qqPlot(model, { width: options.width, height: options.height }),
    scaleLocationPlot(model, { width: options.width, height: options.height }),
    residualsLeveragePlot(model, { width: options.width, height: options.height })
  ];
}

/**
 * Generate effect plot for a specific predictor
 * @param {Object} model - Fitted GLM model
 * @param {string} variable - Variable name
 * @param {Object} data - Original data
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function effectPlot(model, variable, data, options = {}) {
  if (!model.fitted) {
    throw new Error('Model must be fitted before creating effect plots');
  }

  const { grid = 50, confidence = 0.95 } = options;

  // Find the column index for this variable
  const colIdx = model._columnsX ? model._columnsX.indexOf(variable) : -1;

  if (colIdx === -1) {
    throw new Error(`Variable ${variable} not found in model`);
  }

  // Get range of variable
  const values = data.map(d => d[variable]);
  const min = Math.min(...values);
  const max = Math.max(...values);

  // Generate grid of values
  const gridValues = Array.from({ length: grid }, (_, i) => {
    return min + (max - min) * i / (grid - 1);
  });

  // Generate predictions holding other variables at their means
  const otherMeans = model._columnsX
    .filter((_, i) => i !== colIdx)
    .map((col, i) => {
      const vals = data.map(d => d[col]);
      return vals.reduce((sum, v) => sum + v, 0) / vals.length;
    });

  const predictions = gridValues.map(val => {
    const X = model._columnsX.map((col, i) =>
      i === colIdx ? val : otherMeans[i < colIdx ? i : i - 1]
    );
    return { [variable]: val, prediction: model.predict([X])[0] };
  });

  // Add original data points
  const dataPoints = data.map(d => ({
    [variable]: d[variable],
    observed: d[model._columnY]
  }));

  return {
    marks: [
      {mark: "dot", data: dataPoints, x: variable, y: "observed", fill: "steelblue", fillOpacity: 0.3},
      {mark: "line", data: predictions, x: variable, y: "prediction", stroke: "red", strokeWidth: 2}
    ],
    x: {label: variable},
    y: {label: model._columnY || "Response"},
    grid: true,
    title: `Effect of ${variable}`,
    subtitle: "Marginal effect holding other variables constant",
    ...options
  };
}

/**
 * Generate partial residual plot (component + residual plot)
 * @param {Object} model - Fitted GLM model
 * @param {string} variable - Variable name
 * @param {Array} X - Original predictor matrix
 * @param {Object} options - Plot options
 * @returns {Object} Observable Plot specification
 */
export function partialResidualPlot(model, variable, X, options = {}) {
  if (!model.fitted) {
    throw new Error('Model must be fitted before creating partial residual plots');
  }

  const m = model._model;
  const colIdx = model._columnsX ? model._columnsX.indexOf(variable) : -1;

  if (colIdx === -1) {
    throw new Error(`Variable ${variable} not found in model`);
  }

  // Compute partial residuals: residuals + β_j * x_j
  const coefIdx = model._model.intercept ? colIdx + 1 : colIdx;
  const beta = m.coefficients[coefIdx];

  const data = X.map((row, i) => ({
    x: row[colIdx],
    partialResidual: m.residuals[i] + beta * row[colIdx],
    index: i
  }));

  return {
    marks: [
      {mark: "dot", data, x: "x", y: "partialResidual", fill: "steelblue", fillOpacity: 0.6},
      {mark: "linearRegressionY", data, x: "x", y: "partialResidual", stroke: "red"}
    ],
    x: {label: variable},
    y: {label: "Partial residual"},
    grid: true,
    title: `Partial Residual Plot: ${variable}`,
    subtitle: "Check linearity assumption for this predictor",
    ...options
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Inverse normal CDF (quantile function)
 * Approximation using Beasley-Springer-Moro algorithm
 */
function qnorm(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  const a0 = 2.50662823884;
  const a1 = -18.61500062529;
  const a2 = 41.39119773534;
  const a3 = -25.44106049637;
  const b1 = -8.47351093090;
  const b2 = 23.08336743743;
  const b3 = -21.06224101826;
  const b4 = 3.13082909833;
  const c0 = -2.78718931138;
  const c1 = -2.29796479134;
  const c2 = 4.85014127135;
  const c3 = 2.32121276858;
  const d1 = 3.54388924762;
  const d2 = 1.63706781897;

  const q = p - 0.5;

  if (Math.abs(q) <= 0.42) {
    const r = q * q;
    return q * (a0 + r * (a1 + r * (a2 + r * a3))) /
           (1 + r * (b1 + r * (b2 + r * (b3 + r * b4))));
  } else {
    let r = p;
    if (q > 0) r = 1 - p;
    r = Math.sqrt(-Math.log(r));
    const val = (c0 + r * (c1 + r * (c2 + r * c3))) /
                (1 + r * (d1 + r * d2));
    return q < 0 ? -val : val;
  }
}
