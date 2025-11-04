/**
 * Ordinary Least Squares (OLS) Linear Regression
 * Uses normal equations: β = (X'X)^-1 X'y
 */

import { Matrix, solveLeastSquares, toMatrix, pseudoInverse } from "../core/linalg.js";
import { mean, sum } from "../core/math.js";
import { prepareXY } from "../core/table.js";

/**
 * Numerical helpers
 */
function erf(x) {
  const sign = Math.sign(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const absx = Math.abs(x);
  const t = 1 / (1 + p * absx);
  const poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
  return sign * (1 - poly * Math.exp(-absx * absx));
}

function incompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x ** a * (1 - x) ** b;
}

function studentTCdf(t, df) {
  if (df > 30) {
    return 0.5 * (1 + erf(t / Math.sqrt(2)));
  }
  const x = df / (df + t * t);
  return 1 - 0.5 * incompleteBeta(x, df / 2, 0.5);
}

function defaultFeatureNames(count, includeIntercept) {
  return Array.from({ length: count }, (_, idx) => {
    if (includeIntercept && idx === 0) return "Intercept";
    const baseIndex = includeIntercept ? idx : idx + 1;
    return `x${baseIndex}`;
  });
}

/**
 * Fit OLS linear regression
 * Supports two calling styles:
 *  - Array/matrix style (backward compatible): fit(X, y, { intercept })
 *  - Declarative table style: fit({ X: 'col' | ['col1','col2'], y: 'target', data, omit_missing, intercept })
 *
 * @param {Array<Array<number>>|Matrix|Object} X - Design matrix or options object when using table style
 * @param {Array<number>|string} y - Response vector or (when using array/matrix style) ignored if options object provided
 * @param {Object} options - { intercept: boolean, omit_missing: boolean, featureNames?: Array<string> }
 * @returns {Object} model object with coefficients, fitted values, residuals and coefficient statistics
 */
export function fit(
  X,
  y,
  { intercept = true, omit_missing = true, featureNames = null } = {},
) {
  let columnNames = featureNames ? featureNames.slice() : null;

  // Declarative table-style API
  if (
    X && typeof X === "object" && !Array.isArray(X) && "X" in X && "y" in X &&
    "data" in X
  ) {
    const prepared = prepareXY({
      X: X.X,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined
        ? X.omit_missing
        : omit_missing,
    });
    y = prepared.y;
    X = prepared.X;
    columnNames = prepared.columnsX ? prepared.columnsX.slice() : columnNames;
    if (X.intercept !== undefined) {
      intercept = X.intercept;
    }
  }

  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;
  const responseVector = Array.isArray(y) ? y : Array.from(y);

  if (n !== responseVector.length) {
    throw new Error("X and y must have same number of rows");
  }

  // Add intercept column if requested
  if (intercept) {
    const withIntercept = [];
    for (let i = 0; i < n; i++) {
      const row = [1];
      for (let j = 0; j < designMatrix.columns; j++) {
        row.push(designMatrix.get(i, j));
      }
      withIntercept.push(row);
    }
    designMatrix = new Matrix(withIntercept);
    if (columnNames) {
      columnNames = ["Intercept", ...columnNames];
    }
  }

  if (!columnNames) {
    columnNames = defaultFeatureNames(designMatrix.columns, intercept);
  }

  // Solve using least squares
  const coeffMatrix = solveLeastSquares(designMatrix, responseVector);
  const coefficients = coeffMatrix.to1DArray();

  // Compute fitted values
  const fittedMatrix = designMatrix.mmul(coeffMatrix);
  const fitted = fittedMatrix.to1DArray();

  // Residuals
  const residuals = responseVector.map((yi, i) => yi - fitted[i]);

  // R-squared metrics
  const yMean = mean(responseVector);
  const sst = sum(responseVector.map((yi) => (yi - yMean) ** 2));
  const sse = sum(residuals.map((r) => r ** 2));
  const rSquared = 1 - sse / sst;

  const p = coefficients.length;
  const adjRSquared = 1 - (1 - rSquared) * (n - 1) / (n - p);
  const dfResidual = n - p;
  const regressionSE = Math.sqrt(sse / dfResidual);

  // Coefficient covariance and statistics
  let covariance = null;
  let standardErrors = null;
  let tStatistics = null;
  let pValues = null;

  if (n > p) {
    const XtX = designMatrix.transpose().mmul(designMatrix);
    let XtXInv;
    try {
      XtXInv = XtX.inverse();
    } catch (err) {
      XtXInv = pseudoInverse(XtX);
    }

    const sigmaSquared = regressionSE ** 2;
    covariance = XtXInv.mul(sigmaSquared).to2DArray();
    standardErrors = covariance.map((row, idx) => Math.sqrt(Math.max(row[idx], 0)));
    tStatistics = coefficients.map((coef, idx) => standardErrors[idx] > 0 ? coef / standardErrors[idx] : Number.NaN);
    pValues = tStatistics.map((t) => {
      if (!Number.isFinite(t)) return Number.NaN;
      const cdf = studentTCdf(Math.abs(t), dfResidual);
      return 2 * (1 - cdf);
    });
  }

  return {
    coefficients,
    featureNames: columnNames,
    fitted,
    residuals,
    rSquared,
    adjRSquared,
    residualStandardError: regressionSE,
    se: regressionSE,
    covariance,
    standardErrors,
    tStatistics,
    pValues,
    n,
    p,
    dfResidual,
  };
}

/**
 * Predict using fitted model
 */
export function predict(coefficients, X, { intercept = true } = {}) {
  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;

  if (intercept) {
    const withIntercept = [];
    for (let i = 0; i < n; i++) {
      const row = [1];
      for (let j = 0; j < designMatrix.columns; j++) {
        row.push(designMatrix.get(i, j));
      }
      withIntercept.push(row);
    }
    designMatrix = new Matrix(withIntercept);
  }

  if (designMatrix.columns !== coefficients.length) {
    throw new Error(
      `Coefficient length (${coefficients.length}) must match design matrix columns (${designMatrix.columns})`,
    );
  }

  const coeffMatrix = Matrix.columnVector(coefficients);
  const predictions = designMatrix.mmul(coeffMatrix);
  return predictions.to1DArray();
}

/**
 * Summary statistics for regression
 */
export function summary(model) {
  const {
    coefficients,
    featureNames,
    rSquared,
    adjRSquared,
    residualStandardError,
    standardErrors,
    tStatistics,
    pValues,
    n,
    p,
    dfResidual,
  } = model;

  const coefficientTable = coefficients.map((coef, idx) => ({
    term: featureNames ? featureNames[idx] : `x${idx + 1}`,
    estimate: coef,
    stdError: standardErrors ? standardErrors[idx] : Number.NaN,
    tStatistic: tStatistics ? tStatistics[idx] : Number.NaN,
    pValue: pValues ? pValues[idx] : Number.NaN,
  }));
  const hasIntercept = featureNames && featureNames[0] === 'Intercept';
  const dfModel = hasIntercept ? Math.max(p - 1, 0) : Math.max(p, 0);
  const fStatistic = (dfModel > 0 && dfResidual > 0)
    ? (rSquared / dfModel) / ((1 - rSquared) / dfResidual)
    : Number.NaN;

  return {
    coefficients,
    featureNames,
    coefficientTable,
    nObservations: n,
    nPredictors: p,
    dfResidual,
    rSquared,
    adjRSquared,
    residualStandardError,
    fStatistic,
  };
}
