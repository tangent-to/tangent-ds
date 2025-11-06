/**
 * Polynomial regression
 * Extends linear regression with polynomial features
 */

import { toMatrix, Matrix } from '../core/linalg.js';
import { fitGLM as lmFit, predictGLM as lmPredict } from '../stats/glm.js';

// Minimal lm namespace for compatibility
const lm = {
  fit: (X, y, opts) => lmFit(X, y, { ...opts, family: 'gaussian' }),
  predict: (coefficients, X, opts) => {
    const model = { coefficients, family: 'gaussian', link: 'identity', intercept: opts?.intercept !== false, p: coefficients.length };
    return lmPredict(model, X, opts);
  }
};

/**
 * Create polynomial features from input
 * @param {Array<Array<number>>|Matrix} X - Input features (n × 1 for univariate)
 * @param {number} degree - Polynomial degree
 * @returns {Array<Array<number>>} Polynomial features
 */
export function polynomialFeatures(X, degree) {
  let data;
  if (Array.isArray(X)) {
    data = X;
  } else {
    const mat = toMatrix(X);
    data = [];
    for (let i = 0; i < mat.rows; i++) {
      const row = [];
      for (let j = 0; j < mat.columns; j++) {
        row.push(mat.get(i, j));
      }
      data.push(row);
    }
  }
  
  const n = data.length;
  const nFeatures = data[0].length;
  
  // For multivariate, we create all polynomial combinations
  // For now, support univariate (single feature)
  if (nFeatures !== 1) {
    throw new Error('Polynomial regression currently supports only univariate input');
  }
  
  const polyFeatures = [];
  for (let i = 0; i < n; i++) {
    const x = data[i][0];
    const row = [];
    for (let d = 1; d <= degree; d++) {
      row.push(x ** d);
    }
    polyFeatures.push(row);
  }
  
  return polyFeatures;
}

/**
 * Fit polynomial regression model
 * @param {Array<Array<number>>|Array<number>} X - Input data (can be 1D for univariate)
 * @param {Array<number>} y - Target values
 * @param {Object} options - {degree: polynomial degree, intercept: include intercept}
 * @returns {Object} {coefficients, degree, fitted, residuals, rSquared}
 */
export function fit(X, y, { degree = 2, intercept = true } = {}) {
  if (degree < 1) {
    throw new Error('Degree must be at least 1');
  }
  
  // Convert 1D array to 2D
  let inputData;
  if (Array.isArray(X) && !Array.isArray(X[0])) {
    inputData = X.map(x => [x]);
  } else {
    inputData = X;
  }
  
  // Create polynomial features
  const polyX = polynomialFeatures(inputData, degree);
  
  // Fit linear regression on polynomial features
  const model = lm.fit(polyX, y, { intercept });
  
  return {
    coefficients: model.coefficients,
    degree,
    fitted: model.fitted,
    residuals: model.residuals,
    rSquared: model.rSquared,
    adjRSquared: model.adjRSquared,
    se: model.se
  };
}

/**
 * Predict using polynomial regression model
 * @param {Object} model - Fitted model from fit()
 * @param {Array<Array<number>>|Array<number>} X - New data
 * @param {Object} options - {intercept: boolean}
 * @returns {Array<number>} Predictions
 */
export function predict(model, X, { intercept = true } = {}) {
  // Convert 1D array to 2D
  let inputData;
  if (Array.isArray(X) && !Array.isArray(X[0])) {
    inputData = X.map(x => [x]);
  } else {
    inputData = X;
  }
  
  // Create polynomial features
  const polyX = polynomialFeatures(inputData, model.degree);
  
  // Use linear regression predict
  return lm.predict(model.coefficients, polyX, { intercept });
}
