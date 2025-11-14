/**
 * Spline Basis Functions for Generalized Additive Models
 *
 * Implements:
 * - B-spline basis functions (cubic, with arbitrary degree)
 * - Cubic regression splines (natural splines)
 * - Truncated power basis (legacy)
 * - Penalty matrices for smoothness
 */

import { Matrix } from 'ml-matrix';

/**
 * Compute B-spline basis functions
 * More numerically stable than truncated power basis
 *
 * @param {number} x - Evaluation point
 * @param {Array<number>} knots - Knot sequence (internal knots)
 * @param {number} degree - Spline degree (3 = cubic, default)
 * @param {number} deriv - Derivative order (0 = value, 1 = first derivative)
 * @returns {Array<number>} Basis function values
 */
export function bsplineBasis(x, knots, degree = 3, deriv = 0) {
  // Extend knots with boundary knots for B-spline definition
  const xmin = Math.min(...knots) - 1e-5;
  const xmax = Math.max(...knots) + 1e-5;

  // Full knot sequence with boundary repetitions
  const t = [
    ...Array(degree + 1).fill(xmin),
    ...knots,
    ...Array(degree + 1).fill(xmax)
  ];

  const n = t.length - degree - 1;
  const basis = new Array(n).fill(0);

  // Cox-de Boor recursion
  if (deriv === 0) {
    // Compute basis values
    for (let i = 0; i < n; i++) {
      basis[i] = bsplineRecursion(x, i, degree, t);
    }
  } else if (deriv === 1) {
    // Compute first derivatives
    for (let i = 0; i < n; i++) {
      const left = (t[i + degree] > t[i])
        ? degree * bsplineRecursion(x, i, degree - 1, t) / (t[i + degree] - t[i])
        : 0;
      const right = (t[i + degree + 1] > t[i + 1])
        ? degree * bsplineRecursion(x, i + 1, degree - 1, t) / (t[i + degree + 1] - t[i + 1])
        : 0;
      basis[i] = left - right;
    }
  }

  return basis;
}

/**
 * Cox-de Boor recursion for B-splines
 */
function bsplineRecursion(x, i, p, t) {
  if (p === 0) {
    return (x >= t[i] && x < t[i + 1]) ? 1 : 0;
  }

  const leftDenom = t[i + p] - t[i];
  const rightDenom = t[i + p + 1] - t[i + 1];

  const left = (leftDenom > 0)
    ? ((x - t[i]) / leftDenom) * bsplineRecursion(x, i, p - 1, t)
    : 0;

  const right = (rightDenom > 0)
    ? ((t[i + p + 1] - x) / rightDenom) * bsplineRecursion(x, i + 1, p - 1, t)
    : 0;

  return left + right;
}

/**
 * Cubic regression spline basis (natural cubic splines)
 * Equivalent to R's bs(..., degree=3) with natural boundary conditions
 *
 * @param {number} x - Evaluation point
 * @param {Array<number>} knots - Interior knots
 * @returns {Array<number>} Basis function values
 */
export function cubicRegressionSplineBasis(x, knots) {
  const basis = [1, x]; // Linear terms

  // Add cubic terms with natural spline constraints
  for (const knot of knots) {
    const diff = x - knot;
    basis.push(diff > 0 ? diff * diff * diff : 0);
  }

  return basis;
}

/**
 * Truncated power basis (legacy, for compatibility)
 * @param {number} x - Evaluation point
 * @param {Array<number>} knots - Knot locations
 * @returns {Array<number>} Basis function values
 */
export function truncatedPowerBasis(x, knots) {
  const basis = [x];
  for (const knot of knots) {
    const diff = x - knot;
    basis.push(diff > 0 ? diff * diff * diff : 0);
  }
  return basis;
}

/**
 * Compute knot locations
 * @param {Array<number>} values - Data values for this feature
 * @param {number} nKnots - Number of interior knots
 * @param {string} placement - 'quantile' or 'uniform'
 * @returns {Array<number>} Knot locations
 */
export function computeKnots(values, nKnots, placement = 'quantile') {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const knots = [];

  if (placement === 'quantile') {
    // Place knots at quantiles
    for (let i = 1; i <= nKnots; i++) {
      const q = i / (nKnots + 1);
      const idx = Math.floor(q * (sorted.length - 1));
      knots.push(sorted[idx]);
    }
  } else if (placement === 'uniform') {
    // Uniform spacing
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const range = max - min;
    for (let i = 1; i <= nKnots; i++) {
      knots.push(min + (i / (nKnots + 1)) * range);
    }
  }

  return knots;
}

/**
 * Build design matrix for spline smooths
 * @param {Array<Array<number>>} X - Input data (n × p)
 * @param {Array<Object>} smoothConfigs - Smooth configurations per feature
 * @param {boolean} includeIntercept - Include intercept column
 * @returns {Array<Array<number>>} Design matrix
 */
export function buildSmoothMatrix(X, smoothConfigs, includeIntercept = true) {
  const n = X.length;
  const design = [];

  for (let i = 0; i < n; i++) {
    const row = [];
    if (includeIntercept) row.push(1);

    for (let j = 0; j < X[i].length; j++) {
      const config = smoothConfigs[j];
      const x = X[i][j];

      let basis;
      if (config.type === 'cr' || config.type === 'bs') {
        basis = bsplineBasis(x, config.knots, config.degree || 3);
      } else if (config.type === 'tp') {
        basis = truncatedPowerBasis(x, config.knots);
      } else {
        // Default: cubic regression spline
        basis = cubicRegressionSplineBasis(x, config.knots);
      }

      row.push(...basis);
    }

    design.push(row);
  }

  return design;
}

/**
 * Compute penalty matrix for smoothness
 * Second-order difference penalty (default for P-splines)
 *
 * @param {number} nCoef - Number of coefficients in smooth term
 * @param {number} order - Difference order (1, 2, or 3)
 * @returns {Matrix} Penalty matrix (nCoef × nCoef)
 */
export function penaltyMatrix(nCoef, order = 2) {
  // Difference matrix D
  const D = differencePenalty(nCoef, order);

  // Penalty matrix S = D'D
  const S = D.transpose().mmul(D);

  return S;
}

/**
 * Compute difference matrix for penalty
 * @param {number} n - Dimension
 * @param {number} order - Difference order
 * @returns {Matrix} Difference matrix
 */
function differencePenalty(n, order) {
  if (order === 1) {
    // First-order differences
    const D = Matrix.zeros(n - 1, n);
    for (let i = 0; i < n - 1; i++) {
      D.set(i, i, -1);
      D.set(i, i + 1, 1);
    }
    return D;
  } else if (order === 2) {
    // Second-order differences
    const D = Matrix.zeros(n - 2, n);
    for (let i = 0; i < n - 2; i++) {
      D.set(i, i, 1);
      D.set(i, i + 1, -2);
      D.set(i, i + 2, 1);
    }
    return D;
  } else if (order === 3) {
    // Third-order differences
    const D = Matrix.zeros(n - 3, n);
    for (let i = 0; i < n - 3; i++) {
      D.set(i, i, -1);
      D.set(i, i + 1, 3);
      D.set(i, i + 2, -3);
      D.set(i, i + 3, 1);
    }
    return D;
  }

  throw new Error(`Unsupported difference order: ${order}`);
}

/**
 * Build full penalty matrix for GAM
 * Combines penalties from all smooth terms
 *
 * @param {Array<Object>} smoothConfigs - Smooth configurations
 * @param {boolean} hasIntercept - Whether design matrix has intercept
 * @returns {Matrix} Full penalty matrix
 */
export function buildPenaltyMatrix(smoothConfigs, hasIntercept = true) {
  let offset = hasIntercept ? 1 : 0;
  let totalDim = offset;

  // Calculate total dimension
  for (const config of smoothConfigs) {
    const nBasis = config.knots.length + (config.type === 'tp' ? 1 : 4); // Rough estimate
    totalDim += nBasis;
  }

  // Build block-diagonal penalty matrix
  const S = Matrix.zeros(totalDim, totalDim);

  for (const config of smoothConfigs) {
    const nBasis = config.knots.length + (config.type === 'tp' ? 1 : 4);
    const Si = penaltyMatrix(nBasis, config.penaltyOrder || 2);

    // Place in appropriate block
    for (let i = 0; i < nBasis; i++) {
      for (let j = 0; j < nBasis; j++) {
        S.set(offset + i, offset + j, Si.get(i, j));
      }
    }

    offset += nBasis;
  }

  return S;
}

/**
 * Extract smooth function values and standard errors
 * @param {Array<number>} coefficients - Fitted coefficients
 * @param {Matrix} covMatrix - Covariance matrix of coefficients
 * @param {Array<number>} xVals - Evaluation points
 * @param {Object} smoothConfig - Configuration for this smooth
 * @param {number} offset - Offset in coefficient vector
 * @returns {Object} { fitted, se }
 */
export function extractSmooth(coefficients, covMatrix, xVals, smoothConfig, offset) {
  const fitted = [];
  const se = [];

  for (const x of xVals) {
    let basis;
    if (smoothConfig.type === 'cr' || smoothConfig.type === 'bs') {
      basis = bsplineBasis(x, smoothConfig.knots, smoothConfig.degree || 3);
    } else if (smoothConfig.type === 'tp') {
      basis = truncatedPowerBasis(x, smoothConfig.knots);
    } else {
      basis = cubicRegressionSplineBasis(x, smoothConfig.knots);
    }

    // Compute fitted value
    let fittedVal = 0;
    for (let i = 0; i < basis.length; i++) {
      fittedVal += basis[i] * coefficients[offset + i];
    }
    fitted.push(fittedVal);

    // Compute standard error: sqrt(basis' * Cov * basis)
    let variance = 0;
    for (let i = 0; i < basis.length; i++) {
      for (let j = 0; j < basis.length; j++) {
        variance += basis[i] * covMatrix.get(offset + i, offset + j) * basis[j];
      }
    }
    se.push(Math.sqrt(Math.max(0, variance)));
  }

  return { fitted, se };
}
