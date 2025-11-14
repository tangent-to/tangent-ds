/**
 * Generalized Additive Models (regression & classification) using
 * penalized spline basis functions with automatic smoothness selection.
 *
 * Implements mgcv-like functionality:
 * - Multiple basis types (B-spline, cubic regression spline, truncated power)
 * - Automatic smoothness selection (GCV, REML)
 * - Statistical inference (EDF, p-values, confidence intervals)
 */

import { Regressor, Classifier } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';
import { fitGLM, predictGLM } from '../../stats/glm.js';
import { Matrix } from 'ml-matrix';
import {
  computeKnots,
  bsplineBasis,
  cubicRegressionSplineBasis,
  truncatedPowerBasis,
  penaltyMatrix,
  buildSmoothMatrix
} from '../splines.js';
import {
  fitPenalizedRegression,
  optimizeSmoothness,
  computeSmoothEDF,
  computeSmoothPValues,
  computeConfidenceIntervals,
  createGAMSummary
} from '../gam_utils.js';

// Minimal lm/logit namespaces for compatibility
const lm = {
  fit: (X, y, opts) => fitGLM(X, y, { ...opts, family: 'gaussian' }),
  predict: (coefficients, X, opts) => {
    const model = { coefficients, family: 'gaussian', link: 'identity', intercept: opts?.intercept !== false, p: coefficients.length };
    return predictGLM(model, X, opts);
  },
  summary: (model) => ({
    coefficients: model.coefficients,
    rSquared: model.pseudoR2 || model.rSquared,
    adjRSquared: model.adjRSquared
  })
};

const logit = {
  fit: (X, y, opts) => fitGLM(X, y, { ...opts, family: 'binomial', link: 'logit' }),
  predict: (coefficients, X, opts) => {
    const model = { coefficients, family: 'binomial', link: 'logit', intercept: opts?.intercept !== false, p: coefficients.length };
    return predictGLM(model, X, opts);
  }
};

function toNumericMatrix(X) {
  return X.map((row) => Array.isArray(row) ? row.map(Number) : [Number(row)]);
}

function prepareDataset(X, y) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareXY({
      X: X.X || X.columns,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return {
      X: toNumericMatrix(prepared.X),
      y: Array.isArray(prepared.y) ? prepared.y.slice() : Array.from(prepared.y),
      columns: prepared.columnsX
    };
  }
  return {
    X: toNumericMatrix(X),
    y: Array.isArray(y) ? y.slice() : Array.from(y),
    columns: null
  };
}

function preparePredictInput(X, columns) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || columns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

function computeKnots(values, nSplines) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const knots = [];
  for (let i = 1; i < nSplines; i++) {
    const idx = Math.floor((i / (nSplines + 1)) * (sorted.length - 1));
    knots.push(sorted[idx]);
  }
  return knots;
}

function splineBasis(value, knots) {
  const basis = [value];
  for (const knot of knots) {
    const diff = value - knot;
    basis.push(diff > 0 ? diff * diff * diff : 0);
  }
  return basis;
}

function buildDesignMatrix(X, featureConfigs, includeIntercept = true) {
  const design = [];
  for (const row of X) {
    const features = [];
    if (includeIntercept) features.push(1);
    for (let j = 0; j < row.length; j++) {
      const basis = splineBasis(row[j], featureConfigs[j]);
      features.push(...basis);
    }
    design.push(features);
  }
  return design;
}

class GAMBase {
  constructor({
    nSplines = 4,
    basis = 'tp',  // 'bs', 'cr', 'tp' (B-spline, cubic regression, truncated power) - default 'tp' for backward compat
    smoothMethod = null,  // 'GCV', 'REML', or null for no penalty (default: null for backward compat)
    lambda = null,  // Fixed smoothing parameter (null = no penalty if smoothMethod is null)
    penaltyOrder = 2,  // Order of difference penalty (1, 2, or 3)
    knotPlacement = 'quantile',  // 'quantile' or 'uniform'
    task = 'regression',
    maxIter = 100,
    tol = 1e-6
  } = {}) {
    // Legacy compatibility
    if (typeof nSplines === 'number' && nSplines < 10) {
      // Likely old API usage, keep it
    }

    this.nSplines = nSplines;
    this.basis = basis;
    this.smoothMethod = smoothMethod;
    this.lambda = lambda;
    this.penaltyOrder = penaltyOrder;
    this.knotPlacement = knotPlacement;
    this.task = task;
    this.maxIter = maxIter;
    this.tol = tol;

    // Model components
    this.smoothConfigs = null;
    this.columns = null;
    this.coef = null;
    this.classMap = null;

    // Statistical inference
    this.edf = null;  // Total effective degrees of freedom
    this.smoothEDFs = null;  // EDF per smooth term
    this.smoothPValues = null;  // P-values per smooth term
    this.covMatrix = null;  // Covariance matrix of coefficients
    this.hatMatrix = null;  // Hat matrix
    this.rss = null;  // Residual sum of squares
    this.r2 = null;  // R-squared
    this.n = null;  // Training sample size
  }

  _buildSmoothConfigs(X) {
    const p = X[0].length;
    const configs = [];

    for (let j = 0; j < p; j++) {
      const values = X.map((row) => row[j]);
      // For backward compatibility: old code used nSplines-1 interior knots
      const nKnots = this.basis === 'tp' ? Math.max(1, this.nSplines - 1) : this.nSplines;
      const knots = computeKnots(values, nKnots, this.knotPlacement);

      configs.push({
        name: `s(x${j})`,
        knots: knots,
        type: this.basis,  // Use 'type' to match buildSmoothMatrix
        basis: this.basis,  // Keep for backwards compatibility
        degree: 3,
        penaltyOrder: this.penaltyOrder,
        variable: j,
        nBasis: this._getNBasis(knots)
      });
    }

    this.smoothConfigs = configs;
  }

  _getNBasis(knots) {
    // Number of basis functions depends on basis type
    if (this.basis === 'bs') {
      return knots.length + 4;  // B-spline with degree 3
    } else if (this.basis === 'cr') {
      return knots.length + 4;  // Cubic regression spline
    } else if (this.basis === 'tp') {
      return knots.length + 1;  // Truncated power basis
    }
    return knots.length + 4;  // Default
  }

  _designMatrix(X) {
    return buildSmoothMatrix(X, this.smoothConfigs, true);
  }
}

export class GAMRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.gam = new GAMBase({ ...opts, task: 'regression' });
  }

  fit(X, y = null) {
    const prepared = prepareDataset(X, y);
    this.gam.columns = prepared.columns;
    this.gam._buildSmoothConfigs(prepared.X);

    const design = this.gam._designMatrix(prepared.X);
    const designMatrix = new Matrix(design);
    const n = designMatrix.rows;
    const p = designMatrix.columns;
    this.gam.n = n;  // Store training sample size

    // Build penalty matrix for all smooths
    const S = this._buildPenaltyMatrix();

    // Fit with smoothness selection or fixed lambda
    let lambda = this.gam.lambda;
    if (lambda === null && this.gam.smoothMethod) {
      // Automatic smoothness selection
      const optResult = optimizeSmoothness(designMatrix, prepared.y, S, {
        method: this.gam.smoothMethod,
        lambdaMin: 1e-8,
        lambdaMax: 1e4,
        nSteps: 20
      });
      lambda = optResult.lambda;
    } else if (lambda === null) {
      lambda = 0;  // No penalty
    }

    // Fit penalized regression
    const fitResult = fitPenalizedRegression(designMatrix, prepared.y, S, lambda);
    this.gam.coef = fitResult.coefficients;
    this.gam.edf = fitResult.edf;
    this.gam.covMatrix = fitResult.covMatrix;
    this.gam.hatMatrix = fitResult.hatMatrix;

    // Compute fitted values and residuals
    const fitted = designMatrix.mmul(Matrix.columnVector(this.gam.coef)).getColumn(0);
    let rss = 0;
    for (let i = 0; i < n; i++) {
      const resid = prepared.y[i] - fitted[i];
      rss += resid * resid;
    }
    this.gam.rss = rss;

    // Compute R-squared
    const yMean = prepared.y.reduce((sum, val) => sum + val, 0) / n;
    let tss = 0;
    for (let i = 0; i < n; i++) {
      tss += (prepared.y[i] - yMean) * (prepared.y[i] - yMean);
    }
    this.gam.r2 = 1 - rss / tss;

    // Compute EDF and p-values for each smooth term
    this.gam.smoothEDFs = computeSmoothEDF(fitResult.hatMatrix, this.gam.smoothConfigs, true);
    const residualDf = n - this.gam.edf;
    this.gam.smoothPValues = computeSmoothPValues(
      this.gam.smoothEDFs,
      residualDf,
      this.gam.rss,
      this.gam.covMatrix,
      this.gam.smoothConfigs
    );

    this.fitted = true;
    return this;
  }

  _buildPenaltyMatrix() {
    // Build block-diagonal penalty matrix for all smooth terms
    let totalBasis = 1;  // Intercept
    for (const config of this.gam.smoothConfigs) {
      totalBasis += config.nBasis;
    }

    const S = Matrix.zeros(totalBasis, totalBasis);

    let offset = 1;  // Skip intercept (no penalty on intercept)
    for (const config of this.gam.smoothConfigs) {
      const nBasis = config.nBasis;
      const Si = penaltyMatrix(nBasis, this.gam.penaltyOrder);

      // Place in block-diagonal position
      for (let i = 0; i < nBasis; i++) {
        for (let j = 0; j < nBasis; j++) {
          S.set(offset + i, offset + j, Si.get(i, j));
        }
      }

      offset += nBasis;
    }

    return S;
  }

  predict(X) {
    if (!this.fitted) throw new Error('GAMRegressor: estimator not fitted.');
    const data = preparePredictInput(X, this.gam.columns);
    const design = this.gam._designMatrix(data);
    const designMatrix = new Matrix(design);
    const predictions = designMatrix.mmul(Matrix.columnVector(this.gam.coef)).getColumn(0);
    return predictions;
  }

  /**
   * Get confidence intervals for predictions
   * @param {Array} X - Input data
   * @param {number} level - Confidence level (default: 0.95)
   * @returns {Object} { fitted, lower, upper, se }
   */
  predictWithInterval(X, level = 0.95) {
    if (!this.fitted) throw new Error('GAMRegressor: estimator not fitted.');
    const data = preparePredictInput(X, this.gam.columns);
    const design = this.gam._designMatrix(data);
    const designMatrix = new Matrix(design);

    // Fitted values
    const fitted = designMatrix.mmul(Matrix.columnVector(this.gam.coef)).getColumn(0);

    // Standard errors (using training sample size)
    const sigma2 = this.gam.rss / (this.gam.n - this.gam.edf);
    const se = [];

    for (let i = 0; i < design.length; i++) {
      const xi = Matrix.rowVector(design[i]);
      const variance = xi.mmul(this.gam.covMatrix).mmul(xi.transpose()).get(0, 0) * sigma2;
      se.push(Math.sqrt(variance));
    }

    // Confidence intervals
    const intervals = computeConfidenceIntervals(fitted, se, level);

    return {
      fitted,
      se,
      lower: intervals.lower,
      upper: intervals.upper
    };
  }

  summary() {
    if (!this.fitted) {
      throw new Error('GAMRegressor: estimator not fitted.');
    }

    return createGAMSummary({
      coefficients: this.gam.coef,
      edf: this.gam.edf,
      smoothEDFs: this.gam.smoothEDFs,
      smoothPValues: this.gam.smoothPValues,
      rss: this.gam.rss,
      n: this.gam.hatMatrix.rows,
      p: this.gam.coef.length,
      r2: this.gam.r2,
      smoothConfigs: this.gam.smoothConfigs,
      covMatrix: this.gam.covMatrix
    });
  }
}

export class GAMClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.gam = new GAMBase({ ...opts, task: 'classification' });
  }

  fit(X, y = null) {
    const prepared = prepareDataset(X, y);
    const classes = Array.from(new Set(prepared.y));
    if (classes.length !== 2) {
      throw new Error('GAMClassifier currently supports exactly 2 classes.');
    }
    this.gam.classMap = {
      [classes[0]]: 0,
      [classes[1]]: 1
    };
    this.gam.classes = classes;
    const numericY = prepared.y.map((label) => this.gam.classMap[label]);
    this.gam.columns = prepared.columns;
    this.gam._buildSmoothConfigs(prepared.X);
    const design = this.gam._designMatrix(prepared.X);

    // Build penalty matrix for smooths
    const S = this._buildPenaltyMatrix();

    // Set up regularization for GLM
    let lambda = this.gam.lambda;
    if (lambda === null && this.gam.smoothMethod) {
      // For classification, use a moderate default lambda
      // (proper smoothness selection for binomial is more complex)
      lambda = 0.1;
    } else if (lambda === null) {
      lambda = 0;
    }

    // Convert penalty matrix to regularization for GLM
    // GLM uses ridge regularization, so we approximate with a scalar
    const avgPenalty = lambda;

    const logitResult = logit.fit(design, numericY, {
      intercept: false,
      maxIter: this.gam.maxIter,
      tol: this.gam.tol,
      regularization: avgPenalty > 0 ? { alpha: avgPenalty, l1_ratio: 0 } : null
    });
    this.gam.coef = logitResult.coefficients;
    this.gam.model = logitResult;
    this.fitted = true;
    return this;
  }

  _buildPenaltyMatrix() {
    // Build block-diagonal penalty matrix for all smooth terms
    let totalBasis = 1;  // Intercept
    for (const config of this.gam.smoothConfigs) {
      totalBasis += config.nBasis;
    }

    const S = Matrix.zeros(totalBasis, totalBasis);

    let offset = 1;  // Skip intercept (no penalty on intercept)
    for (const config of this.gam.smoothConfigs) {
      const nBasis = config.nBasis;
      const Si = penaltyMatrix(nBasis, this.gam.penaltyOrder);

      // Place in block-diagonal position
      for (let i = 0; i < nBasis; i++) {
        for (let j = 0; j < nBasis; j++) {
          S.set(offset + i, offset + j, Si.get(i, j));
        }
      }

      offset += nBasis;
    }

    return S;
  }

  _decisionFunction(X) {
    const design = this.gam._designMatrix(X);
    const scores = [];
    for (const row of design) {
      let sum = 0;
      for (let i = 0; i < row.length; i++) {
        sum += row[i] * this.gam.coef[i];
      }
      scores.push(sum);
    }
    return scores;
  }

  predictProba(X) {
    if (!this.fitted) throw new Error('GAMClassifier: estimator not fitted.');
    const data = preparePredictInput(X, this.gam.columns);
    const scores = this._decisionFunction(data);
    const probs = scores.map((score) => {
      const p = 1 / (1 + Math.exp(-score));
      return { [this.gam.classes[0]]: 1 - p, [this.gam.classes[1]]: p };
    });
    return probs;
  }

  predict(X) {
    const probs = this.predictProba(X);
    const [negLabel, posLabel] = this.gam.classes;
    return probs.map((dist) => (dist[posLabel] >= 0.5 ? posLabel : negLabel));
  }
}

export default {
  GAMRegressor,
  GAMClassifier
};
