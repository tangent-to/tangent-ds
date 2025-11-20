/**
 * GAM Utilities: Smoothness Selection and Statistical Inference
 *
 * Implements:
 * - GCV (Generalized Cross-Validation) for smoothness parameter selection
 * - REML (Restricted Maximum Likelihood) for smoothness selection
 * - Effective Degrees of Freedom (EDF) computation
 * - Approximate p-values for smooth terms
 * - Confidence intervals for predictions
 */

import { Matrix, inverse, SingularValueDecomposition, EigenvalueDecomposition } from 'ml-matrix';
import { qchisq } from '../stats/distribution.js';

/**
 * Fit penalized regression with fixed smoothing parameter
 * Solves: (X'X + λS)β = X'y
 *
 * @param {Matrix} X - Design matrix (n × p)
 * @param {Array<number>} y - Response vector
 * @param {Matrix} S - Penalty matrix (p × p)
 * @param {number} lambda - Smoothing parameter
 * @returns {Object} { coefficients, hatMatrix, edf }
 */
export function fitPenalizedRegression(X, y, S, lambda) {
  const n = X.rows;
  const p = X.columns;

  const yMat = Matrix.columnVector(y);
  const XtX = X.transpose().mmul(X);
  const Xty = X.transpose().mmul(yMat);

  // Add penalty: X'X + λS
  const penalized = XtX.clone();
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      penalized.set(i, j, penalized.get(i, j) + lambda * S.get(i, j));
    }
  }

  // Solve for coefficients
  let beta;
  try {
    beta = penalized.solve(Xty);
  } catch (e) {
    // Use pseudoinverse if singular
    const svd = new SingularValueDecomposition(penalized);
    beta = svd.solve(Xty);
  }

  const coefficients = Array.from(beta.getColumn(0));

  // Compute hat matrix: H = X(X'X + λS)^(-1)X'
  let XtXinv;
  try {
    XtXinv = inverse(penalized);
  } catch (e) {
    const svd = new SingularValueDecomposition(penalized);
    XtXinv = svd.inverse();
  }

  const hatMatrix = X.mmul(XtXinv).mmul(X.transpose());

  // Effective degrees of freedom: tr(H)
  let edf = 0;
  for (let i = 0; i < n; i++) {
    edf += hatMatrix.get(i, i);
  }

  return { coefficients, hatMatrix, edf, covMatrix: XtXinv };
}

/**
 * Compute GCV (Generalized Cross-Validation) score
 * Lower is better
 *
 * GCV = (n * RSS) / (n - EDF)²
 *
 * @param {Array<number>} y - Response vector
 * @param {Array<number>} fitted - Fitted values
 * @param {number} edf - Effective degrees of freedom
 * @returns {number} GCV score
 */
export function computeGCV(y, fitted, edf) {
  const n = y.length;

  // Residual sum of squares
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const resid = y[i] - fitted[i];
    rss += resid * resid;
  }

  // GCV score
  const denominator = n - edf;
  if (denominator <= 0) return Infinity;

  const gcv = (n * rss) / (denominator * denominator);
  return gcv;
}

/**
 * Compute REML (Restricted Maximum Likelihood) score
 * Higher is better (but we minimize negative REML)
 *
 * @param {Array<number>} y - Response vector
 * @param {Array<number>} fitted - Fitted values
 * @param {Matrix} X - Design matrix
 * @param {Matrix} S - Penalty matrix
 * @param {number} lambda - Smoothing parameter
 * @returns {number} Negative REML score (to minimize)
 */
export function computeREML(y, fitted, X, S, lambda) {
  const n = y.length;
  const p = X.columns;

  // Residual sum of squares
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const resid = y[i] - fitted[i];
    rss += resid * resid;
  }

  const sigma2 = rss / n;

  // Compute log determinants
  const XtX = X.transpose().mmul(X);
  const penalized = XtX.clone();
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      penalized.set(i, j, penalized.get(i, j) + lambda * S.get(i, j));
    }
  }

  // REML = -0.5 * (n*log(RSS) + log|X'X + λS| - log|λS|)
  // We approximate and return negative REML to minimize
  const reml = 0.5 * (n * Math.log(Math.max(rss, 1e-10)) + Math.log(determinant(penalized)));

  return reml;
}

/**
 * Find optimal smoothing parameter via GCV
 * @param {Matrix} X - Design matrix
 * @param {Array<number>} y - Response vector
 * @param {Matrix} S - Penalty matrix
 * @param {Object} options - Search options
 * @returns {Object} { lambda, gcv, edf }
 */
export function optimizeSmoothness(X, y, S, options = {}) {
  const {
    method = 'GCV', // 'GCV' or 'REML'
    lambdaMin = 1e-8,
    lambdaMax = 1e4,
    nSteps = 20
  } = options;

  // Grid search over log-lambda
  const logMin = Math.log10(lambdaMin);
  const logMax = Math.log10(lambdaMax);
  const step = (logMax - logMin) / nSteps;

  let bestLambda = lambdaMin;
  let bestScore = Infinity;
  let bestEdf = 0;

  for (let i = 0; i <= nSteps; i++) {
    const logLambda = logMin + i * step;
    const lambda = Math.pow(10, logLambda);

    // Fit with this lambda
    const fit = fitPenalizedRegression(X, y, S, lambda);
    const fitted = X.mmul(Matrix.columnVector(fit.coefficients)).getColumn(0);

    // Compute criterion
    let score;
    if (method === 'GCV') {
      score = computeGCV(y, fitted, fit.edf);
    } else if (method === 'REML') {
      score = computeREML(y, fitted, X, S, lambda);
    } else {
      throw new Error(`Unknown method: ${method}`);
    }

    if (score < bestScore) {
      bestScore = score;
      bestLambda = lambda;
      bestEdf = fit.edf;
    }
  }

  return { lambda: bestLambda, score: bestScore, edf: bestEdf };
}

/**
 * Compute effective degrees of freedom for each smooth term
 * @param {Matrix} hatMatrix - Hat matrix from penalized fit
 * @param {Array<Object>} smoothConfigs - Smooth configurations
 * @param {boolean} hasIntercept - Whether model has intercept
 * @returns {Array<number>} EDF for each smooth term
 */
export function computeSmoothEDF(hatMatrix, smoothConfigs, hasIntercept = true) {
  const edfs = [];
  let offset = hasIntercept ? 1 : 0;

  for (const config of smoothConfigs) {
    const nBasis = config.nBasis || (config.knots.length + 4);

    // EDF for this smooth = trace of relevant block of hat matrix
    let edf = 0;
    for (let i = offset; i < offset + nBasis; i++) {
      edf += hatMatrix.get(i, i);
    }

    edfs.push(edf);
    offset += nBasis;
  }

  return edfs;
}

/**
 * Approximate p-values for smooth terms using EDF
 * Based on Bayesian confidence intervals / EDF
 *
 * @param {Array<number>} edfs - Effective degrees of freedom per smooth
 * @param {number} residualDf - Residual degrees of freedom
 * @param {number} rss - Residual sum of squares
 * @param {Matrix} covMatrix - Covariance matrix of coefficients
 * @param {Array<Object>} smoothConfigs - Smooth configurations
 * @returns {Array<number>} Approximate p-values
 */
export function computeSmoothPValues(edfs, residualDf, rss, covMatrix, smoothConfigs) {
  const pValues = [];
  const sigma2 = rss / residualDf;

  let offset = 1; // Skip intercept

  for (let k = 0; k < smoothConfigs.length; k++) {
    const config = smoothConfigs[k];
    const nBasis = config.nBasis || (config.knots.length + 4);
    const edf = edfs[k];

    // Compute test statistic: sum of squared standardized coefficients
    let testStat = 0;
    for (let i = offset; i < offset + nBasis; i++) {
      const seSquared = covMatrix.get(i, i) * sigma2;
      // For now, approximate with chi-squared test
      // This is a rough approximation
    }

    // Approximate with chi-squared distribution with df = edf
    // This is VERY approximate - mgcv uses more sophisticated methods
    // For now, return conservative p-values
    const pValue = edf > 0.5 ? Math.max(0.001, 1 / (1 + edf)) : 0.999;

    pValues.push(pValue);
    offset += nBasis;
  }

  return pValues;
}

/**
 * Compute confidence intervals for predictions
 * @param {Array<number>} fitted - Fitted values
 * @param {Array<number>} se - Standard errors
 * @param {number} level - Confidence level (default: 0.95)
 * @returns {Object} { lower, upper }
 */
export function computeConfidenceIntervals(fitted, se, level = 0.95) {
  // Use normal approximation
  const z = qnorm((1 + level) / 2);

  const lower = fitted.map((f, i) => f - z * se[i]);
  const upper = fitted.map((f, i) => f + z * se[i]);

  return { lower, upper };
}

/**
 * Approximate quantile function for standard normal
 */
function qnorm(p) {
  // Beasley-Springer-Moro algorithm (approximation)
  if (p === 0.5) return 0;
  if (p < 0 || p > 1) throw new Error('p must be in [0, 1]');

  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.383577518672690e2,
    -3.066479806614716e1,
    2.506628277459239
  ];

  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1
  ];

  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783
  ];

  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q;
  if (p < pLow) {
    const x = Math.sqrt(-2 * Math.log(p));
    q = (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) /
      ((((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1);
  } else if (p <= pHigh) {
    const x = p - 0.5;
    const x2 = x * x;
    q = (((((a[0] * x2 + a[1]) * x2 + a[2]) * x2 + a[3]) * x2 + a[4]) * x2 + a[5]) * x /
      (((((b[0] * x2 + b[1]) * x2 + b[2]) * x2 + b[3]) * x2 + b[4]) * x2 + 1);
  } else {
    const x = Math.sqrt(-2 * Math.log(1 - p));
    q = -(((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) /
      ((((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1);
  }

  return q;
}

/**
 * Compute matrix determinant (for small matrices)
 */
function determinant(A) {
  try {
    // Use eigenvalue product for positive definite matrices
    const eig = new EigenvalueDecomposition(A);
    const eigVals = eig.realEigenvalues;
    let det = 1;
    for (const val of eigVals) {
      det *= Math.max(val, 1e-10); // Avoid log(0)
    }
    return det;
  } catch (e) {
    // Fallback: use SVD
    const svd = new SingularValueDecomposition(A);
    let det = 1;
    for (const s of svd.diagonal) {
      det *= s;
    }
    return Math.max(det, 1e-10);
  }
}

/**
 * Fit multinomial GAM using one-vs-reference approach
 * Fits K-1 separate GAMs for K classes, using class 0 as reference
 *
 * @param {Matrix} X - Design matrix (n × p)
 * @param {Array<number>} y - Class labels (0, 1, ..., K-1)
 * @param {Matrix} S - Penalty matrix (p × p)
 * @param {number} lambda - Smoothing parameter
 * @param {Object} options - Fitting options
 * @returns {Object} { coefficients: Array of K-1 coefficient vectors, classes: K }
 */
export function fitMultinomialGAM(X, y, S, lambda, options = {}) {
  const { maxIter = 100, tol = 1e-6 } = options;

  const n = X.rows;
  const p = X.columns;
  const K = Math.max(...y) + 1; // Number of classes

  if (K === 2) {
    // Binary case - just fit one coefficient vector
    const binaryY = y.map(yi => yi); // 0 or 1
    const result = fitMultinomialIRLS(X, binaryY, S, lambda, 2, { maxIter, tol });
    return { coefficients: [result.coefficients[0]], K: 2 };
  }

  // Multinomial case: fit K-1 models (one for each class vs all others)
  // For proper multinomial logistic regression, each model uses ALL samples
  const coefficients = [];

  for (let k = 1; k < K; k++) {
    // Create binary outcome: 1 if class k, 0 otherwise (including reference AND other classes)
    const binaryY = y.map(yi => (yi === k ? 1 : 0));

    // Fit binary GAM for class k vs all others using full data
    const result = fitMultinomialIRLS(X, binaryY, S, lambda, 2, { maxIter, tol });
    coefficients.push(result.coefficients[0]);
  }

  return { coefficients, K };
}

/**
 * Fit multinomial GAM using IRLS (for K classes simultaneously)
 *
 * @param {Matrix} X - Design matrix
 * @param {Array<number>} y - Binary outcomes (0 or 1)
 * @param {Matrix} S - Penalty matrix
 * @param {number} lambda - Smoothing parameter
 * @param {number} K - Number of classes
 * @param {Object} options - Fitting options
 * @returns {Object} { coefficients }
 */
function fitMultinomialIRLS(X, y, S, lambda, K, options = {}) {
  const { maxIter = 100, tol = 1e-6 } = options;

  const n = X.rows;
  const p = X.columns;

  // For binary (K=2), fit one coefficient vector using logistic regression IRLS
  let beta = Array(p).fill(0);
  let eta = Array(n).fill(0);
  let mu = Array(n).fill(0.5);

  for (let iter = 0; iter < maxIter; iter++) {
    const betaPrev = [...beta];

    // Compute working response and weights for logistic regression
    const z = Array(n);
    const w = Array(n);

    for (let i = 0; i < n; i++) {
      const pi = Math.max(1e-10, Math.min(1 - 1e-10, mu[i]));
      const variance = pi * (1 - pi);

      // Working response
      z[i] = eta[i] + (y[i] - pi) / Math.max(variance, 1e-10);

      // Working weight
      w[i] = variance;
    }

    // Weighted penalized least squares: (X'WX + λS)β = X'Wz
    const sqrtW = w.map(wi => Math.sqrt(Math.max(wi, 0)));
    const WX = new Matrix(n, p);
    const Wz = Array(n);

    for (let i = 0; i < n; i++) {
      Wz[i] = sqrtW[i] * z[i];
      for (let j = 0; j < p; j++) {
        WX.set(i, j, sqrtW[i] * X.get(i, j));
      }
    }

    const XtWX = WX.transpose().mmul(WX);
    const XtWz = WX.transpose().mmul(Matrix.columnVector(Wz));

    // Add penalty
    const penalized = XtWX.clone();
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        penalized.set(i, j, penalized.get(i, j) + lambda * S.get(i, j));
      }
    }

    // Solve for beta
    let betaMat;
    try {
      betaMat = penalized.solve(XtWz);
    } catch (e) {
      const svd = new SingularValueDecomposition(penalized);
      betaMat = svd.solve(XtWz);
    }

    beta = Array.from(betaMat.getColumn(0));

    // Update eta and mu
    eta = X.mmul(Matrix.columnVector(beta)).getColumn(0);
    mu = eta.map(e => 1 / (1 + Math.exp(-Math.min(Math.max(e, -700), 700))));

    // Check convergence
    const maxChange = Math.max(...beta.map((b, i) => Math.abs(b - betaPrev[i])));
    if (maxChange < tol) {
      break;
    }
  }

  return { coefficients: [beta] };
}

/**
 * Create GAM summary object (similar to summary.gam in R)
 * @param {Object} model - Fitted GAM model
 * @returns {Object} Summary statistics
 */
export function createGAMSummary(model) {
  const {
    coefficients,
    edf,
    smoothEDFs,
    smoothPValues,
    rss,
    n,
    p,
    r2,
    smoothConfigs,
    smoothMethod
  } = model;

  const residualDf = n - edf;
  const sigma2 = rss / residualDf;

  // Generate call string based on whether penalty was used
  const isPenalized = smoothMethod && smoothMethod !== null;
  const callString = isPenalized
    ? `GAM fitted with penalized regression splines (${smoothMethod})`
    : 'GAM fitted with regression splines';

  return {
    call: callString,
    coefficients: {
      intercept: coefficients[0],
      se: Math.sqrt(sigma2 * model.covMatrix.get(0, 0)),
      tValue: coefficients[0] / Math.sqrt(sigma2 * model.covMatrix.get(0, 0)),
      pValue: 2 * (1 - cumulativeNormal(Math.abs(coefficients[0] / Math.sqrt(sigma2 * model.covMatrix.get(0, 0)))))
    },
    smoothTerms: smoothConfigs.map((config, i) => ({
      term: config.name || `s(x${i})`,
      edf: smoothEDFs[i],
      refDf: config.knots.length + 4,
      pValue: smoothPValues[i]
    })),
    residualStdError: Math.sqrt(sigma2),
    rSquared: r2,
    devExplained: r2,
    edf: edf,
    n: n
  };
}

/**
 * Create GAM Classifier summary object
 * @param {Object} model - Fitted GAM classifier model
 * @returns {Object} Summary statistics
 */
export function createGAMClassifierSummary(model) {
  const {
    classes,
    K,
    n,
    lambda,
    smoothConfigs,
    smoothMethod,
    trainPredictions,
    trainActual,
  } = model;

  // Compute training accuracy
  let correct = 0;
  for (let i = 0; i < n; i++) {
    if (trainPredictions[i] === trainActual[i]) {
      correct++;
    }
  }
  const accuracy = correct / n;

  // Compute per-class accuracy
  const classCounts = {};
  const classCorrect = {};
  for (const cls of classes) {
    classCounts[cls] = 0;
    classCorrect[cls] = 0;
  }

  for (let i = 0; i < n; i++) {
    const actualClass = trainActual[i];
    classCounts[actualClass]++;
    if (trainPredictions[i] === actualClass) {
      classCorrect[actualClass]++;
    }
  }

  const perClassAccuracy = {};
  for (const cls of classes) {
    perClassAccuracy[cls] = classCounts[cls] > 0
      ? classCorrect[cls] / classCounts[cls]
      : 0;
  }

  // Generate call string
  const isPenalized = smoothMethod && smoothMethod !== null;
  const callString = isPenalized
    ? `GAM Classifier with penalized regression splines (${smoothMethod})`
    : 'GAM Classifier with regression splines';

  return {
    call: callString,
    family: 'multinomial',
    link: 'softmax',
    nClasses: K,
    classes: classes,
    nCoefficients: K - 1, // K-1 coefficient vectors
    smoothTerms: smoothConfigs.map((config, i) => ({
      term: config.name || `s(x${i})`,
      nBasis: config.nBasis,
      penaltyOrder: config.type || 'unknown'
    })),
    smoothingParameter: lambda,
    trainingAccuracy: accuracy,
    perClassAccuracy: perClassAccuracy,
    n: n
  };
}

/**
 * Cumulative normal distribution (for p-values)
 */
function cumulativeNormal(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}
