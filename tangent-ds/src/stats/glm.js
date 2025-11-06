/**
 * Generalized Linear Models (GLM) and Generalized Linear Mixed Models (GLMM)
 *
 * Implements:
 * - GLM fitting via IRLS (Iteratively Reweighted Least Squares)
 * - GLMM fitting via Laplace approximation
 * - Support for weights, offsets, regularization
 * - Random intercepts and slopes
 */

import { Matrix, SingularValueDecomposition, inverse } from 'ml-matrix';
import { createFamily } from './families.js';
import { mean, sum } from '../core/math.js';

// ============================================================================
// GLM Fitting (Fixed Effects Only)
// ============================================================================

/**
 * Fit GLM via IRLS
 *
 * @param {Array<Array<number>>} X - Design matrix (n × p)
 * @param {Array<number>} y - Response vector
 * @param {Object} options - Fitting options
 * @returns {Object} Fitted model
 */
export function fitGLM(X, y, options = {}) {
  const {
    family = 'gaussian',
    link = null,
    weights = null,
    offset = null,
    intercept = true,
    maxIter = 100,
    tol = 1e-8,
    regularization = null,
    dispersion = 'estimate' // 'estimate', 'fixed', or numeric value
  } = options;

  const n = X.length;
  const p = X[0].length;

  // Create family object
  const familyObj = createFamily({ family, link });

  // Add intercept if requested
  let Xmat = intercept ? addIntercept(X) : X.map(row => [...row]);
  const ncoef = Xmat[0].length;

  // Initialize weights and offset
  const w = weights || Array(n).fill(1);
  const off = offset || Array(n).fill(0);

  // Initialize mu and eta
  let { mu, eta } = familyObj.initialize(y, w);

  // Add offset to eta
  eta = eta.map((e, i) => e + off[i]);

  // IRLS iteration
  let coefficients = null;
  let converged = false;
  let iteration = 0;
  let deviance = Infinity;

  for (iteration = 0; iteration < maxIter; iteration++) {
    const deviancePrev = deviance;

    // Compute working response and weights
    const { z, wt } = computeWorkingWeights(y, mu, eta, off, w, familyObj);

    // Weighted least squares
    coefficients = weightedLeastSquares(Xmat, z, wt, regularization);

    // Update eta and mu
    eta = matrixVectorMultiply(Xmat, coefficients);
    mu = familyObj.link.linkinv(eta);

    // Compute deviance
    deviance = familyObj.deviance(y, mu, w);

    // Check convergence
    if (iteration > 0 && Math.abs(deviance - deviancePrev) < tol) {
      converged = true;
      break;
    }
  }

  // Compute fitted values and residuals
  const fitted = mu.map((m, i) => m);
  const residuals = y.map((yi, i) => yi - fitted[i]);
  const pearsonResiduals = computePearsonResiduals(y, mu, w, familyObj);
  const devianceResiduals = computeDevianceResiduals(y, mu, w, familyObj);

  // Compute dispersion parameter
  let phi;
  if (typeof dispersion === 'number') {
    phi = dispersion;
  } else if (dispersion === 'fixed') {
    phi = 1.0; // Fixed at 1 for Binomial and Poisson
  } else {
    // Estimate dispersion from Pearson chi-square
    const pearsonChiSq = sum(pearsonResiduals.map(r => r * r));
    phi = pearsonChiSq / (n - ncoef);
  }

  // Compute standard errors
  const { standardErrors, covarianceMatrix } = computeStandardErrors(
    Xmat,
    mu,
    w,
    phi,
    familyObj
  );

  // Compute confidence intervals (95%)
  const confidenceIntervals = coefficients.map((coef, i) => {
    const se = standardErrors[i];
    return {
      lower: coef - 1.96 * se,
      upper: coef + 1.96 * se
    };
  });

  // Compute AIC and BIC
  const logLik = computeLogLikelihood(y, mu, w, phi, familyObj);
  const aic = -2 * logLik + 2 * ncoef;
  const bic = -2 * logLik + Math.log(n) * ncoef;

  // Null deviance (intercept-only model)
  const nullDeviance = computeNullDeviance(y, w, familyObj);

  // Pseudo R²
  const pseudoR2 = 1 - deviance / nullDeviance;

  return {
    coefficients,
    fitted,
    residuals,
    pearsonResiduals,
    devianceResiduals,
    mu,
    eta,
    standardErrors,
    confidenceIntervals,
    covarianceMatrix,
    deviance,
    nullDeviance,
    pseudoR2,
    dispersion: phi,
    logLikelihood: logLik,
    aic,
    bic,
    iterations: iteration + 1,
    converged,
    n,
    p: ncoef,
    dfResidual: n - ncoef,
    family: familyObj.family,
    link: familyObj.link.name,
    intercept
  };
}

/**
 * Compute working weights and response for IRLS
 */
function computeWorkingWeights(y, mu, eta, offset, weights, family) {
  const n = y.length;
  const z = new Array(n);
  const wt = new Array(n);

  const variance = family.variance(mu);
  const mu_eta = family.link.mu_eta(eta.map((e, i) => e - offset[i]));

  for (let i = 0; i < n; i++) {
    const v = Math.max(variance[i], 1e-10);
    const dmu = Math.max(Math.abs(mu_eta[i]), 1e-10);

    // Working response
    z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / dmu;

    // Working weight
    wt[i] = weights[i] * dmu * dmu / v;
  }

  return { z, wt };
}

/**
 * Weighted least squares with optional regularization
 */
function weightedLeastSquares(X, y, weights, regularization = null) {
  const n = X.length;
  const p = X[0].length;

  // Create weighted design matrix: W^(1/2) * X
  const sqrtW = weights.map(w => Math.sqrt(Math.max(w, 0)));
  const WX = X.map((row, i) => row.map(x => sqrtW[i] * x));
  const Wy = y.map((yi, i) => sqrtW[i] * yi);

  const XtX = new Matrix(WX).transpose().mmul(new Matrix(WX));
  const Xty = new Matrix(WX).transpose().mmul(Matrix.columnVector(Wy));

  // Add regularization if specified
  if (regularization) {
    const { alpha = 0, l1_ratio = 0 } = regularization;
    const lambda2 = alpha * (1 - l1_ratio);

    // Ridge regularization (L2)
    if (lambda2 > 0) {
      for (let i = 0; i < p; i++) {
        XtX.set(i, i, XtX.get(i, i) + lambda2);
      }
    }

    // L1 regularization would require coordinate descent, not implemented yet
    if (alpha * l1_ratio > 0) {
      console.warn('L1 regularization not yet implemented, using L2 only');
    }
  }

  // Solve: (X'WX)β = X'Wy
  try {
    const beta = XtX.solve(Xty);
    return Array.from(beta.getColumn(0));
  } catch (e) {
    // If singular, use SVD-based pseudoinverse
    try {
      const beta = inverse(XtX).mmul(Xty);
      return Array.from(beta.getColumn(0));
    } catch (e2) {
      // Last resort: use SVD on original problem
      const WXmat = new Matrix(WX);
      const Wymat = Matrix.columnVector(Wy);
      const svd = new SingularValueDecomposition(WXmat);
      const beta = svd.solve(Wymat);
      return Array.from(beta.getColumn(0));
    }
  }
}

/**
 * Compute standard errors for coefficients
 */
function computeStandardErrors(X, mu, weights, phi, family) {
  const n = X.length;
  const p = X[0].length;

  // Compute working weights
  const variance = family.variance(mu);
  const mu_eta = family.link.mu_eta(family.link.linkfun(mu));

  const W = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.max(variance[i], 1e-10);
    const dmu = Math.max(Math.abs(mu_eta[i]), 1e-10);
    W[i] = weights[i] * dmu * dmu / v;
  }

  // Information matrix: I = X'WX
  const sqrtW = W.map(w => Math.sqrt(Math.max(w, 0)));
  const WX = X.map((row, i) => row.map(x => sqrtW[i] * x));
  const XtWX = new Matrix(WX).transpose().mmul(new Matrix(WX));

  // Covariance matrix: Cov = φ * (X'WX)^(-1)
  let covMatrix;
  try {
    covMatrix = inverse(XtWX).mul(phi);
  } catch (e) {
    // If singular, use SVD-based pseudoinverse
    const svd = new SingularValueDecomposition(XtWX);
    const s = svd.diagonal;
    const V = svd.rightSingularVectors;

    // Compute pseudoinverse: V * S^+ * V'
    const Sinv = new Matrix(s.length, s.length);
    for (let i = 0; i < s.length; i++) {
      if (Math.abs(s[i]) > 1e-10) {
        Sinv.set(i, i, 1 / s[i]);
      }
    }

    const VMatrix = new Matrix(V);
    covMatrix = VMatrix.mmul(Sinv).mmul(VMatrix.transpose()).mul(phi);
  }

  // Standard errors are square roots of diagonal
  const standardErrors = new Array(p);
  for (let i = 0; i < p; i++) {
    standardErrors[i] = Math.sqrt(Math.max(covMatrix.get(i, i), 0));
  }

  return {
    standardErrors,
    covarianceMatrix: covMatrix.to2DArray()
  };
}

/**
 * Compute Pearson residuals
 */
function computePearsonResiduals(y, mu, weights, family) {
  const variance = family.variance(mu);
  return y.map((yi, i) => {
    const v = Math.max(variance[i], 1e-10);
    return (yi - mu[i]) / Math.sqrt(v / weights[i]);
  });
}

/**
 * Compute deviance residuals
 */
function computeDevianceResiduals(y, mu, weights, family) {
  const n = y.length;
  const residuals = new Array(n);

  for (let i = 0; i < n; i++) {
    const di = family.deviance([y[i]], [mu[i]], [weights[i]]);
    residuals[i] = Math.sign(y[i] - mu[i]) * Math.sqrt(di);
  }

  return residuals;
}

/**
 * Compute log-likelihood (approximate for non-canonical links)
 */
function computeLogLikelihood(y, mu, weights, phi, family) {
  const n = y.length;
  const deviance = family.deviance(y, mu, weights);

  // Approximate log-likelihood from deviance
  // For exponential families: logLik ≈ -deviance/(2φ) + constant
  let logLik = -deviance / (2 * phi);

  // Add constant term (depends on family)
  if (family.family === 'gaussian') {
    const sumW = sum(weights);
    logLik -= (sumW / 2) * Math.log(2 * Math.PI * phi);
  }

  return logLik;
}

/**
 * Compute null deviance (intercept-only model)
 */
function computeNullDeviance(y, weights, family) {
  const n = y.length;
  const sumW = sum(weights);
  const sumWY = sum(y.map((yi, i) => weights[i] * yi));
  const muNull = sumWY / sumW;
  const mu = Array(n).fill(muNull);

  return family.deviance(y, mu, weights);
}

// ============================================================================
// GLMM Fitting (Mixed Effects via Laplace Approximation)
// ============================================================================

/**
 * Fit GLMM with random effects via Laplace approximation
 *
 * @param {Array<Array<number>>} X - Fixed effects design matrix
 * @param {Array<number>} y - Response vector
 * @param {Object} randomEffects - Random effects specification
 * @param {Object} options - Fitting options
 * @returns {Object} Fitted model
 */
export function fitGLMM(X, y, randomEffects, options = {}) {
  const {
    family = 'gaussian',
    link = null,
    weights = null,
    offset = null,
    intercept = true,
    maxIter = 100,
    tol = 1e-6,
    dispersion = 'estimate'
  } = options;

  const n = X.length;

  // Create family object
  const familyObj = createFamily({ family, link });

  // Add intercept if requested
  let Xmat = intercept ? addIntercept(X) : X.map(row => [...row]);
  const nFixedCoef = Xmat[0].length;

  // Initialize weights and offset
  const w = weights || Array(n).fill(1);
  const off = offset || Array(n).fill(0);

  // Parse random effects structure
  const { Z, groupInfo, nRandomCoef } = buildRandomEffectsMatrix(
    n,
    randomEffects
  );

  // Initialize fixed and random effects
  let beta = new Array(nFixedCoef).fill(0); // fixed effects
  let u = new Array(nRandomCoef).fill(0); // random effects
  let theta = initializeVarianceComponents(groupInfo); // variance components

  // Initialize mu and eta
  let eta = matrixVectorMultiply(Xmat, beta).map((e, i) => e + off[i]);
  let mu = familyObj.link.linkinv(eta);

  let converged = false;
  let iteration = 0;
  let logLik = -Infinity;

  for (iteration = 0; iteration < maxIter; iteration++) {
    const logLikPrev = logLik;

    // Update linear predictor
    eta = matrixVectorMultiply(Xmat, beta);
    for (let i = 0; i < n; i++) {
      eta[i] += matrixVectorMultiply([Z[i]], u)[0] + off[i];
    }
    mu = familyObj.link.linkinv(eta);

    // Compute working weights and response
    const { z, wt } = computeWorkingWeights(y, mu, eta, off, w, familyObj);

    // Update fixed effects (penalized WLS)
    const betaNew = updateFixedEffects(Xmat, Z, z, wt, u, theta);

    // Update random effects (penalized WLS)
    const uNew = updateRandomEffects(Xmat, Z, z, wt, betaNew, theta);

    // Update variance components
    const thetaNew = updateVarianceComponents(uNew, groupInfo);

    // Compute marginal log-likelihood (via Laplace approximation)
    eta = matrixVectorMultiply(Xmat, betaNew);
    for (let i = 0; i < n; i++) {
      eta[i] += matrixVectorMultiply([Z[i]], uNew)[0] + off[i];
    }
    mu = familyObj.link.linkinv(eta);

    logLik = computeMarginalLogLikelihood(
      y, mu, uNew, theta, w, familyObj, groupInfo
    );

    // Check convergence
    if (iteration > 0 && Math.abs(logLik - logLikPrev) < tol) {
      converged = true;
      beta = betaNew;
      u = uNew;
      theta = thetaNew;
      break;
    }

    beta = betaNew;
    u = uNew;
    theta = thetaNew;
  }

  // Final predictions
  eta = matrixVectorMultiply(Xmat, beta);
  for (let i = 0; i < n; i++) {
    eta[i] += matrixVectorMultiply([Z[i]], u)[0] + off[i];
  }
  mu = familyObj.link.linkinv(eta);

  const fitted = mu;
  const residuals = y.map((yi, i) => yi - fitted[i]);

  // Compute standard errors for fixed effects
  const { standardErrors, covarianceMatrix } = computeGLMMStandardErrors(
    Xmat, Z, mu, beta, u, theta, w, familyObj, groupInfo
  );

  // Compute confidence intervals (95%)
  const confidenceIntervals = beta.map((coef, i) => {
    const se = standardErrors[i];
    return {
      lower: coef - 1.96 * se,
      upper: coef + 1.96 * se
    };
  });

  // Compute deviance
  const deviance = familyObj.deviance(y, mu, w);

  // Compute AIC and BIC
  const nParams = nFixedCoef + groupInfo.length; // fixed effects + variance components
  const aic = -2 * logLik + 2 * nParams;
  const bic = -2 * logLik + Math.log(n) * nParams;

  return {
    fixedEffects: beta,
    randomEffects: u,
    varianceComponents: theta,
    groupInfo,
    fitted,
    residuals,
    mu,
    eta,
    standardErrors,
    confidenceIntervals,
    covarianceMatrix,
    deviance,
    logLikelihood: logLik,
    aic,
    bic,
    iterations: iteration + 1,
    converged,
    n,
    nFixedEffects: nFixedCoef,
    nRandomEffects: nRandomCoef,
    dfResidual: n - nFixedCoef,
    family: familyObj.family,
    link: familyObj.link.name,
    intercept
  };
}

/**
 * Build random effects design matrix Z
 */
function buildRandomEffectsMatrix(n, randomEffects) {
  const { intercept: interceptGroup, slopes = {} } = randomEffects;

  const groupInfo = [];
  let nRandomCoef = 0;

  // Random intercepts
  if (interceptGroup) {
    const groups = interceptGroup;
    const uniqueGroups = [...new Set(groups)];
    const groupMap = Object.fromEntries(uniqueGroups.map((g, i) => [g, i]));

    groupInfo.push({
      type: 'intercept',
      groups: uniqueGroups,
      groupMap,
      nGroups: uniqueGroups.length,
      startIdx: nRandomCoef
    });

    nRandomCoef += uniqueGroups.length;
  }

  // Random slopes
  for (const [varName, slopeSpec] of Object.entries(slopes)) {
    const { groups, values } = slopeSpec;
    const uniqueGroups = [...new Set(groups)];
    const groupMap = Object.fromEntries(uniqueGroups.map((g, i) => [g, i]));

    groupInfo.push({
      type: 'slope',
      variable: varName,
      groups: uniqueGroups,
      groupMap,
      nGroups: uniqueGroups.length,
      startIdx: nRandomCoef,
      values // store values for Z matrix construction
    });

    nRandomCoef += uniqueGroups.length;
  }

  // Build Z matrix (n × nRandomCoef)
  const Z = Array(n).fill(null).map(() => Array(nRandomCoef).fill(0));

  // Fill in random intercepts
  if (interceptGroup) {
    const info = groupInfo.find(g => g.type === 'intercept');
    for (let i = 0; i < n; i++) {
      const groupIdx = info.groupMap[interceptGroup[i]];
      Z[i][info.startIdx + groupIdx] = 1;
    }
  }

  // Fill in random slopes
  for (const info of groupInfo.filter(g => g.type === 'slope')) {
    const { values, groupMap, startIdx } = info;
    const groups = slopes[info.variable].groups;

    for (let i = 0; i < n; i++) {
      const groupIdx = groupMap[groups[i]];
      Z[i][startIdx + groupIdx] = values[i]; // slope value * group indicator
    }
  }

  return { Z, groupInfo, nRandomCoef };
}

/**
 * Initialize variance components
 */
function initializeVarianceComponents(groupInfo) {
  return groupInfo.map(() => ({ variance: 1.0 }));
}

/**
 * Update fixed effects given random effects
 */
function updateFixedEffects(X, Z, y, weights, u, theta) {
  const n = X.length;
  const p = X[0].length;

  // Compute residuals: y - Zu
  const Zu = new Array(n);
  for (let i = 0; i < n; i++) {
    Zu[i] = matrixVectorMultiply([Z[i]], u)[0];
  }
  const yResid = y.map((yi, i) => yi - Zu[i]);

  // Weighted least squares
  return weightedLeastSquares(X, yResid, weights);
}

/**
 * Update random effects given fixed effects
 */
function updateRandomEffects(X, Z, y, weights, beta, theta) {
  const n = X.length;
  const q = Z[0].length;

  // Compute residuals: y - Xβ
  const Xbeta = matrixVectorMultiply(X, beta);
  const yResid = y.map((yi, i) => yi - Xbeta[i]);

  // Build system: (Z'WZ + D^{-1})u = Z'Wy
  // where D = block diagonal variance matrix

  const sqrtW = weights.map(w => Math.sqrt(Math.max(w, 0)));
  const WZ = Z.map((row, i) => row.map(z => sqrtW[i] * z));
  const Wy = yResid.map((r, i) => sqrtW[i] * r);

  const ZtWZ = new Matrix(WZ).transpose().mmul(new Matrix(WZ));
  const ZtWy = new Matrix(WZ).transpose().mmul(Matrix.columnVector(Wy));

  // Add precision matrix (inverse of variance)
  for (let i = 0; i < theta.length; i++) {
    const { variance } = theta[i];
    const startIdx = theta[i].startIdx || 0;
    const nGroups = theta[i].nGroups || 1;

    for (let j = 0; j < nGroups; j++) {
      const idx = startIdx + j;
      if (idx < q) {
        ZtWZ.set(idx, idx, ZtWZ.get(idx, idx) + 1 / variance);
      }
    }
  }

  // Solve
  try {
    const u = ZtWZ.solve(ZtWy);
    return Array.from(u.getColumn(0));
  } catch (e) {
    const u = inverse(ZtWZ).mmul(ZtWy);
    return Array.from(u.getColumn(0));
  }
}

/**
 * Update variance components
 */
function updateVarianceComponents(u, groupInfo) {
  return groupInfo.map((info, idx) => {
    const startIdx = info.startIdx;
    const nGroups = info.nGroups;

    // Compute variance from random effects
    let sumSq = 0;
    for (let i = 0; i < nGroups; i++) {
      sumSq += u[startIdx + i] ** 2;
    }

    const variance = sumSq / nGroups;

    return {
      ...info,
      variance: Math.max(variance, 1e-6) // ensure positive
    };
  });
}

/**
 * Compute marginal log-likelihood via Laplace approximation
 */
function computeMarginalLogLikelihood(y, mu, u, theta, weights, family, groupInfo) {
  // Conditional log-likelihood: log p(y|β,u)
  const deviance = family.deviance(y, mu, weights);
  let logLik = -deviance / 2;

  // Random effects penalty: -0.5 * u'D^{-1}u
  for (let i = 0; i < groupInfo.length; i++) {
    const { variance, startIdx, nGroups } = theta[i];
    for (let j = 0; j < nGroups; j++) {
      const idx = startIdx + j;
      logLik -= 0.5 * u[idx] * u[idx] / variance;
    }
  }

  // Determinant term: -0.5 * log|D|
  for (let i = 0; i < groupInfo.length; i++) {
    const { variance, nGroups } = theta[i];
    logLik -= 0.5 * nGroups * Math.log(2 * Math.PI * variance);
  }

  return logLik;
}

/**
 * Compute standard errors for GLMM fixed effects
 */
function computeGLMMStandardErrors(X, Z, mu, beta, u, theta, weights, family, groupInfo) {
  // This is a simplified approximation
  // True standard errors would require Hessian of marginal likelihood

  const n = X.length;
  const p = X[0].length;

  // Compute working weights
  const variance = family.variance(mu);
  const mu_eta = family.link.mu_eta(family.link.linkfun(mu));

  const W = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.max(variance[i], 1e-10);
    const dmu = Math.max(Math.abs(mu_eta[i]), 1e-10);
    W[i] = weights[i] * dmu * dmu / v;
  }

  // Approximate covariance: (X'WX)^{-1}
  const sqrtW = W.map(w => Math.sqrt(Math.max(w, 0)));
  const WX = X.map((row, i) => row.map(x => sqrtW[i] * x));
  const XtWX = new Matrix(WX).transpose().mmul(new Matrix(WX));

  let covMatrix;
  try {
    covMatrix = inverse(XtWX);
  } catch (e) {
    // If singular, use SVD-based pseudoinverse
    const svd = new SingularValueDecomposition(XtWX);
    const s = svd.diagonal;
    const V = svd.rightSingularVectors;

    const Sinv = new Matrix(s.length, s.length);
    for (let i = 0; i < s.length; i++) {
      if (Math.abs(s[i]) > 1e-10) {
        Sinv.set(i, i, 1 / s[i]);
      }
    }

    const VMatrix = new Matrix(V);
    covMatrix = VMatrix.mmul(Sinv).mmul(VMatrix.transpose());
  }

  const standardErrors = new Array(p);
  for (let i = 0; i < p; i++) {
    standardErrors[i] = Math.sqrt(Math.max(covMatrix.get(i, i), 0));
  }

  return {
    standardErrors,
    covarianceMatrix: covMatrix.to2DArray()
  };
}

// ============================================================================
// Prediction Functions
// ============================================================================

/**
 * Predict from GLM
 */
export function predictGLM(model, X, options = {}) {
  const {
    type = 'response', // 'link', 'response', 'confidence', 'prediction'
    interval = false,
    level = 0.95,
    offset = null
  } = options;

  const n = X.length;
  const Xmat = model.intercept ? addIntercept(X) : X;
  const off = offset || Array(n).fill(0);

  // Linear predictor
  const eta = matrixVectorMultiply(Xmat, model.coefficients).map((e, i) => e + off[i]);

  // Create family object for inverse link
  const familyObj = createFamily({ family: model.family, link: model.link });

  if (type === 'link') {
    return interval ? computeIntervals(eta, Xmat, model, level, 'link') : eta;
  }

  // Response scale
  const mu = familyObj.link.linkinv(eta);

  if (type === 'response') {
    return interval ? computeIntervals(mu, Xmat, model, level, 'response', familyObj) : mu;
  }

  // Confidence or prediction intervals would go here
  return mu;
}

/**
 * Predict from GLMM
 */
export function predictGLMM(model, X, randomEffectsData, options = {}) {
  const {
    type = 'response',
    allowNewGroups = true,
    offset = null
  } = options;

  const n = X.length;
  const Xmat = model.intercept ? addIntercept(X) : X;
  const off = offset || Array(n).fill(0);

  // Build Z matrix for new data
  const { Z } = buildRandomEffectsMatrixForPrediction(
    n,
    randomEffectsData,
    model.groupInfo,
    allowNewGroups
  );

  // Linear predictor
  const eta = new Array(n);
  for (let i = 0; i < n; i++) {
    eta[i] = matrixVectorMultiply([Xmat[i]], model.fixedEffects)[0];
    eta[i] += matrixVectorMultiply([Z[i]], model.randomEffects)[0];
    eta[i] += off[i];
  }

  const familyObj = createFamily({ family: model.family, link: model.link });

  if (type === 'link') {
    return eta;
  }

  return familyObj.link.linkinv(eta);
}

/**
 * Build Z matrix for prediction (handles new groups)
 */
function buildRandomEffectsMatrixForPrediction(n, randomEffectsData, groupInfo, allowNewGroups) {
  const nRandomCoef = groupInfo.reduce((sum, info) => sum + info.nGroups, 0);
  const Z = Array(n).fill(null).map(() => Array(nRandomCoef).fill(0));

  // Handle random intercepts
  const interceptInfo = groupInfo.find(g => g.type === 'intercept');
  if (interceptInfo && randomEffectsData.intercept) {
    const groups = randomEffectsData.intercept;
    for (let i = 0; i < n; i++) {
      const group = groups[i];
      if (group in interceptInfo.groupMap) {
        const groupIdx = interceptInfo.groupMap[group];
        Z[i][interceptInfo.startIdx + groupIdx] = 1;
      } else if (!allowNewGroups) {
        throw new Error(`Unknown group: ${group}. Set allowNewGroups=true to predict with zero random effect.`);
      }
      // If new group and allowNewGroups=true, leave as 0 (population-level prediction)
    }
  }

  // Handle random slopes
  for (const info of groupInfo.filter(g => g.type === 'slope')) {
    const varName = info.variable;
    if (randomEffectsData.slopes && randomEffectsData.slopes[varName]) {
      const { groups, values } = randomEffectsData.slopes[varName];
      for (let i = 0; i < n; i++) {
        const group = groups[i];
        if (group in info.groupMap) {
          const groupIdx = info.groupMap[group];
          Z[i][info.startIdx + groupIdx] = values[i];
        } else if (!allowNewGroups) {
          throw new Error(`Unknown group: ${group}. Set allowNewGroups=true to predict with zero random effect.`);
        }
        // If new group and allowNewGroups=true, leave as 0
      }
    }
  }

  return { Z };
}

/**
 * Compute prediction intervals
 */
function computeIntervals(predictions, X, model, level, scale, familyObj = null) {
  const n = predictions.length;
  const p = model.p;
  const z = 1.96; // approximation for 95% CI

  const covMatrix = new Matrix(model.covarianceMatrix);
  const intervals = new Array(n);

  for (let i = 0; i < n; i++) {
    const xi = Matrix.rowVector(X[i]);
    const variance = xi.mmul(covMatrix).mmul(xi.transpose()).get(0, 0);
    const se = Math.sqrt(variance);

    intervals[i] = {
      fit: predictions[i],
      lower: predictions[i] - z * se,
      upper: predictions[i] + z * se
    };

    // Transform to response scale if needed
    if (scale === 'response' && familyObj) {
      intervals[i].lower = familyObj.link.linkinv([intervals[i].lower])[0];
      intervals[i].upper = familyObj.link.linkinv([intervals[i].upper])[0];
    }
  }

  return intervals;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Add intercept column to design matrix
 */
function addIntercept(X) {
  return X.map(row => [1, ...row]);
}

/**
 * Matrix-vector multiplication
 */
function matrixVectorMultiply(A, x) {
  return A.map(row => sum(row.map((aij, j) => aij * x[j])));
}

// ============================================================================
// Functional API Wrappers (for backward compatibility)
// ============================================================================

/**
 * Fit linear model (functional API)
 */
export function fitLM(X, y, options = {}) {
  return fitGLM(X, y, { ...options, family: 'gaussian' });
}

/**
 * Predict from linear model (functional API)
 */
export function predictLM(coefficients, X, options = {}) {
  const model = { coefficients, family: 'gaussian', link: 'identity', intercept: options.intercept !== false, p: coefficients.length };
  return predictGLM(model, X, options);
}

/**
 * Fit logistic model (functional API)
 */
export function fitLogit(X, y, options = {}) {
  return fitGLM(X, y, { ...options, family: 'binomial', link: 'logit' });
}

/**
 * Predict from logistic model (functional API)
 */
export function predictLogit(coefficients, X, options = {}) {
  const model = { coefficients, family: 'binomial', link: 'logit', intercept: options.intercept !== false, p: coefficients.length };
  return predictGLM(model, X, options);
}
