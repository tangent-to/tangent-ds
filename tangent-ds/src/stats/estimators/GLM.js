/**
 * Generalized Linear Model (GLM) Estimator
 *
 * Unified interface for GLM and GLMM with all families:
 * - Gaussian, Binomial, Poisson, Gamma, InverseGaussian, NegativeBinomial
 * - Optional random effects (intercepts and slopes)
 * - Supports weights, offsets, regularization
 * - lme4-style output (no p-values for mixed models)
 */

import { Estimator } from '../../core/estimators/estimator.js';
import { prepareXY } from '../../core/table.js';
import { fitGLM, fitGLMM, predictGLM, predictGLMM } from '../glm.js';
import { createFamily, getCanonicalLink } from '../families.js';
import { mean } from '../../core/math.js';
import { applyFormula } from '../../core/formula.js';

export class GLM extends Estimator {
  /**
   * @param {Object} params - Model parameters
   * @param {string} params.family - GLM family (gaussian, binomial, poisson, gamma, inverse_gaussian, negative_binomial)
   * @param {string} params.link - Link function (default: canonical link for family)
   * @param {Object} params.randomEffects - Random effects specification {intercept: [...], slopes: {...}}
   * @param {boolean} params.intercept - Include intercept term (default: true)
   * @param {number} params.maxIter - Maximum iterations (default: 100)
   * @param {number} params.tol - Convergence tolerance (default: 1e-8 for GLM, 1e-6 for GLMM)
   * @param {Object} params.regularization - Regularization {alpha, l1_ratio}
   * @param {string|number} params.dispersion - Dispersion estimation: 'estimate', 'fixed', or numeric
   * @param {number} params.theta - Theta parameter for negative binomial (default: 1)
   */
  constructor(params = {}) {
    super(params);

    // Default parameters
    this.params = {
      family: 'gaussian',
      link: null,
      randomEffects: null,
      intercept: true,
      maxIter: 100,
      tol: null, // Will be set based on whether GLMM
      regularization: null,
      dispersion: 'estimate',
      theta: 1,
      ...params
    };

    // Set default link if not specified
    if (!this.params.link) {
      this.params.link = getCanonicalLink(this.params.family);
    }

    // Set default tolerance based on model type
    if (this.params.tol === null) {
      this.params.tol = this.params.randomEffects ? 1e-6 : 1e-8;
    }

    this._model = null;
    this._isMixed = false;
    this._columnsX = null;
    this._columnY = null;
  }

  /**
   * Get coefficients (for backward compatibility with lm interface)
   */
  get coefficients() {
    if (!this.fitted) return null;
    return this._isMixed ? this._model.fixedEffects : this._model.coefficients;
  }

  /**
   * Get intercept flag (for backward compatibility)
   */
  get intercept() {
    return this.params.intercept;
  }

  /**
   * Fit the GLM or GLMM
   *
   * Supports multiple calling conventions:
   * - fit(X, y)
   * - fit(X, y, weights, offset)
   * - fit({ X, y, data })
   * - fit({ X, y, groups, data }) for mixed models
   * - fit('y ~ x1 + x2', data) - R-style formula
   * - fit({ formula: 'y ~ x1 + x2', data }) - formula in object
   */
  fit(...args) {
    let X, y, weights, offset, groups, randomEffectsData;

    // Check for formula-based input
    if (args.length >= 1 && typeof args[0] === 'string') {
      // Formula string: fit('y ~ x1 + x2', data)
      const formula = args[0];
      const data = args[1];

      if (!data) {
        throw new Error('Data is required when using formula syntax');
      }

      return this._fitWithFormula(formula, data, args[2] || {});
    }

    // Parse arguments
    if (args.length === 1 && typeof args[0] === 'object' && !Array.isArray(args[0])) {
      // Check if it's a formula-style object
      if (args[0].formula) {
        return this._fitWithFormula(args[0].formula, args[0].data, args[0]);
      }

      // Table-style: { X, y, data, groups, weights, offset }
      const opts = args[0];

      // Prepare X and y from table
      const prepared = prepareXY({
        X: opts.X,
        y: opts.y,
        data: opts.data,
        omit_missing: opts.omit_missing !== undefined ? opts.omit_missing : true
      });

      X = prepared.X;
      y = prepared.y;
      this._columnsX = prepared.columnsX;
      this._columnY = opts.y;

      // Extract groups for random effects
      if (opts.groups || this.params.randomEffects) {
        randomEffectsData = this._parseRandomEffects(opts, prepared.rows);
      }

      // Extract weights and offset if present in data
      if (opts.weights) {
        weights = this._extractColumn(opts.weights, opts.data, prepared.rows);
      }
      if (opts.offset) {
        offset = this._extractColumn(opts.offset, opts.data, prepared.rows);
      }
    } else {
      // Array-style: (X, y) or (X, y, weights, offset)
      X = args[0];
      y = args[1];
      weights = args[2];
      offset = args[3];
    }

    // Determine if this is a mixed model
    this._isMixed = !!(randomEffectsData || this.params.randomEffects);

    // Fit the model
    const options = {
      family: this.params.family,
      link: this.params.link,
      weights,
      offset,
      intercept: this.params.intercept,
      maxIter: this.params.maxIter,
      tol: this.params.tol,
      regularization: this.params.regularization,
      dispersion: this.params.dispersion
    };

    if (this._isMixed) {
      // Fit GLMM
      const randomEffects = randomEffectsData || this.params.randomEffects;
      this._model = fitGLMM(X, y, randomEffects, options);
    } else {
      // Fit GLM
      this._model = fitGLM(X, y, options);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Fit using R-style formula
   * @private
   */
  _fitWithFormula(formula, data, options = {}) {
    // Parse formula and apply to data
    const result = applyFormula(formula, data, {
      intercept: this.params.intercept
    });

    // Store column names for later use
    this._columnsX = result.columnNames.filter(c => c !== '(Intercept)');
    this._columnY = result.parsed.response.variable;
    this._formula = formula;

    // Extract weights and offset from options if provided
    const weights = options.weights;
    const offset = options.offset;

    // Determine if this is a mixed model
    this._isMixed = !!result.randomEffects;

    // Fit the model
    // Note: result.X already includes intercept from applyFormula, so don't add it again
    const fitOptions = {
      family: this.params.family,
      link: this.params.link,
      weights,
      offset,
      intercept: false, // intercept already in result.X from applyFormula
      maxIter: this.params.maxIter,
      tol: this.params.tol,
      regularization: this.params.regularization,
      dispersion: this.params.dispersion
    };

    if (this._isMixed) {
      // Fit GLMM with random effects from formula
      this._model = fitGLMM(result.X, result.y, result.randomEffects, fitOptions);
    } else {
      // Fit GLM
      this._model = fitGLM(result.X, result.y, fitOptions);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Parse random effects specification from table-style input
   */
  _parseRandomEffects(opts, rows) {
    const data = opts.data;
    const randomEffects = {};

    // Handle intercept groups
    if (opts.groups) {
      const groupCol = opts.groups;
      randomEffects.intercept = rows.map(i => data[i][groupCol]);
    }

    // Handle slopes (if specified in params)
    if (this.params.randomEffects && this.params.randomEffects.slopes) {
      randomEffects.slopes = {};
      for (const [varName, groupCol] of Object.entries(this.params.randomEffects.slopes)) {
        // Extract group assignments and values for this slope
        const groups = rows.map(i => data[i][groupCol]);
        const values = rows.map(i => data[i][varName]);
        randomEffects.slopes[varName] = { groups, values };
      }
    }

    return randomEffects;
  }

  /**
   * Extract a column from table data
   */
  _extractColumn(columnName, data, rows) {
    return rows.map(i => data[i][columnName]);
  }

  /**
   * Predict from the fitted model
   *
   * @param {Array|Object} X - Predictors or table-style object
   * @param {Object} options - Prediction options
   * @param {string} options.type - Prediction type: 'link', 'response' (default)
   * @param {boolean} options.interval - Compute confidence intervals (default: false)
   * @param {number} options.level - Confidence level (default: 0.95)
   * @param {boolean} options.allowNewGroups - For GLMM: allow new groups (default: true)
   * @returns {Array} Predictions
   */
  predict(X, options = {}) {
    if (!this.fitted) {
      throw new Error('Model has not been fitted yet. Call fit() first.');
    }

    let Xmat, offset, randomEffectsData;

    // Parse input
    if (typeof X === 'object' && !Array.isArray(X) && X.data) {
      // Table-style input
      const opts = X;
      const prepared = prepareXY({
        X: opts.X || this._columnsX,
        y: opts.y || this._columnY,
        data: opts.data,
        omit_missing: opts.omit_missing !== undefined ? opts.omit_missing : true
      });

      Xmat = prepared.X;

      // Extract offset if present
      if (opts.offset) {
        offset = this._extractColumn(opts.offset, opts.data, prepared.rows);
      }

      // Extract random effects data for GLMM
      if (this._isMixed && opts.groups) {
        randomEffectsData = this._parseRandomEffects(opts, prepared.rows);
      }
    } else {
      // Array-style input
      Xmat = X;
    }

    // Make predictions
    const predOptions = {
      type: options.type || 'response',
      interval: options.interval || false,
      level: options.level || 0.95,
      offset: offset || options.offset
    };

    if (this._isMixed) {
      predOptions.allowNewGroups = options.allowNewGroups !== undefined
        ? options.allowNewGroups
        : true;
      return predictGLMM(this._model, Xmat, randomEffectsData, predOptions);
    } else {
      return predictGLM(this._model, Xmat, predOptions);
    }
  }

  /**
   * Get model summary (lme4-style for mixed models)
   */
  summary() {
    if (!this.fitted) {
      throw new Error('Model has not been fitted yet. Call fit() first.');
    }

    if (this._isMixed) {
      return this._summaryGLMM();
    } else {
      return this._summaryGLM();
    }
  }

  /**
   * Format GLM summary
   */
  _summaryGLM() {
    const m = this._model;
    const family = m.family;
    const link = m.link;

    let output = `\nGeneralized Linear Model\n`;
    output += `Family: ${family}, Link: ${link}\n\n`;

    output += `Coefficients:\n`;
    output += `                Estimate  Std.Error  z value    95% CI\n`;

    const labels = this._getCoefLabels();
    for (let i = 0; i < m.coefficients.length; i++) {
      const label = labels[i].padEnd(15);
      const est = m.coefficients[i].toFixed(6).padStart(10);
      const se = m.standardErrors[i].toFixed(6).padStart(10);
      const z = (m.coefficients[i] / m.standardErrors[i]).toFixed(3).padStart(8);
      const ci = `[${m.confidenceIntervals[i].lower.toFixed(3)}, ${m.confidenceIntervals[i].upper.toFixed(3)}]`;

      output += `${label} ${est} ${se} ${z}  ${ci}\n`;
    }

    output += `\n`;
    output += `Null Deviance: ${m.nullDeviance.toFixed(4)} on ${m.n - 1} degrees of freedom\n`;
    output += `Residual Deviance: ${m.deviance.toFixed(4)} on ${m.dfResidual} degrees of freedom\n`;
    output += `AIC: ${m.aic.toFixed(2)}\n`;
    output += `BIC: ${m.bic.toFixed(2)}\n`;
    output += `Dispersion: ${m.dispersion.toFixed(4)}\n`;
    output += `Pseudo R²: ${m.pseudoR2.toFixed(4)}\n`;
    output += `Iterations: ${m.iterations}, Converged: ${m.converged}\n`;

    return output;
  }

  /**
   * Format GLMM summary (lme4-style, no p-values)
   */
  _summaryGLMM() {
    const m = this._model;
    const family = m.family;
    const link = m.link;

    let output = `\nGeneralized Linear Mixed Model\n`;
    output += `Family: ${family}, Link: ${link}\n\n`;

    output += `Fixed Effects:\n`;
    output += `                Estimate  Std.Error  z value    95% CI\n`;

    const labels = this._getCoefLabels();
    for (let i = 0; i < m.fixedEffects.length; i++) {
      const label = labels[i].padEnd(15);
      const est = m.fixedEffects[i].toFixed(6).padStart(10);
      const se = m.standardErrors[i].toFixed(6).padStart(10);
      const z = (m.fixedEffects[i] / m.standardErrors[i]).toFixed(3).padStart(8);
      const ci = `[${m.confidenceIntervals[i].lower.toFixed(3)}, ${m.confidenceIntervals[i].upper.toFixed(3)}]`;

      output += `${label} ${est} ${se} ${z}  ${ci}\n`;
    }

    output += `\nRandom Effects:\n`;
    output += ` Groups   Name        Variance  Std.Dev.\n`;

    for (let i = 0; i < m.varianceComponents.length; i++) {
      const comp = m.varianceComponents[i];
      const groupName = comp.type === 'intercept' ? 'group' : comp.variable;
      const effectName = comp.type === 'intercept' ? '(Intercept)' : comp.variable;
      const variance = comp.variance.toFixed(4).padStart(9);
      const stdDev = Math.sqrt(comp.variance).toFixed(4).padStart(9);

      output += ` ${groupName.padEnd(8)} ${effectName.padEnd(11)} ${variance} ${stdDev}\n`;
    }

    const nGroups = m.groupInfo[0]?.nGroups || 0;
    output += `\nNumber of obs: ${m.n}, groups: ${nGroups}\n`;
    output += `AIC: ${m.aic.toFixed(2)}, BIC: ${m.bic.toFixed(2)}\n`;
    output += `Log-Likelihood: ${m.logLikelihood.toFixed(2)}\n`;
    output += `Iterations: ${m.iterations}, Converged: ${m.converged}\n`;

    output += `\n⚠️  Note: p-values for fixed effects in mixed models are based on\n`;
    output += `    questionable assumptions, can be misleading, and there is no single\n`;
    output += `    universally agreed, correct method for computing them in the\n`;
    output += `    frequentist mixed-effects framework. Prefer effect estimates ± CIs\n`;
    output += `    and variance components.\n`;

    return output;
  }

  /**
   * Get coefficient labels
   */
  _getCoefLabels() {
    const labels = [];

    if (this._model.intercept) {
      labels.push('(Intercept)');
    }

    if (this._columnsX) {
      labels.push(...this._columnsX);
    } else {
      const nCoef = this._isMixed ? this._model.fixedEffects.length : this._model.coefficients.length;
      const start = this._model.intercept ? 1 : 0;
      for (let i = start; i < nCoef; i++) {
        labels.push(`X${i}`);
      }
    }

    return labels;
  }

  /**
   * Score the model (R² for regression families, accuracy for binomial)
   */
  score(yTrue, yPred) {
    // If called with table-style input
    if (arguments.length === 1 && typeof yTrue === 'object' && yTrue.data) {
      const predictions = this.predict(yTrue);
      const prepared = prepareXY({
        X: yTrue.X || this._columnsX,
        y: yTrue.y || this._columnY,
        data: yTrue.data,
        omit_missing: true
      });
      yTrue = prepared.y;
      yPred = predictions;
    }

    const family = this.params.family.toLowerCase();

    if (family === 'binomial') {
      // Classification accuracy
      const correct = yTrue.reduce((sum, yi, i) => {
        const pred = yPred[i] >= 0.5 ? 1 : 0;
        return sum + (yi === pred ? 1 : 0);
      }, 0);
      return correct / yTrue.length;
    } else {
      // R² for continuous families
      const yMean = mean(yTrue);
      const ssTot = yTrue.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
      const ssRes = yTrue.reduce((sum, yi, i) => sum + Math.pow(yi - yPred[i], 2), 0);
      return 1 - ssRes / ssTot;
    }
  }

  /**
   * Serialize to JSON
   */
  toJSON() {
    return {
      __class__: 'GLM',
      params: this.params,
      fitted: this.fitted,
      model: this._model,
      isMixed: this._isMixed,
      columnsX: this._columnsX,
      columnY: this._columnY
    };
  }

  /**
   * Deserialize from JSON
   */
  static fromJSON(obj) {
    const instance = new GLM(obj.params);
    instance.fitted = obj.fitted;
    instance._model = obj.model;
    instance._isMixed = obj.isMixed;
    instance._columnsX = obj.columnsX;
    instance._columnY = obj.columnY;
    return instance;
  }
}

// Attach static methods for functional API compatibility
import * as glmFunctional from '../glm.js';
Object.assign(GLM, glmFunctional);
