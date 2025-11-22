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
import { prepareX, prepareXY } from '../../core/table.js';
import { fitGLM, fitGLMM, predictGLM, predictGLMM } from '../glm.js';
import { fitMultinomial, predictMultinomial } from '../multinomial.js';
import { createFamily, getCanonicalLink } from '../families.js';
import { mean } from '../../core/math.js';
import { applyFormula } from '../../core/formula.js';

export class GLM extends Estimator {
  /**
   * @param {Object} params - Model parameters
   * @param {string} params.family - GLM family (gaussian, binomial, poisson, gamma, inverse_gaussian, negative_binomial)
   * @param {string} params.link - Link function (default: canonical link for family)
   * @param {string} params.multiclass - Multiclass strategy: 'ovr' (one-vs-rest), 'multinomial' (softmax), or null (binary/regression)
   * @param {Object} params.randomEffects - Random effects specification {intercept: [...], slopes: {...}}
   * @param {boolean} params.intercept - Include intercept term (default: true)
   * @param {number} params.maxIter - Maximum iterations (default: 100)
   * @param {number} params.tol - Convergence tolerance (default: 1e-8 for GLM, 1e-6 for GLMM)
   * @param {Object} params.regularization - Regularization {alpha, l1_ratio}
   * @param {string|number} params.dispersion - Dispersion estimation: 'estimate', 'fixed', or numeric
   * @param {number} params.theta - Theta parameter for negative binomial (default: 1)
   * @param {number} params.alpha - Significance level for confidence intervals (default: 0.05 for 95% CIs)
   * @param {boolean} params.compress - Compress model to save memory (default: false)
   * @param {boolean} params.keepFittedValues - Keep fitted values and residuals (default: true)
   * @param {boolean} params.warnOnNoConvergence - Warn if model doesn't converge (default: true)
   * @param {boolean} params.warnLargeDataset - Warn about large datasets in browser (default: true)
   */
  constructor(params = {}) {
    super(params);

    // Default parameters
    this.params = {
      family: 'gaussian',
      link: null,
      multiclass: null, // 'ovr' for one-vs-rest, 'multinomial' for softmax multiclass
      randomEffects: null,
      intercept: true,
      maxIter: 100,
      tol: null, // Will be set based on whether GLMM
      regularization: null,
      dispersion: 'estimate',
      theta: 1,
      alpha: 0.05, // Default 95% confidence intervals
      compress: false, // Memory optimization
      keepFittedValues: true, // Keep fitted values and residuals
      warnOnNoConvergence: true,
      warnLargeDataset: true,
      ...params,
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
    this._models = null; // For multiclass/multi-output: stores multiple models
    this._classes = null; // Unique class labels for multiclass
    this._targetNames = null; // Target names for multi-output
    this._isMulticlass = false;
    this._isMultiOutput = false;
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

      // Check if y is an array (multi-output)
      if (Array.isArray(opts.y)) {
        // Multi-output: fit separate model for each target
        return this._fitMultiOutput(opts);
      }

      // Check if multiclass is enabled and we have table-style data
      if (this.params.multiclass === 'ovr' && this.params.family === 'binomial' && opts.data) {
        // Multiclass one-vs-rest
        return this._fitMulticlass(opts);
      }

      // Check if multinomial is enabled
      if (this.params.multiclass === 'multinomial' && this.params.family === 'binomial' && opts.data) {
        // Multiclass multinomial (softmax)
        return this._fitMultinomial(opts);
      }

      // Standard single-model fitting
      return this._fitSingleFromOpts(opts);
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
      dispersion: this.params.dispersion,
    };

    if (this._isMixed) {
      // Fit GLMM
      const randomEffects = randomEffectsData || this.params.randomEffects;
      this._model = fitGLMM(X, y, randomEffects, options);
    } else {
      // Fit GLM
      this._model = fitGLM(X, y, options);
    }

    this._postFitProcessing();
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
      intercept: this.params.intercept,
    });

    // Store column names for later use
    this._columnsX = result.columnNames.filter((c) => c !== '(Intercept)');
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
      dispersion: this.params.dispersion,
    };

    if (this._isMixed) {
      // Fit GLMM with random effects from formula
      this._model = fitGLMM(result.X, result.y, result.randomEffects, fitOptions);
    } else {
      // Fit GLM
      this._model = fitGLM(result.X, result.y, fitOptions);
    }

    this._postFitProcessing();
    this.fitted = true;
    return this;
  }

  /**
   * Fit multiclass GLM using one-vs-rest strategy
   * @private
   */
  _fitMulticlass(opts) {
    const { data, y: yColumn } = opts;

    // Store original column names for predictions
    this._columnsX = opts.X;
    this._columnY = yColumn;

    // Extract y values to find classes
    const yValues = data.map((row) => row[yColumn]);
    this._classes = [...new Set(yValues)].sort();

    if (this._classes.length < 2) {
      throw new Error('Multiclass requires at least 2 classes');
    }

    if (this._classes.length === 2) {
      // Binary case - fit single model with 0/1 encoding
      console.warn('Only 2 classes detected. Consider using binary GLM without multiclass option.');
      this._isMulticlass = false;

      // Encode as 0/1
      const binaryData = data.map((row) => ({
        ...row,
        __binary_target__: row[yColumn] === this._classes[1] ? 1 : 0,
      }));

      return this._fitSingleFromOpts({
        ...opts,
        y: '__binary_target__',
        data: binaryData,
      });
    }

    // Multiclass case - fit one model per class
    console.log(
      `ℹ️  Multiclass: Fitting ${this._classes.length} binary models using one-vs-rest strategy.`,
    );
    this._models = {};

    for (const targetClass of this._classes) {
      // Create binary target: 1 if this class, 0 otherwise
      const binaryData = data.map((row) => ({
        ...row,
        __binary_target__: row[yColumn] === targetClass ? 1 : 0,
      }));

      // Create new GLM instance for this class
      const classModel = new GLM({
        ...this.params,
        multiclass: null, // Prevent recursion
      });

      classModel.fit({
        ...opts,
        y: '__binary_target__',
        data: binaryData,
      });

      this._models[targetClass] = classModel;
    }

    this._isMulticlass = true;
    this.fitted = true;
    return this;
  }

  /**
   * Fit multiclass GLM using true multinomial logistic regression (softmax)
   * Automatically converts categorical y column into K-1 binary indicators
   * @private
   */
  _fitMultinomial(opts) {
    const { data, y: yColumn } = opts;

    // Store original column names for predictions
    this._columnsX = opts.X;
    this._columnY = yColumn;

    // Extract y values to find classes
    const yValues = data.map((row) => row[yColumn]);
    this._classes = [...new Set(yValues)].sort();

    if (this._classes.length < 2) {
      throw new Error('Multinomial requires at least 2 classes');
    }

    if (this._classes.length === 2) {
      // Binary case - fit single model with 0/1 encoding
      console.warn('Only 2 classes detected. Consider using binary GLM without multiclass option.');
      this._isMulticlass = false;

      // Encode as 0/1
      const binaryData = data.map((row) => ({
        ...row,
        __binary_target__: row[yColumn] === this._classes[1] ? 1 : 0,
      }));

      return this._fitSingleFromOpts({
        ...opts,
        y: '__binary_target__',
        data: binaryData,
      });
    }

    // Multiclass case - create K-1 binary indicators and fit true multinomial
    console.log(
      `ℹ️  Multinomial: Fitting true multinomial model with K=${this._classes.length} classes (K-1=${
        this._classes.length - 1
      } parameters, joint optimization with softmax).`,
    );

    // Use first class as reference category (coded as all zeros)
    const referenceClass = this._classes[0];
    const nonReferenceClasses = this._classes.slice(1);

    // Create binary indicator columns for K-1 classes
    const indicatorColumns = nonReferenceClasses.map((cls) => `__indicator_${cls}__`);
    const dataWithIndicators = data.map((row) => {
      const newRow = { ...row };
      for (let i = 0; i < nonReferenceClasses.length; i++) {
        newRow[indicatorColumns[i]] = row[yColumn] === nonReferenceClasses[i] ? 1 : 0;
      }
      return newRow;
    });

    // Store original class names (not indicator column names) for summary and predictions
    this._targetNames = nonReferenceClasses;
    this._indicatorColumns = indicatorColumns; // Store mapping for internal use

    // Fit true multinomial using multi-output infrastructure
    return this._fitTrueMultinomial({
      ...opts,
      y: indicatorColumns,
      data: dataWithIndicators,
    });
  }

  /**
   * Fit multi-output models (multiple independent targets)
   * For multinomial logit: fit K-1 models for K classes (reference category has all zeros)
   * @private
   */
  _fitMultiOutput(opts) {
    const { data, y: yColumns } = opts;

    if (!Array.isArray(yColumns) || yColumns.length < 1) {
      throw new Error('Multi-output requires array of target column names');
    }

    // Store original column names
    this._columnsX = opts.X;
    this._columnY = yColumns;
    this._targetNames = yColumns;

    // Determine if this is multinomial logit (all targets are binary 0/1)
    const isMultinomial = this.params.family === 'binomial' &&
      this._checkIfMultinomial(data, yColumns);

    if (isMultinomial) {
      console.log(
        `ℹ️  Multinomial logit: Fitting true multinomial model (K-1=${yColumns.length} for K=${
          yColumns.length + 1
        } classes, joint optimization).`,
      );

      // Use true multinomial logistic regression
      return this._fitTrueMultinomial(opts);
    } else {
      console.warn(
        `⚠️  Multi-output: Fitting ${yColumns.length} independent models. Targets are modeled separately without considering correlations.`,
      );
    }

    this._models = {};

    for (const targetCol of yColumns) {
      // Fit separate model for each target
      const targetModel = new GLM({
        ...this.params,
        multiclass: null, // Prevent recursion
      });

      targetModel.fit({
        ...opts,
        y: targetCol,
      });

      this._models[targetCol] = targetModel;
    }

    this._isMultiOutput = true;
    this._isMultinomial = false;
    this.fitted = true;
    return this;
  }

  /**
   * Fit true multinomial logistic regression (joint optimization)
   * @private
   */
  _fitTrueMultinomial(opts) {
    const { data, y: yColumns } = opts;

    // Store column names
    this._columnsX = opts.X;
    this._columnY = yColumns;

    // Only set _targetNames if not already set (e.g., by _fitMultinomial)
    if (!this._targetNames) {
      this._targetNames = yColumns;
    }

    // Prepare X
    const prepared = prepareX({
      columns: opts.X,
      data: data,
      naOmit: opts.omit_missing !== undefined ? opts.omit_missing : true,
    });

    const X = prepared.X;

    // Create y vector with class indices
    // Reference class (all zeros) = 0, other classes = 1, 2, ..., K-1
    const y = data.map((row) => {
      // Check which target is 1
      for (let k = 0; k < yColumns.length; k++) {
        if (row[yColumns[k]] === 1) {
          return k + 1; // Classes 1, 2, ..., K-1
        }
      }
      return 0; // Reference class
    });

    // Fit multinomial model
    this._model = fitMultinomial(X, y, {
      intercept: this.params.intercept,
      maxIter: this.params.maxIter,
      tol: this.params.tol,
    });

    this._isMultiOutput = true;
    this._isMultinomial = true;
    this.fitted = true;
    return this;
  }

  /**
   * Check if multi-output targets represent multinomial logit
   * (all binary 0/1 and mutually exclusive)
   * @private
   */
  _checkIfMultinomial(data, yColumns) {
    // Check if all targets are binary (0 or 1)
    for (const col of yColumns) {
      const values = [...new Set(data.map((row) => row[col]))];
      const allBinary = values.every((v) => v === 0 || v === 1);
      if (!allBinary) {
        return false;
      }
    }

    // Check if targets are mutually exclusive (at most one 1 per row)
    for (const row of data) {
      const sum = yColumns.reduce((s, col) => s + row[col], 0);
      if (sum > 1) {
        return false; // Not mutually exclusive
      }
    }

    return true;
  }

  /**
   * Fit single model (non-multiclass)
   * @private
   */
  _fitSingleFromOpts(opts) {
    // Prepare X and y from table
    const prepared = prepareXY({
      X: opts.X,
      y: opts.y,
      data: opts.data,
      omit_missing: opts.omit_missing !== undefined ? opts.omit_missing : true,
    });

    const X = prepared.X;
    const y = prepared.y;
    this._columnsX = prepared.columnsX;
    this._columnY = opts.y;

    // Check dataset size and warn if needed
    this._checkDatasetSize(X, y, {
      warnLargeDataset: this.params.warnLargeDataset,
      largeSampleThreshold: 10000,
      largeFeatureThreshold: this.params.family === 'binomial' ? 50 : 100
    });

    // Extract groups for random effects
    let randomEffectsData;
    if (opts.groups) {
      randomEffectsData = this._parseRandomEffects(opts, prepared.rows);
    } else if (this.params.randomEffects) {
      // Random effects passed directly in constructor (already in correct format)
      randomEffectsData = this.params.randomEffects;
    }

    // Extract weights and offset if present in data
    let weights, offset;
    if (opts.weights) {
      weights = this._extractColumn(opts.weights, opts.data, prepared.rows);
    }
    if (opts.offset) {
      offset = this._extractColumn(opts.offset, opts.data, prepared.rows);
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
      dispersion: this.params.dispersion,
    };

    if (this._isMixed) {
      // Fit GLMM
      const randomEffects = randomEffectsData || this.params.randomEffects;
      this._model = fitGLMM(X, y, randomEffects, options);
    } else {
      // Fit GLM
      this._model = fitGLM(X, y, options);
    }

    this._postFitProcessing();
    this.fitted = true;
    return this;
  }

  /**
   * Post-fit processing: warnings and memory optimization
   * @private
   */
  _postFitProcessing() {
    if (!this._model) return;

    // Check convergence and add warning if needed
    if (!this._model.converged && this.params.warnOnNoConvergence) {
      const modelType = this._isMixed ? 'GLMM' : 'GLM';
      const message =
        `⚠️ ${modelType} did not converge after ${this._model.iterations} iterations.\n` +
        `Possible causes:\n` +
        `  • Ill-conditioned data (check for perfect separation or multicollinearity)\n` +
        `  • maxIter too low (current: ${this.params.maxIter})\n` +
        `  • Tolerance too strict (current: ${this.params.tol})\n` +
        `Recommendations:\n` +
        `  • Increase maxIter or adjust tol\n` +
        `  • Check model.summary() for coefficient estimates\n` +
        `  • Consider regularization`;

      console.warn(message);
      this._addWarning('convergence', message, {
        iterations: this._model.iterations,
        maxIter: this.params.maxIter,
        tol: this.params.tol
      });
    }

    // Apply memory optimizations if requested
    if (this.params.compress) {
      this._compressModel();
    }

    if (!this.params.keepFittedValues) {
      delete this._model.fitted;
      delete this._model.residuals;
      delete this._model.pearsonResiduals;
      delete this._model.devianceResiduals;
    }
  }

  /**
   * Compress model to reduce memory footprint
   * @private
   */
  _compressModel() {
    if (!this._model) return;

    const roundTo = 1e10; // 10 decimal places

    // Round coefficients
    if (this._model.coefficients) {
      this._model.coefficients = this._model.coefficients.map(c =>
        Math.round(c * roundTo) / roundTo
      );
    }

    // Round fixed effects (for GLMM)
    if (this._model.fixedEffects) {
      this._model.fixedEffects = this._model.fixedEffects.map(c =>
        Math.round(c * roundTo) / roundTo
      );
    }

    // Round standard errors
    if (this._model.standardErrors) {
      this._model.standardErrors = this._model.standardErrors.map(se =>
        Math.round(se * roundTo) / roundTo
      );
    }

    // Round confidence intervals
    if (this._model.confidenceIntervals) {
      this._model.confidenceIntervals = this._model.confidenceIntervals.map(ci => ({
        lower: Math.round(ci.lower * roundTo) / roundTo,
        upper: Math.round(ci.upper * roundTo) / roundTo
      }));
    }
  }

  /**
   * Parse random effects specification from table-style input
   */
  _parseRandomEffects(opts, rows) {
    const randomEffects = {};

    // Handle intercept groups
    if (opts.groups) {
      const groupCol = opts.groups;
      // rows is array of row objects, not indices
      randomEffects.intercept = rows.map((row) => row[groupCol]);
    }

    // Handle slopes (if specified in params)
    if (this.params.randomEffects && this.params.randomEffects.slopes) {
      randomEffects.slopes = {};
      for (const [varName, groupCol] of Object.entries(this.params.randomEffects.slopes)) {
        // Extract group assignments and values for this slope
        const groups = rows.map((row) => row[groupCol]);
        const values = rows.map((row) => row[varName]);
        randomEffects.slopes[varName] = { groups, values };
      }
    }

    return randomEffects;
  }

  /**
   * Extract a column from table data
   */
  _extractColumn(columnName, data, rows) {
    // rows is array of row objects, not indices
    return rows.map((row) => row[columnName]);
  }

  /**
   * Predict probabilities for multiclass (one-vs-rest)
   * @private
   */
  _predictProba(X, options = {}) {
    if (!this._isMulticlass) {
      throw new Error('_predictProba is only for multiclass models');
    }

    // Get predictions from each binary model
    const classPredictions = {};
    for (const [targetClass, model] of Object.entries(this._models)) {
      const preds = model.predict(X, { ...options, type: 'response' });
      classPredictions[targetClass] = preds;
    }

    // Determine number of samples
    const nSamples = classPredictions[this._classes[0]].length;

    // Build probability array for each sample
    const probabilities = Array(nSamples).fill(null).map(() => ({}));

    for (let i = 0; i < nSamples; i++) {
      // Collect raw predictions for this sample
      for (const cls of this._classes) {
        probabilities[i][cls] = classPredictions[cls][i];
      }

      // Normalize to sum to 1
      const total = Object.values(probabilities[i]).reduce((sum, p) => sum + p, 0);
      if (total > 0) {
        for (const cls of this._classes) {
          probabilities[i][cls] /= total;
        }
      } else {
        // If all zero, assign equal probability
        const equalProb = 1.0 / this._classes.length;
        for (const cls of this._classes) {
          probabilities[i][cls] = equalProb;
        }
      }
    }

    return probabilities;
  }

  /**
   * Predict class labels for multiclass (one-vs-rest)
   * @private
   */
  _predictClass(X, options = {}) {
    const probas = this._predictProba(X, options);

    return probas.map((probs) => {
      // Find class with highest probability
      let maxClass = null;
      let maxProb = -Infinity;

      for (const [cls, prob] of Object.entries(probs)) {
        if (prob > maxProb) {
          maxProb = prob;
          maxClass = cls;
        }
      }

      return maxClass;
    });
  }

  /**
   * Predict probabilities for multi-output multinomial logit using softmax
   * @private
   */
  _predictMultiOutputProba(X, options = {}) {
    if (!this._isMultiOutput) {
      throw new Error('_predictMultiOutputProba is only for multi-output models');
    }

    // Get predictions from each model (logits on probability scale)
    const modelPredictions = {};
    for (const [targetName, model] of Object.entries(this._models)) {
      const preds = model.predict(X, { ...options, type: 'response' });
      modelPredictions[targetName] = preds;
    }

    const nSamples = modelPredictions[this._targetNames[0]].length;

    // For multinomial logit, use softmax
    if (this._isMultinomial) {
      const probabilities = Array(nSamples).fill(null).map(() => ({}));

      for (let i = 0; i < nSamples; i++) {
        // Get predicted probabilities for each K-1 class
        const logits = {};
        for (const targetName of this._targetNames) {
          const p = modelPredictions[targetName][i];
          // Convert probability back to log-odds
          // Clamp to avoid infinities
          const pClamped = Math.max(1e-15, Math.min(1 - 1e-15, p));
          logits[targetName] = Math.log(pClamped / (1 - pClamped));
        }

        // Reference category has logit = 0
        logits['__reference__'] = 0;

        // Softmax with numerical stability: subtract max before exp
        const maxLogit = Math.max(...Object.values(logits));
        const expLogits = {};
        let sumExp = 0;
        for (const [cls, logit] of Object.entries(logits)) {
          expLogits[cls] = Math.exp(logit - maxLogit);
          sumExp += expLogits[cls];
        }

        // Normalize
        for (const cls of Object.keys(logits)) {
          probabilities[i][cls] = expLogits[cls] / sumExp;
        }
      }

      return probabilities;
    } else {
      // Non-multinomial multi-output: just return raw predictions
      const predictions = Array(nSamples).fill(null).map(() => ({}));
      for (let i = 0; i < nSamples; i++) {
        for (const targetName of this._targetNames) {
          predictions[i][targetName] = modelPredictions[targetName][i];
        }
      }
      return predictions;
    }
  }

  /**
   * Predict for multi-output models
   * @private
   */
  _predictMultiOutput(X, options = {}) {
    const type = options.type || 'response';

    // Handle true multinomial (joint model)
    if (this._isMultinomial && this._model && !this._models) {
      // Parse input
      let Xmat;
      if (typeof X === 'object' && !Array.isArray(X) && X.data) {
        const prepared = prepareX({
          columns: X.X || this._columnsX,
          data: X.data,
          naOmit: X.omit_missing !== undefined ? X.omit_missing : true,
        });
        Xmat = prepared.X;
      } else {
        Xmat = X;
      }

      // Predict using true multinomial model
      if (type === 'proba' || type === 'probs') {
        const probs = predictMultinomial(this._model, Xmat, { type: 'probs' });

        // Convert to object format with class names
        // If we have _classes (from _fitMultinomial), use actual class names including reference
        if (this._classes && this._classes.length > 0) {
          return probs.map((probArray) => {
            const probObj = {};
            probObj[this._classes[0]] = probArray[0]; // Reference class
            for (let k = 0; k < this._targetNames.length; k++) {
              probObj[this._targetNames[k]] = probArray[k + 1];
            }
            return probObj;
          });
        } else {
          // Fallback to old behavior (when fit with manual indicators)
          return probs.map((probArray) => {
            const probObj = { __reference__: probArray[0] };
            for (let k = 0; k < this._targetNames.length; k++) {
              probObj[this._targetNames[k]] = probArray[k + 1];
            }
            return probObj;
          });
        }
      } else {
        // Return class indices or names
        const classIndices = predictMultinomial(this._model, Xmat, { type: 'class' });

        // Convert to class names
        // If we have _classes (from _fitMultinomial), use actual class names
        if (this._classes && this._classes.length > 0) {
          return classIndices.map((idx) => this._classes[idx]);
        } else {
          // Fallback to old behavior (when fit with manual indicators)
          return classIndices.map((idx) => {
            if (idx === 0) return '__reference__';
            return this._targetNames[idx - 1];
          });
        }
      }
    }

    // Handle K-1 separate models (old approach)
    if (this._isMultinomial && type === 'proba') {
      return this._predictMultiOutputProba(X, options);
    }

    // For non-multinomial or type='response', return individual predictions
    const predictions = {};
    for (const [targetName, model] of Object.entries(this._models)) {
      predictions[targetName] = model.predict(X, options);
    }

    return predictions;
  }

  /**
   * Predict from the fitted model
   *
   * @param {Array|Object} X - Predictors or table-style object
   * @param {Object} options - Prediction options
   * @param {string} options.type - Prediction type: 'link', 'response', 'class' (multiclass), 'proba' (multiclass)
   * @param {boolean} options.interval - Compute confidence intervals (default: false)
   * @param {number} options.level - Confidence level (default: 0.95)
   * @param {boolean} options.allowNewGroups - For GLMM: allow new groups (default: true)
   * @returns {Array} Predictions
   */
  predict(X, options = {}) {
    this._ensureFitted('predict');

    // Handle multiclass predictions
    if (this._isMulticlass) {
      const type = options.type || 'class'; // Default to class labels for multiclass

      if (type === 'proba' || type === 'response') {
        return this._predictProba(X, options);
      } else if (type === 'class') {
        return this._predictClass(X, options);
      } else {
        throw new Error(
          `Invalid prediction type '${type}' for multiclass. Use 'class' or 'proba'.`,
        );
      }
    }

    // Handle multi-output predictions
    if (this._isMultiOutput) {
      return this._predictMultiOutput(X, options);
    }

    let Xmat, offset, randomEffectsData;

    // Parse input
    if (typeof X === 'object' && !Array.isArray(X) && X.data) {
      // Table-style input
      const opts = X;
      const prepared = prepareX({
        columns: opts.X || this._columnsX,
        data: opts.data,
        naOmit: opts.omit_missing !== undefined ? opts.omit_missing : true,
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
      offset: offset || options.offset,
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
   * @param {Object} options - Summary options
   * @param {number} options.alpha - Significance level for CIs (default: from constructor)
   */
  summary(options = {}) {
    this._ensureFitted('summary');

    const alpha = options.alpha !== undefined ? options.alpha : this.params.alpha;

    if (this._isMulticlass) {
      return this._summaryMulticlass(alpha);
    } else if (this._isMultiOutput) {
      return this._summaryMultiOutput(alpha);
    } else if (this._isMixed) {
      return this._summaryGLMM(alpha);
    } else {
      return this._summaryGLM(alpha);
    }
  }

  /**
   * Compute confidence intervals for coefficients
   * @param {number} alpha - Significance level (default: 0.05 for 95% CIs)
   * @returns {Array<Object>} Array of {lower, upper} for each coefficient
   */
  confint(alpha = null) {
    this._ensureFitted('confint');

    const a = alpha !== null ? alpha : this.params.alpha;
    const z = this._getZCritical(a);

    const coeffs = this._isMixed ? this._model.fixedEffects : this._model.coefficients;
    const ses = this._model.standardErrors;

    return coeffs.map((coef, i) => ({
      lower: coef - z * ses[i],
      upper: coef + z * ses[i],
    }));
  }

  /**
   * Compute p-values for coefficients (Wald test)
   *
   * Note: For mixed models, p-values are controversial and should be
   * interpreted with caution. Prefer confidence intervals.
   *
   * @returns {Array<number>} P-values for each coefficient
   */
  pvalues() {
    this._ensureFitted('pvalues');

    const coeffs = this._isMixed ? this._model.fixedEffects : this._model.coefficients;
    const ses = this._model.standardErrors;

    return coeffs.map((coef, i) => {
      const z = coef / ses[i];
      return 2 * (1 - this._normalCDF(Math.abs(z)));
    });
  }

  /**
   * Get z-critical value for given alpha
   * @private
   */
  _getZCritical(alpha) {
    // Two-tailed z-critical value
    // For 95% CI: alpha=0.05, z=1.96
    return this._normalQuantile(1 - alpha / 2);
  }

  /**
   * Standard normal CDF (approximation)
   * @private
   */
  _normalCDF(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t *
      (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }

  /**
   * Standard normal quantile (inverse CDF) - approximation
   * @private
   */
  _normalQuantile(p) {
    if (p <= 0 || p >= 1) {
      throw new Error('Probability must be between 0 and 1');
    }

    // Beasley-Springer-Moro algorithm
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

  /**
   * Jupyter notebook display support
   * Returns HTML representation for better notebook rendering
   */
  _repr_html_() {
    if (!this.fitted) {
      return '<p>Model not fitted yet. Call fit() first.</p>';
    }

    if (this._isMixed) {
      return this._summaryGLMMHTML();
    } else {
      return this._summaryGLMHTML();
    }
  }

  /**
   * Display method for Jupyter notebooks
   * Automatically called when object is displayed
   */
  [Symbol.for('nodejs.util.inspect.custom')]() {
    return this.summary();
  }

  /**
   * Format GLM summary
   */
  _summaryGLM(alpha = 0.05) {
    const m = this._model;
    const family = m.family;
    const link = m.link;
    const confidence = ((1 - alpha) * 100).toFixed(0);

    let output = `\nGeneralized Linear Model\n`;
    output += `Family: ${family}, Link: ${link}\n\n`;

    output += `Coefficients:\n`;

    const labels = this._getCoefLabels();
    const cis = this.confint(alpha);

    // Calculate dynamic column widths based on actual data
    const longestLabel = labels.length ? Math.max(...labels.map((label) => label.length)) : 0;
    const labelPadding = longestLabel + 2;

    // Format all values first to determine column widths
    const formattedRows = [];
    for (let i = 0; i < m.coefficients.length; i++) {
      formattedRows.push({
        label: labels[i],
        est: m.coefficients[i].toFixed(6),
        se: m.standardErrors[i].toFixed(6),
        z: (m.coefficients[i] / m.standardErrors[i]).toFixed(3),
        ci: `[${cis[i].lower.toFixed(3)}, ${cis[i].upper.toFixed(3)}]`
      });
    }

    // Calculate column widths
    const estWidth = Math.max(8, ...formattedRows.map(r => r.est.length));
    const seWidth = Math.max(9, ...formattedRows.map(r => r.se.length));
    const zWidth = Math.max(7, ...formattedRows.map(r => r.z.length));

    // Header
    const headerPadding = ''.padEnd(labelPadding);
    output += `${headerPadding}${'Estimate'.padStart(estWidth)}  ${'Std.Error'.padStart(seWidth)}  ${'z value'.padStart(zWidth)}    ${confidence}% CI\n`;

    // Data rows
    for (const row of formattedRows) {
      const label = row.label.padEnd(labelPadding);
      const est = row.est.padStart(estWidth);
      const se = row.se.padStart(seWidth);
      const z = row.z.padStart(zWidth);

      output += `${label} ${est}  ${se}  ${z}  ${row.ci}\n`;
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
   * Format multiclass GLM summary
   */
  _summaryMulticlass(alpha = 0.05) {
    const confidence = ((1 - alpha) * 100).toFixed(0);

    let output = `\nMulticlass GLM (One-vs-Rest)\n`;
    output += `Family: ${this.params.family}, Link: ${this.params.link}\n`;
    output += `Classes: ${this._classes.join(', ')}\n`;
    output += `Number of binary models: ${this._classes.length}\n\n`;

    // Show summary for each class model
    for (const [targetClass, model] of Object.entries(this._models)) {
      output += `${'='.repeat(70)}\n`;
      output += `Model for class: ${targetClass} (vs rest)\n`;
      output += `${'='.repeat(70)}\n`;
      output += model.summary({ alpha });
      output += '\n';
    }

    return output;
  }

  /**
   * Format multi-output GLM summary
   */
  _summaryMultiOutput(alpha = 0.05) {
    const confidence = ((1 - alpha) * 100).toFixed(0);

    let output = `\nMulti-Output GLM\n`;
    output += `Family: ${this.params.family}, Link: ${this.params.link}\n`;

    if (this._isMultinomial) {
      // True multinomial (joint optimization)
      if (this._model && !this._models) {
        return this._summaryTrueMultinomial(alpha);
      }

      // Old K-1 separate models
      output += `Type: Multinomial Logit (K-1=${this._targetNames.length} models for K=${
        this._targetNames.length + 1
      } classes)\n`;
      output += `Targets: ${this._targetNames.join(', ')} (+ reference category)\n\n`;
      output += `Note: Coefficients represent log-odds relative to the reference category.\n`;
    } else {
      output += `Type: Independent targets (${this._targetNames.length} separate models)\n`;
      output += `Targets: ${this._targetNames.join(', ')}\n\n`;
      output +=
        `⚠️  Note: Targets are modeled independently. Correlations between targets are not considered.\n`;
    }
    output += '\n';

    // Show summary for each target model
    for (const [targetName, model] of Object.entries(this._models)) {
      output += `${'='.repeat(70)}\n`;
      output += `Model for target: ${targetName}\n`;
      output += `${'='.repeat(70)}\n`;
      output += model.summary({ alpha });
      output += '\n';
    }

    return output;
  }

  /**
   * Format true multinomial logistic regression summary
   */
  _summaryTrueMultinomial(alpha = 0.05) {
    const m = this._model;
    const K = m.K;
    const confidence = ((1 - alpha) * 100).toFixed(0);
    const z = this._normalQuantile(1 - alpha / 2);

    let output = `\nMultinomial Logistic Regression\n`;
    output += `Family: multinomial, Link: softmax\n`;
    output += `Classes: K=${K} (reference + ${K - 1} modeled)\n`;

    // Show reference class name if available (from _fitMultinomial)
    if (this._classes && this._classes.length > 0) {
      output += `Reference class: ${this._classes[0]}\n`;
      output += `Modeled classes: ${this._targetNames.join(', ')}\n\n`;
    } else {
      // Fallback to old behavior (when fit with manual indicators)
      output += `Targets: ${this._targetNames.join(', ')} (reference category: all zeros)\n\n`;
    }

    // Coefficients for each class
    const labels = this._getCoefLabels();
    const longestLabel = labels.length ? Math.max(...labels.map((label) => label.length)) : 0;
    const labelPadding = Math.max(15, longestLabel + 2);

    // Pre-compute all formatted values to determine column widths
    const allFormattedRows = [];
    for (let k = 0; k < K - 1; k++) {
      for (let j = 0; j < m.p; j++) {
        const lower = (m.coefficients[k][j] - z * m.standardErrors[k][j]).toFixed(3);
        const upper = (m.coefficients[k][j] + z * m.standardErrors[k][j]).toFixed(3);
        allFormattedRows.push({
          est: m.coefficients[k][j].toFixed(6),
          se: m.standardErrors[k][j].toFixed(6),
          z: (m.coefficients[k][j] / m.standardErrors[k][j]).toFixed(3),
          ci: `[${lower}, ${upper}]`
        });
      }
    }

    // Calculate column widths
    const estWidth = Math.max(8, ...allFormattedRows.map(r => r.est.length));
    const seWidth = Math.max(9, ...allFormattedRows.map(r => r.se.length));
    const zWidth = Math.max(7, ...allFormattedRows.map(r => r.z.length));

    for (let k = 0; k < K - 1; k++) {
      const className = this._targetNames[k];
      output += `${'='.repeat(70)}\n`;
      output += `Class: ${className} (vs reference)\n`;
      output += `${'='.repeat(70)}\n`;

      const headerPadding = ''.padEnd(labelPadding);
      output += `${headerPadding}${'Estimate'.padStart(estWidth)}  ${'Std.Error'.padStart(seWidth)}  ${'z value'.padStart(zWidth)}    ${confidence}% CI\n`;

      for (let j = 0; j < m.p; j++) {
        const rowIdx = k * m.p + j;
        const label = labels[j].padEnd(labelPadding);
        const est = allFormattedRows[rowIdx].est.padStart(estWidth);
        const se = allFormattedRows[rowIdx].se.padStart(seWidth);
        const zval = allFormattedRows[rowIdx].z.padStart(zWidth);

        output += `${label} ${est}  ${se}  ${zval}  ${allFormattedRows[rowIdx].ci}\n`;
      }
      output += '\n';
    }

    output += `${'='.repeat(70)}\n`;
    output += `Model Fit\n`;
    output += `${'='.repeat(70)}\n`;
    output += `Null Deviance: ${m.nullDeviance.toFixed(4)}\n`;
    output += `Residual Deviance: ${m.deviance.toFixed(4)}\n`;
    output += `AIC: ${m.aic.toFixed(2)}\n`;
    output += `BIC: ${m.bic.toFixed(2)}\n`;
    output += `Pseudo R²: ${m.pseudoR2.toFixed(4)}\n`;
    output += `Log-Likelihood: ${m.logLikelihood.toFixed(4)}\n`;
    output += `Iterations: ${m.iterations}, Converged: ${m.converged}\n`;
    output += `Number of observations: ${m.n}\n`;
    output += `Number of parameters: ${m.nParams}\n`;

    return output;
  }

  /**
   * Format GLM summary as HTML for Jupyter
   */
  _summaryGLMHTML() {
    const m = this._model;
    const family = m.family;
    const link = m.link;

    let html = '<div style="font-family: monospace; white-space: pre;">';

    // Show warnings if any
    if (this.hasWarnings()) {
      const warnings = this.getWarnings();
      html += '<div style="padding: 0.75em; margin-bottom: 1em; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;">';
      html += `<strong style="color: #856404;">⚠️ ${warnings.length} Warning${warnings.length > 1 ? 's' : ''}</strong>`;
      html += '<ul style="margin: 0.5em 0; padding-left: 1.5em;">';
      warnings.forEach(w => {
        const firstLine = w.message.split('\n')[0];
        html += `<li><strong>${w.type}:</strong> ${firstLine}</li>`;
      });
      html += '</ul></div>';
    }

    html += '<h3>Generalized Linear Model</h3>';
    html += `<p><strong>Family:</strong> ${family}, <strong>Link:</strong> ${link}</p>`;

    html += '<h4>Coefficients:</h4>';
    html += '<table style="border-collapse: collapse; font-family: monospace;">';
    html += '<tr style="border-bottom: 1px solid #ccc;">';
    html += '<th style="text-align: left; padding: 5px;"></th>';
    html += '<th style="text-align: right; padding: 5px;">Estimate</th>';
    html += '<th style="text-align: right; padding: 5px;">Std.Error</th>';
    html += '<th style="text-align: right; padding: 5px;">z value</th>';
    html += '<th style="text-align: left; padding: 5px;">95% CI</th>';
    html += '</tr>';

    const labels = this._getCoefLabels();
    for (let i = 0; i < m.coefficients.length; i++) {
      const label = labels[i];
      const est = m.coefficients[i].toFixed(6);
      const se = m.standardErrors[i].toFixed(6);
      const z = (m.coefficients[i] / m.standardErrors[i]).toFixed(3);
      const ci = `[${m.confidenceIntervals[i].lower.toFixed(3)}, ${
        m.confidenceIntervals[i].upper.toFixed(3)
      }]`;

      html += '<tr>';
      html += `<td style="text-align: left; padding: 5px;">${label}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${est}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${se}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${z}</td>`;
      html += `<td style="text-align: left; padding: 5px;">${ci}</td>`;
      html += '</tr>';
    }
    html += '</table>';

    html += '<h4>Model Fit:</h4>';
    html += '<table style="font-family: monospace;">';
    html += `<tr><td>Null Deviance:</td><td>${m.nullDeviance.toFixed(4)} on ${
      m.n - 1
    } degrees of freedom</td></tr>`;
    html += `<tr><td>Residual Deviance:</td><td>${
      m.deviance.toFixed(4)
    } on ${m.dfResidual} degrees of freedom</td></tr>`;
    html += `<tr><td>AIC:</td><td>${m.aic.toFixed(2)}</td></tr>`;
    html += `<tr><td>BIC:</td><td>${m.bic.toFixed(2)}</td></tr>`;
    html += `<tr><td>Dispersion:</td><td>${m.dispersion.toFixed(4)}</td></tr>`;
    html += `<tr><td>Pseudo R²:</td><td>${m.pseudoR2.toFixed(4)}</td></tr>`;
    html += `<tr><td>Iterations:</td><td>${m.iterations}, Converged: ${m.converged}</td></tr>`;
    html += '</table>';
    html += '</div>';

    return html;
  }

  /**
   * Format GLMM summary (lme4-style, no p-values)
   */
  _summaryGLMM(alpha = 0.05) {
    const m = this._model;
    const family = m.family;
    const link = m.link;
    const confidence = ((1 - alpha) * 100).toFixed(0);

    let output = `\nGeneralized Linear Mixed Model\n`;
    output += `Family: ${family}, Link: ${link}\n\n`;

    output += `Fixed Effects:\n`;

    const labels = this._getCoefLabels();
    const cis = this.confint(alpha);

    // Calculate dynamic column widths based on actual data
    const longestLabel = labels.length ? Math.max(...labels.map((label) => label.length)) : 0;
    const labelPadding = Math.max(15, longestLabel + 2);

    // Format all values first to determine column widths
    const formattedRows = [];
    for (let i = 0; i < m.fixedEffects.length; i++) {
      formattedRows.push({
        label: labels[i],
        est: m.fixedEffects[i].toFixed(6),
        se: m.standardErrors[i].toFixed(6),
        z: (m.fixedEffects[i] / m.standardErrors[i]).toFixed(3),
        ci: `[${cis[i].lower.toFixed(3)}, ${cis[i].upper.toFixed(3)}]`
      });
    }

    // Calculate column widths
    const estWidth = Math.max(8, ...formattedRows.map(r => r.est.length));
    const seWidth = Math.max(9, ...formattedRows.map(r => r.se.length));
    const zWidth = Math.max(7, ...formattedRows.map(r => r.z.length));

    // Header
    const headerPadding = ''.padEnd(labelPadding);
    output += `${headerPadding}${'Estimate'.padStart(estWidth)}  ${'Std.Error'.padStart(seWidth)}  ${'z value'.padStart(zWidth)}    ${confidence}% CI\n`;

    // Data rows
    for (const row of formattedRows) {
      const label = row.label.padEnd(labelPadding);
      const est = row.est.padStart(estWidth);
      const se = row.se.padStart(seWidth);
      const z = row.z.padStart(zWidth);

      output += `${label} ${est}  ${se}  ${z}  ${row.ci}\n`;
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
   * Format GLMM summary as HTML for Jupyter
   */
  _summaryGLMMHTML() {
    const m = this._model;
    const family = m.family;
    const link = m.link;

    let html = '<div style="font-family: monospace; white-space: pre;">';

    // Show warnings if any
    if (this.hasWarnings()) {
      const warnings = this.getWarnings();
      html += '<div style="padding: 0.75em; margin-bottom: 1em; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;">';
      html += `<strong style="color: #856404;">⚠️ ${warnings.length} Warning${warnings.length > 1 ? 's' : ''}</strong>`;
      html += '<ul style="margin: 0.5em 0; padding-left: 1.5em;">';
      warnings.forEach(w => {
        const firstLine = w.message.split('\n')[0];
        html += `<li><strong>${w.type}:</strong> ${firstLine}</li>`;
      });
      html += '</ul></div>';
    }

    html += '<h3>Generalized Linear Mixed Model</h3>';
    html += `<p><strong>Family:</strong> ${family}, <strong>Link:</strong> ${link}</p>`;

    html += '<h4>Fixed Effects:</h4>';
    html += '<table style="border-collapse: collapse; font-family: monospace;">';
    html += '<tr style="border-bottom: 1px solid #ccc;">';
    html += '<th style="text-align: left; padding: 5px;"></th>';
    html += '<th style="text-align: right; padding: 5px;">Estimate</th>';
    html += '<th style="text-align: right; padding: 5px;">Std.Error</th>';
    html += '<th style="text-align: right; padding: 5px;">z value</th>';
    html += '<th style="text-align: left; padding: 5px;">95% CI</th>';
    html += '</tr>';

    const labels = this._getCoefLabels();
    for (let i = 0; i < m.fixedEffects.length; i++) {
      const label = labels[i];
      const est = m.fixedEffects[i].toFixed(6);
      const se = m.standardErrors[i].toFixed(6);
      const z = (m.fixedEffects[i] / m.standardErrors[i]).toFixed(3);
      const ci = `[${m.confidenceIntervals[i].lower.toFixed(3)}, ${
        m.confidenceIntervals[i].upper.toFixed(3)
      }]`;

      html += '<tr>';
      html += `<td style="text-align: left; padding: 5px;">${label}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${est}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${se}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${z}</td>`;
      html += `<td style="text-align: left; padding: 5px;">${ci}</td>`;
      html += '</tr>';
    }
    html += '</table>';

    html += '<h4>Random Effects:</h4>';
    html += '<table style="border-collapse: collapse; font-family: monospace;">';
    html += '<tr style="border-bottom: 1px solid #ccc;">';
    html += '<th style="text-align: left; padding: 5px;">Groups</th>';
    html += '<th style="text-align: left; padding: 5px;">Name</th>';
    html += '<th style="text-align: right; padding: 5px;">Variance</th>';
    html += '<th style="text-align: right; padding: 5px;">Std.Dev.</th>';
    html += '</tr>';

    for (let i = 0; i < m.varianceComponents.length; i++) {
      const comp = m.varianceComponents[i];
      const groupName = comp.type === 'intercept' ? 'group' : comp.variable;
      const effectName = comp.type === 'intercept' ? '(Intercept)' : comp.variable;
      const variance = comp.variance.toFixed(4);
      const stdDev = Math.sqrt(comp.variance).toFixed(4);

      html += '<tr>';
      html += `<td style="text-align: left; padding: 5px;">${groupName}</td>`;
      html += `<td style="text-align: left; padding: 5px;">${effectName}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${variance}</td>`;
      html += `<td style="text-align: right; padding: 5px;">${stdDev}</td>`;
      html += '</tr>';
    }
    html += '</table>';

    const nGroups = m.groupInfo[0]?.nGroups || 0;
    html += '<h4>Model Fit:</h4>';
    html += '<table style="font-family: monospace;">';
    html += `<tr><td>Number of obs:</td><td>${m.n}, groups: ${nGroups}</td></tr>`;
    html += `<tr><td>AIC:</td><td>${m.aic.toFixed(2)}</td></tr>`;
    html += `<tr><td>BIC:</td><td>${m.bic.toFixed(2)}</td></tr>`;
    html += `<tr><td>Log-Likelihood:</td><td>${m.logLikelihood.toFixed(2)}</td></tr>`;
    html += `<tr><td>Iterations:</td><td>${m.iterations}, Converged: ${m.converged}</td></tr>`;
    html += '</table>';

    html += '<p style="font-size: 0.9em; color: #666; margin-top: 10px;">';
    html +=
      '⚠️ Note: p-values for fixed effects in mixed models are based on questionable assumptions, ';
    html +=
      'can be misleading, and there is no single universally agreed, correct method for computing them ';
    html +=
      'in the frequentist mixed-effects framework. Prefer effect estimates ± CIs and variance components.';
    html += '</p>';
    html += '</div>';

    return html;
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
      const nCoef = this._isMixed
        ? this._model.fixedEffects.length
        : this._model.coefficients.length;
      const start = this._model.intercept ? 1 : 0;
      for (let i = start; i < nCoef; i++) {
        labels.push(`X${i}`);
      }
    }

    return labels;
  }

  /**
   * Score the model (R² for regression families, accuracy for binomial/multiclass)
   */
  score(yTrue, yPred) {
    // If called with table-style input
    if (arguments.length === 1 && typeof yTrue === 'object' && yTrue.data) {
      const predictions = this.predict(yTrue);

      // For multiclass, extract target directly (don't encode to numbers)
      if (this._isMulticlass) {
        const yColumn = yTrue.y || this._columnY;
        yTrue = yTrue.data.map((row) => row[yColumn]);
        yPred = predictions;
      } else {
        const prepared = prepareXY({
          X: yTrue.X || this._columnsX,
          y: yTrue.y || this._columnY,
          data: yTrue.data,
          omit_missing: true,
        });
        yTrue = prepared.y;
        yPred = predictions;
      }
    }

    const family = this.params.family.toLowerCase();

    if (this._isMulticlass || family === 'binomial') {
      // Classification accuracy (multiclass or binary)
      const correct = yTrue.reduce((sum, yi, i) => {
        let pred = yPred[i];

        // Handle binary case
        if (!this._isMulticlass && typeof pred === 'number') {
          pred = pred >= 0.5 ? 1 : 0;
        }

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
      models: this._models
        ? Object.fromEntries(
          Object.entries(this._models).map(([cls, model]) => [cls, model.toJSON()]),
        )
        : null,
      classes: this._classes,
      targetNames: this._targetNames,
      isMulticlass: this._isMulticlass,
      isMultiOutput: this._isMultiOutput,
      isMultinomial: this._isMultinomial,
      isMixed: this._isMixed,
      columnsX: this._columnsX,
      columnY: this._columnY,
    };
  }

  /**
   * Deserialize from JSON
   */
  static fromJSON(obj) {
    const instance = new GLM(obj.params);
    instance.fitted = obj.fitted;
    instance._model = obj.model;
    instance._classes = obj.classes;
    instance._targetNames = obj.targetNames;
    instance._isMulticlass = obj.isMulticlass;
    instance._isMultiOutput = obj.isMultiOutput;
    instance._isMultinomial = obj.isMultinomial;
    instance._isMixed = obj.isMixed;
    instance._columnsX = obj.columnsX;
    instance._columnY = obj.columnY;

    // Deserialize multiclass/multi-output models
    if (obj.models) {
      instance._models = {};
      for (const [cls, modelJSON] of Object.entries(obj.models)) {
        instance._models[cls] = GLM.fromJSON(modelJSON);
      }
    }

    return instance;
  }
}

// Attach static methods for functional API compatibility
import * as glmFunctional from '../glm.js';
Object.assign(GLM, glmFunctional);
