/**
 * Base estimator classes for @tangent.to/ds
 *
 * Provides a lightweight, scikit-learn inspired base:
 *  - Estimator: common utilities (params, serialization)
 *  - Regressor: adds a default R^2 scoring helper
 *  - Classifier: adds a default accuracy scoring helper
 *  - Transformer: adds fitTransform convenience
 *
 * Subclasses should implement `fit(...)` and `predict(...)` / `transform(...)`
 */

import { prepareX, prepareXY } from "../table.js";
import { mean } from "../math.js";

/**
 * Minimal base Estimator
 */
export class Estimator {
  /**
   * @param {Object} params Hyperparameters / options for the estimator
   */
  constructor(params = {}) {
    this.params = { ...params };
    this.fitted = false;
    // place to store internal learned attributes (e.g. coefficients)
    this._state = {};
    // track warnings during fitting
    this._warnings = [];
  }

  /**
   * Check if model is fitted
   * @returns {boolean}
   */
  isFitted() {
    return !!this.fitted;
  }

  /**
   * Ensure model is fitted before executing method
   * Throws informative error with Observable-specific guidance
   *
   * @param {string} methodName - Name of the method being called
   * @throws {Error} If model is not fitted
   * @private
   */
  _ensureFitted(methodName) {
    if (!this.fitted) {
      const className = this.constructor.name;
      throw new Error(
        `${className}.${methodName}() requires a fitted model.\n\n` +
        `Please call ${className}.fit() first before using ${methodName}().\n\n` +
        `💡 Observable Tip: Ensure the cell calling fit() executes before ` +
        `cells that use ${methodName}(). You can check fitted state with ` +
        `model.isFitted() to avoid this error in reactive cells.`
      );
    }
  }

  /**
   * Get comprehensive model state
   * @returns {Object} State information including fitted status, memory estimate, warnings
   */
  getState() {
    return {
      fitted: this.fitted,
      className: this.constructor.name,
      params: this.getParams(),
      memoryEstimate: this._estimateMemoryUsage(),
      warnings: this._warnings.length,
      hasWarnings: this._warnings.length > 0
    };
  }

  /**
   * Estimate memory usage in MB
   * @returns {number} Estimated memory in megabytes
   * @private
   */
  _estimateMemoryUsage() {
    if (!this.fitted) return 0;

    try {
      const json = this.toJSON();
      const jsonStr = JSON.stringify(json);
      return (jsonStr.length * 2) / (1024 * 1024);
    } catch (e) {
      return -1;
    }
  }

  /**
   * Get memory usage in human-readable format
   * @returns {string} Memory usage string (e.g., "2.3 MB" or "145 KB")
   */
  getMemoryUsage() {
    const mb = this._estimateMemoryUsage();
    if (mb < 0) return "Unknown";
    if (mb < 0.1) return `${(mb * 1024).toFixed(1)} KB`;
    return `${mb.toFixed(2)} MB`;
  }

  /**
   * Add a warning to the model
   * @param {string} type - Warning type (e.g., 'convergence', 'memory', 'performance')
   * @param {string} message - Warning message
   * @param {Object} metadata - Additional metadata
   * @private
   */
  _addWarning(type, message, metadata = {}) {
    this._warnings.push({
      type,
      message,
      timestamp: new Date().toISOString(),
      ...metadata
    });
  }

  /**
   * Get all warnings
   * @returns {Array<Object>} Array of warning objects
   */
  getWarnings() {
    return this._warnings.slice();
  }

  /**
   * Check if model has warnings
   * @returns {boolean}
   */
  hasWarnings() {
    return this._warnings.length > 0;
  }

  /**
   * Clear all warnings
   */
  clearWarnings() {
    this._warnings = [];
  }

  /**
   * Get warnings of a specific type
   * @param {string} type - Warning type
   * @returns {Array<Object>} Filtered warnings
   */
  getWarningsByType(type) {
    return this._warnings.filter(w => w.type === type);
  }

  /**
   * Check dataset size and warn if potentially problematic
   * @param {Array} X - Design matrix
   * @param {Array} y - Response vector
   * @param {Object} options - Warning options
   * @private
   */
  _checkDatasetSize(X, y, options = {}) {
    const {
      warnLargeDataset = true,
      largeSampleThreshold = 10000,
      largeFeatureThreshold = 100
    } = options;

    if (!warnLargeDataset) return;

    const n = X.length;
    const p = X[0]?.length || 0;
    const isBrowser = typeof window !== 'undefined';

    if (n > largeSampleThreshold && isBrowser) {
      const message =
        `⚠️ Large dataset: Fitting on ${n.toLocaleString()} samples may be slow in browser environments.\n` +
        `Consider:\n` +
        `  • Using a sample for interactive development\n` +
        `  • Switching to Node.js for production fitting\n` +
        `  • Using incremental/batch fitting if available`;

      console.warn(message);
      this._addWarning('performance', message, { n, p, environment: 'browser' });
    }

    if (p > largeFeatureThreshold) {
      const message =
        `⚠️ High dimensionality: ${p} features may cause performance or convergence issues.\n` +
        `Consider:\n` +
        `  • Feature selection or dimensionality reduction\n` +
        `  • Regularization to prevent overfitting\n` +
        `  • Checking for multicollinearity`;

      console.warn(message);
      this._addWarning('performance', message, { n, p });
    }
  }

  /**
   * Observable/Jupyter HTML representation
   * @returns {string} HTML representation
   */
  _repr_html_() {
    const className = this.constructor.name;

    if (!this.fitted) {
      return `
        <div style="padding: 1em; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; font-family: sans-serif;">
          <strong style="color: #856404;">⚠️ ${className} - Not Fitted</strong>
          <p style="margin: 0.5em 0;">This model has not been fitted yet.</p>
          <p style="margin: 0.5em 0; font-size: 0.9em;">
            Call <code style="background: #f8f9fa; padding: 2px 6px; border-radius: 3px;">model.fit(X, y)</code> first.
          </p>
          <p style="margin: 0.5em 0; font-size: 0.85em; color: #666;">
            <em>💡 Observable: Ensure the fit() cell executes before cells that use the model.</em>
          </p>
        </div>
      `;
    }

    const state = this.getState();
    let html = `
      <div style="font-family: sans-serif; border: 1px solid #dee2e6; border-radius: 4px; overflow: hidden;">
        <div style="padding: 0.75em 1em; background: #f8f9fa; border-bottom: 1px solid #dee2e6;">
          <strong>${className}</strong>
          <span style="margin-left: 1em; font-size: 0.9em; color: #28a745;">✓ Fitted</span>
          <span style="margin-left: 1em; font-size: 0.85em; color: #6c757d;">
            Memory: ${this.getMemoryUsage()}
          </span>
        </div>
    `;

    if (this.hasWarnings()) {
      const warnings = this.getWarnings();
      html += `
        <div style="padding: 0.75em 1em; background: #fff3cd; border-bottom: 1px solid #ffc107;">
          <strong style="color: #856404;">⚠️ ${warnings.length} Warning${warnings.length > 1 ? 's' : ''}</strong>
          <details style="margin-top: 0.5em;">
            <summary style="cursor: pointer; font-size: 0.9em;">Show details</summary>
            <ul style="margin: 0.5em 0; padding-left: 1.5em; font-size: 0.85em;">
      `;

      warnings.forEach(w => {
        const firstLine = w.message.split('\n')[0];
        html += `<li><strong>${w.type}:</strong> ${firstLine}</li>`;
      });

      html += `
            </ul>
          </details>
        </div>
      `;
    }

    html += `
        <div style="padding: 0.75em 1em;">
          <details>
            <summary style="cursor: pointer; font-weight: 500;">Parameters</summary>
            <pre style="margin: 0.5em 0; padding: 0.5em; background: #f8f9fa; border-radius: 3px; font-size: 0.85em; overflow-x: auto;">${JSON.stringify(this.params, null, 2)}</pre>
          </details>
        </div>
      </div>
    `;

    return html;
  }

  /**
   * Node.js inspect customization
   */
  [Symbol.for('nodejs.util.inspect.custom')]() {
    const className = this.constructor.name;
    if (!this.fitted) {
      return `${className}(not fitted)`;
    }

    const state = this.getState();
    const warnings = state.hasWarnings ? ` [${state.warnings} warnings]` : '';
    return `${className}(fitted, ${this.getMemoryUsage()})${warnings}`;
  }

  /**
   * Set parameters (mutates instance).
   * @param {Object} params
   * @returns {this}
   */
  setParams(params = {}) {
    Object.assign(this.params, params);
    return this;
  }

  /**
   * Get a shallow copy of parameters.
   * @returns {Object}
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Serialize minimal model metadata.
   * Subclasses may override to include learned parameters.
   */
  toJSON() {
    return {
      params: this.getParams(),
      fitted: !!this.fitted,
      state: this._state || {},
      warnings: this._warnings || []
    };
  }

  /**
   * Basic deserialization. Subclasses should override if they need
   * to restore learned arrays / matrices.
   * @param {Object} obj
   * @returns {Estimator}
   */
  static fromJSON(obj = {}) {
    const inst = new this(obj.params || {});
    if (obj.state) inst._state = obj.state;
    inst.fitted = !!obj.fitted;
    inst._warnings = obj.warnings || [];
    return inst;
  }

  /**
   * Save model to JSON string
   * @returns {string} JSON representation of the model
   */
  save() {
    const json = this.toJSON();
    const wrapped = {
      __tangentds__: true,
      version: "0.7.0",
      timestamp: new Date().toISOString(),
      estimatorType: this.constructor.name,
      data: json,
    };
    return JSON.stringify(wrapped, null, 2);
  }

  /**
   * Load model from JSON string
   * @param {string} jsonString - JSON representation
   * @returns {Estimator} Reconstructed estimator instance
   */
  static load(jsonString) {
    const parsed = JSON.parse(jsonString);
    if (!parsed.__tangentds__) {
      throw new Error("Invalid model format: missing __tangentds__ marker");
    }
    return this.fromJSON(parsed.data);
  }

  /**
   * Convenience helper: parse arguments passed to fit/predict/transform.
   *
   * Supports declarative table-style inputs:
   *  - fit({ X, y, data, omit_missing })
   *  - fit({ data, columns, ... })
   *
   * Returns an object { X, y, prepared, rows } where X/y are numeric arrays
   * if preparation was required, otherwise returns the original values.
   *
   * Note: this helper only prepares numeric matrices/vectors using core table utilities;
   * it does not perform encoding of categorical predictors.
   */
  _prepareArgsForFit(args = []) {
    // If called as fit({ X, y, data, ... })
    if (
      args.length === 1 &&
      args[0] &&
      typeof args[0] === "object" &&
      !Array.isArray(args[0])
    ) {
      const opts = args[0];
      // { X, y, data }
      if ((opts.X || opts.columns) && opts.data) {
        if (opts.X && opts.y) {
          const prepared = prepareXY({
            X: opts.X,
            y: opts.y,
            data: opts.data,
            omit_missing:
              opts.omit_missing !== undefined ? opts.omit_missing : true,
          });
          return {
            X: prepared.X,
            y: prepared.y,
            columnsX: prepared.columnsX,
            rows: prepared.rows,
            prepared: true,
          };
        }
        // columns-only -> prepareX
        const prepared = prepareX({
          columns: opts.columns || opts.X,
          data: opts.data,
          omit_missing:
            opts.omit_missing !== undefined ? opts.omit_missing : true,
        });
        return {
          X: prepared.X,
          columns: prepared.columns,
          rows: prepared.rows,
          prepared: true,
        };
      }
    }
    // Otherwise, return original args to be interpreted by subclass
    return { raw: args };
  }

  /**
   * Fit should be implemented by subclasses.
   * Return `this` for chaining.
   */
  fit(/* ...args */) {
    throw new Error("fit() not implemented for this estimator");
  }

  /**
   * Predict should be implemented by supervised estimators.
   */
  predict(/* X, options */) {
    throw new Error("predict() not implemented for this estimator");
  }

  /**
   * Transform should be implemented by transformers.
   */
  transform(/* X, options */) {
    throw new Error("transform() not implemented for this estimator");
  }
}

/**
 * Regressor base class
 */
export class Regressor extends Estimator {
  constructor(params = {}) {
    super(params);
  }

  /**
   * Predict - subclasses must override
   * Ensures model is fitted before prediction
   */
  predict(X, options = {}) {
    this._ensureFitted('predict');
    throw new Error("predict() not implemented for this regressor");
  }

  /**
   * Default R^2 scoring implementation:
   *   1 - SS_res / SS_tot
   *
   * Accepts either:
   *  - arrays: score(yTrue, yPred)
   *  - table-style: score({ X, y, data }) where predict will be called internally
   */
  score(yTrueOrOpts, yPred = null, opts = {}) {
    this._ensureFitted('score');

    // If first argument is an options object assume we need to predict
    if (
      arguments.length === 1 &&
      yTrueOrOpts &&
      typeof yTrueOrOpts === "object" &&
      !Array.isArray(yTrueOrOpts)
    ) {
      // Expect { X, y, data }
      const { X, y, data, omit_missing = true } = yTrueOrOpts;
      // prepare inputs using table helper
      if (X && y && data) {
        const prepared = prepareXY({ X, y, data, omit_missing });
        const preds = this.predict(prepared.X);
        const yTrue = prepared.y;
        return this._r2(yTrue, preds);
      }
      throw new Error(
        "score({ X, y, data }) expects X,y column names and data table",
      );
    }

    // Otherwise regular usage: score(yTrue, yPred)
    return this._r2(yTrueOrOpts, yPred);
  }

  _r2(yTrue, yPred) {
    if (
      !Array.isArray(yTrue) ||
      !Array.isArray(yPred) ||
      yTrue.length !== yPred.length
    ) {
      throw new Error("yTrue and yPred must be arrays of same length for R^2");
    }
    const yMean = mean(yTrue);
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < yTrue.length; i++) {
      ssTot += (yTrue[i] - yMean) ** 2;
      ssRes += (yTrue[i] - yPred[i]) ** 2;
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }
}

/**
 * Classifier base class
 */
export class Classifier extends Estimator {
  constructor(params = {}) {
    super(params);
    this.labelEncoder_ = null;
    this.classes_ = null;
  }

  /**
   * Predict - subclasses must override
   * Ensures model is fitted before prediction
   */
  predict(X, options = {}) {
    this._ensureFitted('predict');
    throw new Error("predict() not implemented for this classifier");
  }

  /**
   * Predict probabilities - subclasses should override
   * Ensures model is fitted before prediction
   */
  predictProba(X) {
    this._ensureFitted('predictProba');
    throw new Error("predictProba() not implemented for this classifier");
  }

  /**
   * Extract and store label encoder from prepared data
   * @param {Object} prepared - Result from prepareXY/prepareDataset
   * @returns {boolean} True if encoder was found and stored
   */
  _extractLabelEncoder(prepared) {
    if (prepared && prepared.encoders && prepared.encoders.y) {
      this.labelEncoder_ = prepared.encoders.y;
      if (this.labelEncoder_.classes_) {
        this.classes_ = this.labelEncoder_.classes_.slice();
      }
      return true;
    }
    this.labelEncoder_ = null;
    this.classes_ = null;
    return false;
  }

  /**
   * Get unique classes from labels (encoded or raw)
   * If labelEncoder exists, preparedY is assumed to be numeric indices [0, 1, 2, ...]
   * Otherwise, creates classes from unique values in preparedY
   *
   * @param {Array} preparedY - Label array (numeric if encoded, or raw labels)
   * @param {boolean} onlyPresentClasses - If true, only return classes present in preparedY
   * @returns {Object} { numericY, classes }
   */
  _getClasses(preparedY, onlyPresentClasses = true) {
    if (this.labelEncoder_) {
      // preparedY is already encoded as numbers by prepareXY
      const numericY = preparedY;

      if (onlyPresentClasses) {
        // Get only classes actually present in training data
        const uniqueIndices = Array.from(new Set(numericY)).sort(
          (a, b) => a - b,
        );
        const classes = uniqueIndices.map(
          (idx) => this.labelEncoder_.classes_[idx],
        );
        return { numericY, classes };
      } else {
        // Return all classes from encoder
        return { numericY, classes: this.labelEncoder_.classes_.slice() };
      }
    } else {
      // No encoder - handle string or numeric labels
      const uniqueLabels = Array.from(new Set(preparedY));

      if (typeof uniqueLabels[0] === "string") {
        // Create our own mapping for string labels
        const classes = uniqueLabels.sort();
        const classMap = {};
        classes.forEach((label, idx) => (classMap[label] = idx));
        const numericY = preparedY.map((label) => classMap[label]);
        return { numericY, classes };
      } else {
        // Already numeric
        const numericY = preparedY;
        const classes = uniqueLabels.sort((a, b) => a - b);
        return { numericY, classes };
      }
    }
  }

  /**
   * Decode numeric predictions to original labels
   * @param {Array} predictions - Numeric predictions or label strings
   * @returns {Array} Decoded labels (or original if no encoder)
   */
  _decodeLabels(predictions) {
    if (this.labelEncoder_) {
      return this.labelEncoder_.inverseTransform(predictions);
    }
    // If we have classes_ but no encoder, manually decode
    if (this.classes_ && typeof this.classes_[0] === "string") {
      return predictions.map((pred) => this.classes_[pred]);
    }
    return predictions;
  }

  /**
   * Default accuracy scoring:
   *  - score(yTrue, yPred)
   *  - or score({ X, y, data }) which predicts internally
   */
  score(yTrueOrOpts, yPred = null, opts = {}) {
    this._ensureFitted('score');

    if (
      arguments.length === 1 &&
      yTrueOrOpts &&
      typeof yTrueOrOpts === "object" &&
      !Array.isArray(yTrueOrOpts)
    ) {
      const { X, y, data, omit_missing = true } = yTrueOrOpts;
      if (X && y && data) {
        const prepared = prepareXY({ X, y, data, omit_missing });
        const preds = this.predict(prepared.X);
        return this._accuracy(prepared.y, preds);
      }
      throw new Error(
        "score({ X, y, data }) expects X,y column names and data table",
      );
    }
    return this._accuracy(yTrueOrOpts, yPred);
  }

  _accuracy(yTrue, yPred) {
    if (
      !Array.isArray(yTrue) ||
      !Array.isArray(yPred) ||
      yTrue.length !== yPred.length
    ) {
      throw new Error(
        "yTrue and yPred must be arrays of same length for accuracy",
      );
    }
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === yPred[i]) correct += 1;
    }
    return correct / yTrue.length;
  }
}

/**
 * Transformer base class
 */
export class Transformer extends Estimator {
  constructor(params = {}) {
    super(params);
  }

  /**
   * Transform - subclasses must override
   * Ensures model is fitted before transformation
   */
  transform(X, options = {}) {
    this._ensureFitted('transform');
    throw new Error("transform() not implemented for this transformer");
  }

  /**
   * Convenience: fit then transform
   * Returns transformed data.
   */
  fitTransform(...args) {
    this.fit(...args);
    return this.transform(...args);
  }
}
