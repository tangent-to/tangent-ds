/**
 * Enhanced Estimator Base Classes with Observable Safeguards
 *
 * This file demonstrates the recommended enhancements to the base Estimator class
 * to improve Observable compatibility, add safeguards, and optimize memory usage.
 *
 * TO IMPLEMENT: Copy the relevant methods into src/core/estimators/estimator.js
 */

import { prepareX, prepareXY } from "../core/table.js";
import { mean } from "../core/math.js";

/**
 * Enhanced Base Estimator with Observable safeguards
 */
export class Estimator {
  constructor(params = {}) {
    this.params = { ...params };
    this.fitted = false;
    this._state = {};
    this._warnings = []; // Track warnings during fitting
  }

  // ============================================================================
  // NEW: Fitted State Safeguards
  // ============================================================================

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

  // ============================================================================
  // NEW: Model State Inspection
  // ============================================================================

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
      // Rough estimation based on JSON serialization
      const json = this.toJSON();
      const jsonStr = JSON.stringify(json);
      // Characters * 2 bytes (UTF-16) / 1024 / 1024
      return (jsonStr.length * 2) / (1024 * 1024);
    } catch (e) {
      // If serialization fails, return -1 to indicate error
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

  // ============================================================================
  // NEW: Warning System
  // ============================================================================

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
    return this._warnings.slice(); // Return copy
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

  // ============================================================================
  // NEW: Performance Warnings
  // ============================================================================

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

    // Check if running in browser
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

  // ============================================================================
  // NEW: Observable Display Support
  // ============================================================================

  /**
   * Observable/Jupyter HTML representation
   * Shows model state and warnings in a user-friendly format
   * @returns {string} HTML representation
   */
  _repr_html_() {
    const className = this.constructor.name;

    // Not fitted state
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

    // Fitted state with warnings
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

    // Show warnings if any
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

    // Model parameters
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
   * Shows concise model info in console
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

  // ============================================================================
  // Existing Methods (keep as is)
  // ============================================================================

  setParams(params = {}) {
    Object.assign(this.params, params);
    return this;
  }

  getParams() {
    return { ...this.params };
  }

  toJSON() {
    return {
      params: this.getParams(),
      fitted: !!this.fitted,
      state: this._state || {},
      warnings: this._warnings || []
    };
  }

  static fromJSON(obj = {}) {
    const inst = new this(obj.params || {});
    if (obj.state) inst._state = obj.state;
    inst.fitted = !!obj.fitted;
    inst._warnings = obj.warnings || [];
    return inst;
  }

  save() {
    const json = this.toJSON();
    const wrapped = {
      __tangentds__: true,
      version: "0.3.0",
      timestamp: new Date().toISOString(),
      estimatorType: this.constructor.name,
      data: json,
    };
    return JSON.stringify(wrapped, null, 2);
  }

  static load(jsonString) {
    const parsed = JSON.parse(jsonString);
    if (!parsed.__tangentds__) {
      throw new Error("Invalid model format: missing __tangentds__ marker");
    }
    return this.fromJSON(parsed.data);
  }

  fit(/* ...args */) {
    throw new Error("fit() not implemented for this estimator");
  }

  predict(/* X, options */) {
    throw new Error("predict() not implemented for this estimator");
  }

  transform(/* X, options */) {
    throw new Error("transform() not implemented for this estimator");
  }
}

/**
 * Enhanced Regressor with safeguards
 */
export class Regressor extends Estimator {
  constructor(params = {}) {
    super(params);
  }

  /**
   * Enhanced predict with fitted check
   */
  predict(X, options = {}) {
    this._ensureFitted('predict');
    // Subclass implements actual prediction
    throw new Error("predict() not implemented for this regressor");
  }

  /**
   * Default R² scoring with fitted check
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
        const yTrue = prepared.y;
        return this._r2(yTrue, preds);
      }
      throw new Error(
        "score({ X, y, data }) expects X, y column names and data table",
      );
    }

    return this._r2(yTrueOrOpts, yPred);
  }

  _r2(yTrue, yPred) {
    if (
      !Array.isArray(yTrue) ||
      !Array.isArray(yPred) ||
      yTrue.length !== yPred.length
    ) {
      throw new Error("yTrue and yPred must be arrays of same length for R²");
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
 * Enhanced Classifier with safeguards
 */
export class Classifier extends Estimator {
  constructor(params = {}) {
    super(params);
    this.labelEncoder_ = null;
    this.classes_ = null;
  }

  /**
   * Enhanced predict with fitted check
   */
  predict(X, options = {}) {
    this._ensureFitted('predict');
    // Subclass implements actual prediction
    throw new Error("predict() not implemented for this classifier");
  }

  /**
   * Enhanced predictProba with fitted check
   */
  predictProba(X) {
    this._ensureFitted('predictProba');
    throw new Error("predictProba() not implemented for this classifier");
  }

  /**
   * Default accuracy scoring with fitted check
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
        "score({ X, y, data }) expects X, y column names and data table",
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

  // ... existing label encoder methods
}

/**
 * Enhanced Transformer with safeguards
 */
export class Transformer extends Estimator {
  constructor(params = {}) {
    super(params);
  }

  /**
   * Enhanced transform with fitted check
   */
  transform(X, options = {}) {
    this._ensureFitted('transform');
    // Subclass implements actual transformation
    throw new Error("transform() not implemented for this transformer");
  }

  /**
   * Convenience: fit then transform
   */
  fitTransform(...args) {
    this.fit(...args);
    return this.transform(...args);
  }
}

// ============================================================================
// Example Usage in Subclass
// ============================================================================

/**
 * Example: How to use enhanced base class in a model
 */
export class ExampleGLM extends Estimator {
  constructor(params = {}) {
    super(params);
    this._model = null;
  }

  fit(X, y) {
    // Add dataset size warnings
    this._checkDatasetSize(X, y, {
      warnLargeDataset: this.params.warnOnLargeDataset !== false
    });

    // ... fitting logic

    // Simulate convergence warning
    const converged = true; // from fitting algorithm
    if (!converged) {
      const message =
        `⚠️ GLM did not converge after iterations.\n` +
        `Consider increasing maxIter or checking data quality.`;

      console.warn(message);
      this._addWarning('convergence', message, {
        iterations: this.params.maxIter
      });
    }

    this.fitted = true;
    return this;
  }

  predict(X) {
    // Use centralized fitted check
    this._ensureFitted('predict');

    // ... prediction logic
    return [];
  }

  summary() {
    // Use centralized fitted check
    this._ensureFitted('summary');

    // ... summary logic
    return "Summary output";
  }
}
