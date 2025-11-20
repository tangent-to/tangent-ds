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
      version: '0.7.0',
      timestamp: new Date().toISOString(),
      estimatorType: this.constructor.name,
      data: json
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
      throw new Error('Invalid model format: missing __tangentds__ marker');
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
      args.length === 1 && args[0] && typeof args[0] === "object" &&
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
            omit_missing: opts.omit_missing !== undefined
              ? opts.omit_missing
              : true,
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
          omit_missing: opts.omit_missing !== undefined
            ? opts.omit_missing
            : true,
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
   * Default R^2 scoring implementation:
   *   1 - SS_res / SS_tot
   *
   * Accepts either:
   *  - arrays: score(yTrue, yPred)
   *  - table-style: score({ X, y, data }) where predict will be called internally
   */
  score(yTrueOrOpts, yPred = null, opts = {}) {
    // If first argument is an options object assume we need to predict
    if (
      arguments.length === 1 && yTrueOrOpts &&
      typeof yTrueOrOpts === "object" && !Array.isArray(yTrueOrOpts)
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
      !Array.isArray(yTrue) || !Array.isArray(yPred) ||
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
    return ssTot === 0 ? 0 : 1 - (ssRes / ssTot);
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
        const uniqueIndices = Array.from(new Set(numericY)).sort((a, b) => a - b);
        const classes = uniqueIndices.map(idx => this.labelEncoder_.classes_[idx]);
        return { numericY, classes };
      } else {
        // Return all classes from encoder
        return { numericY, classes: this.labelEncoder_.classes_.slice() };
      }
    } else {
      // No encoder - handle string or numeric labels
      const uniqueLabels = Array.from(new Set(preparedY));

      if (typeof uniqueLabels[0] === 'string') {
        // Create our own mapping for string labels
        const classes = uniqueLabels.sort();
        const classMap = {};
        classes.forEach((label, idx) => classMap[label] = idx);
        const numericY = preparedY.map(label => classMap[label]);
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
    return predictions;
  }

  /**
   * Default accuracy scoring:
   *  - score(yTrue, yPred)
   *  - or score({ X, y, data }) which predicts internally
   */
  score(yTrueOrOpts, yPred = null, opts = {}) {
    if (
      arguments.length === 1 && yTrueOrOpts &&
      typeof yTrueOrOpts === "object" && !Array.isArray(yTrueOrOpts)
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
      !Array.isArray(yTrue) || !Array.isArray(yPred) ||
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

  /**
   * Optionally implement predictProba in subclasses.
   */
  predictProba(/* X */) {
    throw new Error("predictProba() not implemented for this classifier");
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
   * Convenience: fit then transform
   * Returns transformed data.
   */
  fitTransform(...args) {
    this.fit(...args);
    return this.transform(...args);
  }
}
