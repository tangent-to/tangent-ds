/**
 * LDA - Linear Discriminant Analysis estimator (class wrapper)
 *
 * Provides a scikit-learn style estimator around the functional LDA implementation
 * in src/mva/lda.js. Accepts both numeric arrays and declarative table-style inputs:
 *
 *  - Numeric style:
 *      const lda = new LDA();
 *      lda.fit(X, y);
 *      const preds = lda.predict(Xnew);
 *
 *  - Declarative style (uses core table helpers via underlying functions):
 *      lda.fit({ X: ['col1','col2'], y: 'label', data: tableLike, omit_missing: true });
 *      lda.predict({ X: ['col1','col2'], data: otherTable });
 *
 * The wrapper stores the fitted internal model returned by the functional API and
 * exposes .predict(), .transform(), .summary(), .toJSON()/fromJSON().
 */

import { Classifier } from '../../core/estimators/estimator.js';
import * as ldaFn from '../lda.js';
import { normalizeScaling, toLoadingObjects, toScoreObjects } from '../scaling.js';

const DEFAULT_PARAMS = {
  scale: false,
  scaling: 2,
};

export class LDA extends Classifier {
  /**
   * @param {Object} params - optional hyperparameters (none required for basic LDA)
   */
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
    this.fitted = false;
  }

  /**
   * Fit the LDA model.
   *
   * Supports:
   *  - fit(Xarray, yarray)
   *  - fit({ X: 'col'|'[cols]', y: 'label', data: tableLike, omit_missing, encoders })
   *
   * Returns: this
   */
  fit(X, y = null, opts = {}) {
    let result;

    // If first argument is a declarative options object (contains data/X/y), forward directly
    if (X && typeof X === 'object' && !Array.isArray(X) && ('data' in X || 'X' in X || 'x' in X)) {
      const baseOpts = {
        scale: this.params.scale,
        scaling: this.params.scaling,
      };
      const callOpts = { ...baseOpts, ...X };
      // If intercept-like or other params existed, they'd be merged here.
      // Underlying ldaFn.fit supports receiving a single object { X, y, data, encoders, ... }
      result = ldaFn.fit(callOpts);
    } else {
      // Positional numeric call: fit(Xarray, yarray, opts)
      // pass opts through if provided
      const mergedOpts = {
        scale: this.params.scale,
        scaling: this.params.scaling,
        ...opts,
      };
      result = ldaFn.fit(X, y, mergedOpts);
    }

    // Save model and metadata
    this.model = result;
    this.fitted = true;
    this.params.scaling = normalizeScaling(result.scaling ?? this.params.scaling);

    // Extract label encoder if present
    if (result.labelEncoder) {
      this.labelEncoder_ = result.labelEncoder;
      if (this.labelEncoder_.classes_) {
        this.classes_ = this.labelEncoder_.classes_.slice();
      }
    }

    return this;
  }

  /**
   * Transform input X to discriminant scores (delegates to functional transform).
   *
   * Accepts:
   *  - numeric array X
   *  - declarative object { X: cols, data: tableLike }
   */
  transform(X) {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    return ldaFn.transform(this.model, X);
  }

  /**
   * Predict class labels for X.
   *
   * Accepts:
   *  - numeric array X
   *  - declarative object { X: cols, data: tableLike }
   *
   * Returns decoded labels if label encoder is present, otherwise numeric predictions
   */
  predict(X) {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    const predictions = ldaFn.predict(this.model, X);

    // Decode labels using centralized method if encoder exists
    return this._decodeLabels(predictions);
  }

  /**
   * Return a small summary of the fitted model.
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    const { classes, discriminantAxes, eigenvalues } = this.model;
    return {
      classes,
      nComponents: discriminantAxes ? discriminantAxes.length : 0,
      eigenvalues,
      scaling: this.params.scaling,
    };
  }

  /**
   * Retrieve site or variable scores (scaled or raw).
   * @param {'sites'|'samples'|'variables'|'loadings'} type
   * @param {boolean} [scaled=true]
   */
  getScores(type = 'sites', scaled = true) {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    const modelType = type.toLowerCase();
    if (modelType === 'sites' || modelType === 'samples') {
      if (scaled) return this.model.scores;
      return toScoreObjects(
        this.model.rawScores,
        'ld',
        (idx) => ({ class: this.model.sampleClasses?.[idx] })
      );
    }

    if (modelType === 'variables' || modelType === 'loadings') {
      if (scaled) return this.model.loadings;
      return toLoadingObjects(this.model.rawLoadings, this.model.featureNames, 'ld');
    }

    throw new Error(`Unknown score type "${type}". Expected "sites" or "variables".`);
  }

  /**
   * JSON serialization helper.
   */
  toJSON() {
    return {
      __class__: 'LDA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
    };
  }

  /**
   * Restore an instance from JSON produced by toJSON().
   */
  static fromJSON(obj = {}) {
    const inst = new LDA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = true;
    }
    return inst;
  }
}

export default LDA;
