/**
 * PCA - Principal Component Analysis estimator (class wrapper)
 *
 * Provides a scikit-learn style interface around the functional PCA utilities
 * in src/mva/pca.js. Supports both numeric array usage and declarative
 * table-style inputs that reference columns within a dataset.
 *
 * Example:
 *   const pca = new PCA({ scale: true });
 *   pca.fit({ data: table, columns: ['bill_length_mm', 'bill_depth_mm'] });
 *   const scores = pca.transform({ data: table, columns: ['bill_length_mm', 'bill_depth_mm'] });
 *
 * The fitted PCA model is stored on `this.model` and exposes helper methods
 * such as summary(), toJSON(), and fromJSON().
 */

import { Transformer } from '../../core/estimators/estimator.js';
import { prepareX } from '../../core/table.js';
import * as pcaFn from '../pca.js';
import { normalizeScaling, toScoreObjects, toLoadingObjects } from '../scaling.js';

const DEFAULT_PARAMS = {
  center: true,
  scale: false,
  columns: null,
  omit_missing: true,
  scaling: 2
};

export class PCA extends Transformer {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
  }

  /**
   * Fit PCA on the provided data.
   *
   * Accepts either:
   *  - fit(X[, opts]) where X is an array-of-arrays numeric matrix
   *  - fit({ data, columns, center, scale, omit_missing })
   *
   * Returns `this` for chaining.
   */
  fit(X, opts = {}) {
    let model;
    let effectiveParams;

    // Declarative call: fit({ data, columns, ... })
    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.columns || X.X)
    ) {
      effectiveParams = {
        ...DEFAULT_PARAMS,
        ...this.params,
        ...X
      };
      model = pcaFn.fit(effectiveParams);
    } else {
      // Numeric matrix call
      effectiveParams = {
        ...DEFAULT_PARAMS,
        ...this.params,
        ...opts
      };
      model = pcaFn.fit(X, effectiveParams);
    }

    this.model = model;
    this.fitted = true;
    this.params = {
      center: effectiveParams.center,
      scale: effectiveParams.scale,
      columns: effectiveParams.columns ?? null,
      omit_missing: effectiveParams.omit_missing,
      scaling: normalizeScaling(effectiveParams.scaling ?? this.model.scaling),
    };

    return this;
  }

  /**
   * Transform new data using fitted PCA model.
   *
   * Accepts numeric arrays or declarative table objects { data, columns }.
   */
  transform(X) {
    if (!this.fitted || !this.model) {
      throw new Error('PCA: estimator not fitted. Call fit() before transform().');
    }

    let matrix = X;
    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.columns || X.X)
    ) {
      const prepared = prepareX({
        columns: X.columns || X.X || this.params.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined
          ? X.omit_missing
          : this.params.omit_missing
      });
      matrix = prepared.X;
    }

    return pcaFn.transform(this.model, matrix);
  }

  /**
   * Helper to expose functional cumulative variance.
   */
  cumulativeVariance() {
    if (!this.fitted || !this.model) {
      throw new Error('PCA: estimator not fitted. Call fit() before cumulativeVariance().');
    }
    return pcaFn.cumulativeVariance(this.model);
  }

  /**
   * Provide lightweight summary of the fitted model.
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('PCA: estimator not fitted. Call fit() before summary().');
    }

    const { eigenvalues, varianceExplained, means, sds } = this.model;
    return {
      nComponents: eigenvalues?.length || 0,
      eigenvalues,
      varianceExplained,
      cumulativeVariance: this.cumulativeVariance(),
      centered: !!this.params.center,
      scaled: !!this.params.scale,
      scaling: this.params.scaling ?? 2,
      means,
      sds
    };
  }

  /**
   * Retrieve site or variable scores with optional scaling.
   *
   * @param {'sites'|'samples'|'variables'|'loadings'} type
   * @param {boolean} [scaled=true] - return scaled or raw coordinates
   */
  getScores(type = 'sites', scaled = true) {
    if (!this.fitted || !this.model) {
      throw new Error('PCA: estimator not fitted. Call fit() before getScores().');
    }

    const modelType = type.toLowerCase();
    if (modelType === 'sites' || modelType === 'samples') {
      if (scaled) return this.model.scores;
      return toScoreObjects(this.model.rawScores, 'pc');
    }

    if (modelType === 'variables' || modelType === 'loadings') {
      if (scaled) return this.model.loadings;
      return toLoadingObjects(this.model.rawLoadings, this.model.featureNames, 'pc');
    }

    throw new Error(`Unknown score type "${type}". Expected "sites" or "variables".`);
  }

  /**
   * Serialization helper for saving estimator state.
   */
  toJSON() {
    return {
      __class__: 'PCA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  /**
   * Restore PCA instance from JSON produced by toJSON().
   */
  static fromJSON(obj = {}) {
    const inst = new PCA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default PCA;
