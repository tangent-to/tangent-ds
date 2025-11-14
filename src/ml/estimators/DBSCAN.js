/**
 * DBSCAN estimator - class wrapper around functional dbscan utilities
 *
 * Usage:
 *   const dbscan = new DBSCAN({ eps: 0.3, minSamples: 5 });
 *   dbscan.fit({ data: myData, columns: ['x', 'y'] });
 *   const labels = dbscan.labels;  // -1 for noise, 0+ for cluster IDs
 *
 * The class accepts both numeric-array inputs and the declarative table-style objects
 * supported by the core table utilities.
 */

import { Estimator } from '../../core/estimators/estimator.js';
import * as dbscanFn from '../dbscan.js';
import { prepareX } from '../../core/table.js';

export class DBSCAN extends Estimator {
  /**
   * @param {Object} params - { eps, minSamples }
   * @param {number} params.eps - Maximum distance between two points for neighborhood (default: 0.5)
   * @param {number} params.minSamples - Minimum number of points to form a dense region (default: 5)
   */
  constructor({ eps = 0.5, minSamples = 5 } = {}) {
    super({ eps, minSamples });
    this.eps = eps;
    this.minSamples = minSamples;

    // fitted model placeholder (result of dbscanFn.fit)
    this.model = null;
    this.fitted = false;
    this.X_train = null;  // Store training data for prediction
  }

  /**
   * Fit the DBSCAN model.
   *
   * Accepts:
   *  - numeric input: fit(Xarray, { eps, minSamples })
   *  - declarative input: fit({ data: tableLike, columns: ['c1','c2'], eps, ... })
   *
   * Returns this.
   */
  fit(X, opts = {}) {
    // If invoked with a single options-object that contains `data` or `columns`,
    // forward as declarative call to the underlying function.
    let fitResult;
    let trainData;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.columns)
    ) {
      const callOpts = {
        eps: X.eps !== undefined ? X.eps : this.eps,
        minSamples: X.minSamples !== undefined ? X.minSamples : this.minSamples,
        columns: X.columns !== undefined ? X.columns : X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      };

      // Prepare data to store for prediction
      const prepared = prepareX({
        columns: callOpts.columns,
        data: callOpts.data,
        omit_missing: callOpts.omit_missing
      });
      trainData = prepared.X;

      // underlying dbscan.fit supports declarative object form
      fitResult = dbscanFn.fit(callOpts);
    } else {
      // Positional numeric-style call
      const callOpts = {
        eps: opts.eps !== undefined ? opts.eps : this.eps,
        minSamples: opts.minSamples !== undefined ? opts.minSamples : this.minSamples
      };
      fitResult = dbscanFn.fit(X, callOpts);
      trainData = Array.isArray(X[0]) ? X : X.map((x) => [x]);
    }

    // store model details
    this.model = fitResult;
    this.labels = fitResult.labels;
    this.nClusters = fitResult.nClusters;
    this.nNoise = fitResult.nNoise;
    this.coreSampleIndices = fitResult.coreSampleIndices;
    this.X_train = trainData;
    this.fitted = true;

    return this;
  }

  /**
   * Predict cluster labels for new data.
   *
   * Note: DBSCAN doesn't naturally support prediction on new points.
   * This assigns new points to the cluster of their nearest core point
   * if within eps distance, otherwise marks as noise (-1).
   *
   * Accepts:
   *  - numeric array: predict([[x1,x2], [x1,x2], ...])
   *  - declarative: predict({ data: tableLike, columns: ['c1','c2'], omit_missing: true })
   */
  predict(X) {
    if (!this.fitted || !this.model) {
      throw new Error('DBSCAN: estimator is not fitted. Call fit() first.');
    }

    // If declarative object with data/columns provided, prepare numeric matrix
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      return dbscanFn.predict(this.model, prepared.X, this.X_train, this.eps);
    }

    // Otherwise assume numeric arrays and delegate to functional predict
    return dbscanFn.predict(this.model, X, this.X_train, this.eps);
  }

  /**
   * Get core sample mask (boolean array indicating which samples are core points)
   */
  get coreSampleMask() {
    if (!this.fitted) {
      throw new Error('DBSCAN: estimator is not fitted.');
    }
    const mask = new Array(this.labels.length).fill(false);
    for (const idx of this.coreSampleIndices) {
      mask[idx] = true;
    }
    return mask;
  }

  /**
   * Get components (core samples) - returns array of core sample data points
   */
  get components() {
    if (!this.fitted) {
      throw new Error('DBSCAN: estimator is not fitted.');
    }
    return this.coreSampleIndices.map(idx => this.X_train[idx]);
  }

  /**
   * Convenience: return summary stats for fitted model
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('DBSCAN: estimator is not fitted.');
    }

    const nSamples = this.labels.length;
    const nCore = this.coreSampleIndices.length;

    return {
      eps: this.eps,
      minSamples: this.minSamples,
      nClusters: this.nClusters,
      nNoise: this.nNoise,
      nSamples,
      nCore,
      noiseRatio: this.nNoise / nSamples,
      coreRatio: nCore / nSamples
    };
  }

  /**
   * Serialization helper
   */
  toJSON() {
    return {
      __class__: 'DBSCAN',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
      X_train: this.X_train
    };
  }

  static fromJSON(obj = {}) {
    const inst = new DBSCAN(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.labels = obj.model.labels;
      inst.nClusters = obj.model.nClusters;
      inst.nNoise = obj.model.nNoise;
      inst.coreSampleIndices = obj.model.coreSampleIndices;
      inst.X_train = obj.X_train;
      inst.fitted = true;
    }
    return inst;
  }
}

export default DBSCAN;
