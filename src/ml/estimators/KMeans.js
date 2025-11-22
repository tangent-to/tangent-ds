/**
 * KMeans estimator - class wrapper around functional kmeans utilities
 *
 * Usage:
 *   const km = new KMeans({ k: 3 });
 *   km.fit({ data: penguins, columns: ['flipper_length_mm','body_mass_g'] });
 *   const labels = km.predict({ data: someTable, columns: ['flipper_length_mm','body_mass_g'] });
 *
 * The class accepts both numeric-array inputs and the declarative table-style objects
 * supported by the core table utilities.
 */

import { Estimator } from '../../core/estimators/estimator.js';
import * as kmeansFn from '../kmeans.js';
import { prepareX } from '../../core/table.js';

export class KMeans extends Estimator {
  /**
   * @param {Object} params - { k, maxIter, tol, seed }
   */
  constructor({ k = 3, maxIter = 300, tol = 1e-4, seed = null } = {}) {
    super({ k, maxIter, tol, seed });
    this.k = k;
    this.maxIter = maxIter;
    this.tol = tol;
    this.seed = seed;

    // fitted model placeholder (result of kmeansFn.fit)
    this.model = null;
    this.fitted = false;
  }

  /**
   * Fit the KMeans model.
   *
   * Accepts:
   *  - numeric input: fit(Xarray, { k, maxIter, tol, seed })
   *  - declarative input: fit({ data: tableLike, columns: ['c1','c2'], k, ... })
   *
   * Returns this.
   */
  fit(X, opts = {}) {
    // If invoked with a single options-object that contains `data` or `columns`,
    // forward as declarative call to the underlying function.
    let fitResult;
    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.columns)
    ) {
      const callOpts = {
        // pass through provided options, falling back to instance defaults
        k: X.k !== undefined ? X.k : this.k,
        maxIter: X.maxIter !== undefined ? X.maxIter : this.maxIter,
        tol: X.tol !== undefined ? X.tol : this.tol,
        seed: X.seed !== undefined ? X.seed : this.seed,
        columns: X.columns !== undefined ? X.columns : X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      };
      // underlying kmeans.fit supports declarative object form
      fitResult = kmeansFn.fit(callOpts);
    } else {
      // Positional numeric-style call
      const callOpts = {
        k: opts.k !== undefined ? opts.k : this.k,
        maxIter: opts.maxIter !== undefined ? opts.maxIter : this.maxIter,
        tol: opts.tol !== undefined ? opts.tol : this.tol,
        seed: opts.seed !== undefined ? opts.seed : this.seed
      };
      fitResult = kmeansFn.fit(X, callOpts);
    }

    // store model details
    this.model = fitResult;
    this.labels = fitResult.labels;
    this.centroids = fitResult.centroids;
    this.inertia = fitResult.inertia;
    this.iterations = fitResult.iterations;
    this.converged = fitResult.converged;
    this.fitted = true;

    return this;
  }

  /**
   * Predict cluster labels for new data.
   *
   * Accepts:
   *  - numeric array: predict([[x1,x2], [x1,x2], ...])
   *  - declarative: predict({ data: tableLike, columns: ['c1','c2'], omit_missing: true })
   */
  predict(X) {
    this._ensureFitted('predict');

    // If declarative object with data/columns provided, prepare numeric matrix
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      return kmeansFn.predict(this.model, prepared.X);
    }

    // Otherwise assume numeric arrays and delegate to functional predict
    return kmeansFn.predict(this.model, X);
  }

  /**
   * Compute silhouette score for given X and labels (or use fitted labels if omitted).
   *
   * Accepts:
   *  - numeric X array, and optional labels array
   *  - declarative object { data, columns } will be prepared
   */
  silhouetteScore(X, labels = null) {
    let dataArr = X;
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      dataArr = prepared.X;
    }

    // If labels not provided, try to use fitted labels
    const useLabels = labels || this.labels;
    if (!useLabels) {
      throw new Error('silhouetteScore requires labels (or call after fit to use fitted labels)');
    }
    return kmeansFn.silhouetteScore(dataArr, useLabels);
  }

  /**
   * Convenience: return summary stats for fitted model
   */
  summary() {
    this._ensureFitted('summary');
    return {
      k: this.k,
      iterations: this.iterations,
      inertia: this.inertia,
      converged: this.converged,
      centroids: this.centroids
    };
  }

  /**
   * Serialization helper
   */
  toJSON() {
    return {
      __class__: 'KMeans',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new KMeans(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.labels = obj.model.labels;
      inst.centroids = obj.model.centroids;
      inst.inertia = obj.model.inertia;
      inst.iterations = obj.model.iterations;
      inst.converged = obj.model.converged;
      inst.fitted = true;
    }
    return inst;
  }
}

export default KMeans;
