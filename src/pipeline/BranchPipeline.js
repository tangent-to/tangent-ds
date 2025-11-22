/**
 * BranchPipeline - Run multiple pipelines in parallel and combine results
 *
 * Usage:
 *   const pipeline = new BranchPipeline({
 *     branches: {
 *       model1: new KMeans({ k: 3 }),
 *       model2: new DBSCAN({ eps: 0.5 })
 *     },
 *     combiner: 'vote'
 *   });
 *   pipeline.fit(X);
 *   const labels = pipeline.predict(X);
 */

import { Estimator } from '../core/estimators/estimator.js';

export class BranchPipeline extends Estimator {
  /**
   * @param {Object} options
   * @param {Object<string, Object>} options.branches - Named branches (estimators or pipelines)
   * @param {string|Function} options.combiner - How to combine results: 'vote', 'average', 'max', or custom function
   * @param {Array<number>} options.weights - Optional weights for each branch (for weighted voting)
   */
  constructor({ branches = {}, combiner = 'vote', weights = null } = {}) {
    super({ combiner, weights });
    this.branches = branches;
    this.combiner = combiner;
    this.weights = weights;
    this.branchNames = Object.keys(branches);

    // Validate weights
    if (weights && weights.length !== this.branchNames.length) {
      throw new Error('Weights length must match number of branches');
    }
  }

  /**
   * Fit all branches
   * @param {Array} X - Training data
   * @param {Array} y - Target labels (optional)
   * @returns {this}
   */
  fit(X, y = null) {
    // Fit each branch
    for (const [name, estimator] of Object.entries(this.branches)) {
      if (typeof estimator.fit !== 'function') {
        throw new Error(`Branch '${name}' does not have a fit() method`);
      }
      estimator.fit(X, y);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict using all branches and combine results
   * @param {Array} X - Data to predict
   * @returns {Array} Combined predictions
   */
  predict(X) {
    this._ensureFitted('predict');

    // Get predictions from each branch
    const branchPredictions = [];
    for (const [name, estimator] of Object.entries(this.branches)) {
      if (typeof estimator.predict !== 'function') {
        throw new Error(`Branch '${name}' does not have a predict() method`);
      }

      const pred = estimator.predict(X);
      branchPredictions.push({
        name,
        predictions: Array.isArray(pred) ? pred : pred.labels || [pred]
      });
    }

    // Combine predictions
    return this._combine(branchPredictions, X.length || X[0]?.length);
  }

  /**
   * Get predictions from all branches without combining
   * @param {Array} X - Data to predict
   * @returns {Object<string, Array>} Predictions keyed by branch name
   */
  predictAll(X) {
    this._ensureFitted('predictAll');

    const allPredictions = {};
    for (const [name, estimator] of Object.entries(this.branches)) {
      const pred = estimator.predict(X);
      allPredictions[name] = Array.isArray(pred) ? pred : pred.labels || [pred];
    }

    return allPredictions;
  }

  /**
   * Combine predictions from multiple branches
   * @private
   * @param {Array<Object>} branchPredictions - Predictions from each branch
   * @param {number} n - Number of samples
   * @returns {Array} Combined predictions
   */
  _combine(branchPredictions, n) {
    const combined = new Array(n);

    switch (this.combiner) {
      case 'vote':
        return this._voteStrategy(branchPredictions, n);

      case 'weighted_vote':
        return this._weightedVoteStrategy(branchPredictions, n);

      case 'average':
        return this._averageStrategy(branchPredictions, n);

      case 'max':
        return this._maxStrategy(branchPredictions, n);

      case 'min':
        return this._minStrategy(branchPredictions, n);

      default:
        if (typeof this.combiner === 'function') {
          return this.combiner(branchPredictions);
        }
        throw new Error(`Unknown combiner strategy: ${this.combiner}`);
    }
  }

  /**
   * Majority voting strategy
   * @private
   */
  _voteStrategy(branchPredictions, n) {
    const combined = new Array(n);

    for (let i = 0; i < n; i++) {
      const votes = {};

      branchPredictions.forEach(({ predictions }) => {
        const label = predictions[i];
        votes[label] = (votes[label] || 0) + 1;
      });

      // Get most voted label
      combined[i] = Number(
        Object.entries(votes)
          .sort((a, b) => b[1] - a[1])[0][0]
      );
    }

    return combined;
  }

  /**
   * Weighted voting strategy
   * @private
   */
  _weightedVoteStrategy(branchPredictions, n) {
    if (!this.weights) {
      throw new Error('Weights must be provided for weighted_vote combiner');
    }

    const combined = new Array(n);

    for (let i = 0; i < n; i++) {
      const votes = {};

      branchPredictions.forEach(({ predictions }, idx) => {
        const label = predictions[i];
        const weight = this.weights[idx] || 1;
        votes[label] = (votes[label] || 0) + weight;
      });

      // Get most voted label (weighted)
      combined[i] = Number(
        Object.entries(votes)
          .sort((a, b) => b[1] - a[1])[0][0]
      );
    }

    return combined;
  }

  /**
   * Average predictions (for regression)
   * @private
   */
  _averageStrategy(branchPredictions, n) {
    const combined = new Array(n);

    for (let i = 0; i < n; i++) {
      let sum = 0;
      let count = 0;

      branchPredictions.forEach(({ predictions }) => {
        const value = predictions[i];
        if (typeof value === 'number' && !isNaN(value)) {
          sum += value;
          count++;
        }
      });

      combined[i] = count > 0 ? sum / count : 0;
    }

    return combined;
  }

  /**
   * Maximum value strategy
   * @private
   */
  _maxStrategy(branchPredictions, n) {
    const combined = new Array(n);

    for (let i = 0; i < n; i++) {
      const values = branchPredictions
        .map(({ predictions }) => predictions[i])
        .filter(v => typeof v === 'number' && !isNaN(v));

      combined[i] = values.length > 0 ? Math.max(...values) : 0;
    }

    return combined;
  }

  /**
   * Minimum value strategy
   * @private
   */
  _minStrategy(branchPredictions, n) {
    const combined = new Array(n);

    for (let i = 0; i < n; i++) {
      const values = branchPredictions
        .map(({ predictions }) => predictions[i])
        .filter(v => typeof v === 'number' && !isNaN(v));

      combined[i] = values.length > 0 ? Math.min(...values) : 0;
    }

    return combined;
  }

  /**
   * Get agreement score (how often branches agree)
   * @param {Array} X - Data
   * @returns {number} Agreement score between 0 and 1
   */
  agreementScore(X) {
    const allPredictions = this.predictAll(X);
    const branchPreds = Object.values(allPredictions);
    const n = branchPreds[0].length;

    let agreements = 0;
    for (let i = 0; i < n; i++) {
      const values = branchPreds.map(pred => pred[i]);
      const allSame = values.every(v => v === values[0]);
      if (allSame) agreements++;
    }

    return agreements / n;
  }

  /**
   * Get per-sample confidence (based on branch agreement)
   * @param {Array} X - Data
   * @returns {Array<number>} Confidence scores between 0 and 1
   */
  confidence(X) {
    const allPredictions = this.predictAll(X);
    const branchPreds = Object.values(allPredictions);
    const n = branchPreds[0].length;
    const confidence = new Array(n);

    for (let i = 0; i < n; i++) {
      const votes = {};
      branchPreds.forEach(pred => {
        const label = pred[i];
        votes[label] = (votes[label] || 0) + 1;
      });

      // Confidence = max votes / total branches
      const maxVotes = Math.max(...Object.values(votes));
      confidence[i] = maxVotes / branchPreds.length;
    }

    return confidence;
  }

  /**
   * Summary statistics
   * @returns {Object}
   */
  summary() {
    return {
      nBranches: this.branchNames.length,
      branches: this.branchNames,
      combiner: typeof this.combiner === 'function' ? 'custom' : this.combiner,
      weights: this.weights,
      fitted: this.fitted
    };
  }

  /**
   * Serialization
   */
  toJSON() {
    // Note: This is a simplified version
    // Full implementation would need to serialize each branch
    return {
      __class__: 'BranchPipeline',
      combiner: typeof this.combiner === 'function' ? 'custom' : this.combiner,
      weights: this.weights,
      fitted: this.fitted,
      branches: Object.keys(this.branches)
    };
  }
}

export default BranchPipeline;
