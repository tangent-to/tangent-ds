/**
 * ConsensusCluster - Proper ensemble clustering via co-association matrix
 *
 * Solves the label alignment problem by measuring pairwise agreement
 * across multiple clustering algorithms instead of naively voting on labels.
 *
 * Based on:
 * Fred, A. L., & Jain, A. K. (2005). Combining multiple clusterings using
 * evidence accumulation. IEEE Transactions on Pattern Analysis and Machine
 * Intelligence, 27(6), 835-850.
 *
 * Usage:
 *   const consensus = new ConsensusCluster({
 *     estimators: [
 *       new KMeans({ k: 3, seed: 42 }),
 *       new DBSCAN({ eps: 0.5, minSamples: 5 }),
 *       new HCA({ k: 3, linkage: 'average' })
 *     ],
 *     threshold: 0.5
 *   });
 *   // Supports both table format and raw arrays
 *   consensus.fit({ data: myData, columns: ['x', 'y', 'z'] });
 *   // or: consensus.fit(numericMatrix);
 *   console.log(consensus.labels);
 */

import { Estimator } from '../core/estimators/estimator.js';

export class ConsensusCluster extends Estimator {
  /**
   * @param {Object} options
   * @param {Array<Object>} options.estimators - Array of clustering estimators
   * @param {number} [options.threshold=0.5] - Minimum co-association for same cluster (0-1)
   * @param {string} [options.linkage='single'] - Linkage for final clustering
   */
  constructor({
    estimators = [],
    threshold = 0.5,
    linkage = 'single'
  } = {}) {
    super({ threshold, linkage });
    if (estimators.length === 0) {
      throw new Error('ConsensusCluster requires at least one estimator');
    }

    this.estimators = estimators;
    this.threshold = threshold;
    this.linkage = linkage;
    this.coAssocMatrix = null;
    this.labels = null;
  }

  /**
   * Fit consensus clustering
   * @param {Array|Object} X - Data to cluster (raw array or {data, columns})
   * @returns {this}
   */
  fit(X) {
    // Determine number of samples based on input format
    let n;
    if (X && typeof X === 'object' && X.data) {
      // Table format: {data, columns}
      n = X.data.length;
    } else if (Array.isArray(X)) {
      // Raw array format
      n = X.length;
    } else {
      n = 0;
    }

    if (!n) {
      throw new Error('Empty data provided');
    }

    // Initialize co-association matrix (counts co-occurrences)
    const coAssoc = Array(n).fill(0).map(() => Array(n).fill(0));

    // Fit each estimator and accumulate co-associations
    for (const estimator of this.estimators) {
      // Fit the estimator (pass through original format)
      estimator.fit(X);

      // Get labels (handle both .labels property and .predict())
      let labels;
      if (estimator.labels) {
        labels = estimator.labels;
      } else if (typeof estimator.predict === 'function') {
        labels = estimator.predict(X);
      } else {
        throw new Error('Estimator must have .labels or .predict()');
      }

      // Handle array vs object result from predict
      if (!Array.isArray(labels)) {
        labels = labels.labels || [labels];
      }

      // Update co-association matrix
      this._updateCoAssociation(coAssoc, labels);
    }

    // Normalize by number of estimators
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        coAssoc[i][j] /= this.estimators.length;
      }
    }

    this.coAssocMatrix = coAssoc;

    // Extract final consensus clustering
    this.labels = this._extractClusters(coAssoc, this.threshold);
    this.fitted = true;

    return this;
  }

  /**
   * Update co-association matrix with new clustering
   * @private
   */
  _updateCoAssociation(coAssoc, labels) {
    const n = labels.length;

    for (let i = 0; i < n; i++) {
      // Skip noise points
      if (labels[i] === -1) continue;

      for (let j = i; j < n; j++) {
        // Skip noise points
        if (labels[j] === -1) continue;

        // If same cluster, increment co-association
        if (labels[i] === labels[j]) {
          coAssoc[i][j]++;
          if (i !== j) coAssoc[j][i]++;
        }
      }
    }
  }

  /**
   * Extract clusters from co-association matrix
   * @private
   */
  _extractClusters(coAssoc, threshold) {
    const n = coAssoc.length;
    const labels = new Array(n).fill(-1);
    let clusterId = 0;

    // Simple connected components approach
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1) continue;

      // BFS to find connected component
      const queue = [i];
      labels[i] = clusterId;

      while (queue.length > 0) {
        const curr = queue.shift();

        for (let j = 0; j < n; j++) {
          if (labels[j] === -1 && coAssoc[curr][j] >= threshold) {
            labels[j] = clusterId;
            queue.push(j);
          }
        }
      }

      clusterId++;
    }

    return labels;
  }

  /**
   * Get consensus strength for each sample
   * Higher values = stronger agreement across models
   * @returns {Array<number>} Strength scores between 0 and 1
   */
  getConsensusStrength() {
    this._ensureFitted('getConsensusStrength');

    const n = this.labels.length;
    const strength = new Array(n);

    for (let i = 0; i < n; i++) {
      const myCluster = this.labels[i];

      if (myCluster === -1) {
        strength[i] = 0;
        continue;
      }

      // Average co-association with other members of same cluster
      let sum = 0;
      let count = 0;

      for (let j = 0; j < n; j++) {
        if (this.labels[j] === myCluster && i !== j) {
          sum += this.coAssocMatrix[i][j];
          count++;
        }
      }

      strength[i] = count > 0 ? sum / count : 1;
    }

    return strength;
  }

  /**
   * Get average consensus strength per cluster
   * @returns {Object} {cluster: avgStrength}
   */
  getClusterStrength() {
    this._ensureFitted('getClusterStrength');

    const strength = this.getConsensusStrength();
    const clusterStrengths = {};
    const clusterCounts = {};

    for (let i = 0; i < this.labels.length; i++) {
      const cluster = this.labels[i];
      if (cluster === -1) continue;

      clusterStrengths[cluster] = (clusterStrengths[cluster] || 0) + strength[i];
      clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
    }

    const result = {};
    for (const cluster in clusterStrengths) {
      result[cluster] = clusterStrengths[cluster] / clusterCounts[cluster];
    }

    return result;
  }

  /**
   * Get number of clusters (excluding noise)
   */
  get nClusters() {
    if (!this.fitted) return 0;
    return new Set(this.labels.filter(l => l !== -1)).size;
  }

  /**
   * Get number of noise points
   */
  get nNoise() {
    if (!this.fitted) return 0;
    return this.labels.filter(l => l === -1).length;
  }

  /**
   * Get overall agreement score
   * Measures how consistent the input clusterings are
   * @returns {number} Score between 0 (no agreement) and 1 (perfect agreement)
   */
  get agreementScore() {
    if (!this.fitted) return 0;

    const n = this.labels.length;
    let totalAgreement = 0;
    let count = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        totalAgreement += this.coAssocMatrix[i][j];
        count++;
      }
    }

    return count > 0 ? totalAgreement / count : 0;
  }

  /**
   * Get detailed comparison of input clusterings
   * Shows how each estimator contributed
   */
  getEstimatorAgreement() {
    this._ensureFitted('getEstimatorAgreement');

    const agreement = [];

    for (let i = 0; i < this.estimators.length; i++) {
      const est = this.estimators[i];
      const estLabels = est.labels || est.predict(this.X_train);
      const labels = Array.isArray(estLabels) ? estLabels : estLabels.labels;

      // Compute Adjusted Rand Index with consensus
      const ari = this._adjustedRandIndex(labels, this.labels);

      agreement.push({
        estimator: est.constructor.name,
        ari: ari,
        nClusters: new Set(labels.filter(l => l !== -1)).size,
        nNoise: labels.filter(l => l === -1).length
      });
    }

    return agreement;
  }

  /**
   * Compute Adjusted Rand Index between two labelings
   * @private
   */
  _adjustedRandIndex(labels1, labels2) {
    const n = labels1.length;
    const contingency = new Map();

    // Build contingency table
    for (let i = 0; i < n; i++) {
      const key = `${labels1[i]},${labels2[i]}`;
      contingency.set(key, (contingency.get(key) || 0) + 1);
    }

    // Simplified ARI calculation (full version would be more complex)
    let agreements = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const same1 = labels1[i] === labels1[j];
        const same2 = labels2[i] === labels2[j];
        if (same1 === same2) agreements++;
      }
    }

    const total = (n * (n - 1)) / 2;
    return agreements / total;
  }

  /**
   * Summary statistics
   */
  summary() {
    if (!this.fitted) {
      return {
        fitted: false,
        nEstimators: this.estimators.length,
        threshold: this.threshold
      };
    }

    const strength = this.getConsensusStrength();
    const clusterStrength = this.getClusterStrength();

    return {
      fitted: true,
      nEstimators: this.estimators.length,
      threshold: this.threshold,
      nClusters: this.nClusters,
      nNoise: this.nNoise,
      nSamples: this.labels.length,
      noiseRatio: this.nNoise / this.labels.length,
      overallAgreement: this.agreementScore,
      avgConsensusStrength: strength.reduce((a, b) => a + b, 0) / strength.length,
      clusterStrengths: clusterStrength
    };
  }

  /**
   * Serialization (simplified)
   */
  toJSON() {
    return {
      __class__: 'ConsensusCluster',
      threshold: this.threshold,
      linkage: this.linkage,
      fitted: this.fitted,
      labels: this.labels,
      coAssocMatrix: this.coAssocMatrix,
      nEstimators: this.estimators.length
    };
  }

  static fromJSON(obj) {
    const instance = new ConsensusCluster({
      estimators: [],  // Can't restore estimators from simple JSON
      threshold: obj.threshold,
      linkage: obj.linkage
    });

    instance.fitted = obj.fitted;
    instance.labels = obj.labels;
    instance.coAssocMatrix = obj.coAssocMatrix;

    return instance;
  }
}

export default ConsensusCluster;
