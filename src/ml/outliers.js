/**
 * Outlier Detection
 *
 * Provides sklearn-compatible outlier detectors:
 * - IsolationForest: Tree-based anomaly detection
 * - LocalOutlierFactor: Density-based outlier detection
 * - MahalanobisDistance: Statistical distance-based outlier detection
 */

import { prepareX } from '../core/table.js';
import { random, randomInt, sample as randomSample } from './utils.js';
import { Matrix, pseudoInverse } from 'ml-matrix';

// ============= IsolationForest =============

/**
 * Isolation Tree Node
 */
class IsolationTreeNode {
  constructor(size) {
    this.size = size;
    this.splitAttr = null;
    this.splitValue = null;
    this.left = null;
    this.right = null;
  }

  isExternal() {
    return this.left === null && this.right === null;
  }
}

/**
 * Build an isolation tree
 */
function buildIsolationTree(X, heightLimit, currentHeight = 0) {
  const n = X.length;
  const node = new IsolationTreeNode(n);

  // External node (leaf)
  if (currentHeight >= heightLimit || n <= 1) {
    return node;
  }

  // Randomly select attribute and split value
  const nFeatures = X[0].length;
  const splitAttr = randomInt(0, nFeatures);

  // Get min and max of selected attribute
  let minVal = X[0][splitAttr];
  let maxVal = X[0][splitAttr];
  for (let i = 1; i < n; i++) {
    if (X[i][splitAttr] < minVal) minVal = X[i][splitAttr];
    if (X[i][splitAttr] > maxVal) maxVal = X[i][splitAttr];
  }

  // If all values are the same, create external node
  if (minVal === maxVal) {
    return node;
  }

  // Random split value between min and max
  const splitValue = minVal + random() * (maxVal - minVal);

  node.splitAttr = splitAttr;
  node.splitValue = splitValue;

  // Split data
  const leftData = [];
  const rightData = [];
  for (let i = 0; i < n; i++) {
    if (X[i][splitAttr] < splitValue) {
      leftData.push(X[i]);
    } else {
      rightData.push(X[i]);
    }
  }

  // Recursively build subtrees
  node.left = buildIsolationTree(leftData, heightLimit, currentHeight + 1);
  node.right = buildIsolationTree(rightData, heightLimit, currentHeight + 1);

  return node;
}

/**
 * Calculate path length for a sample in isolation tree
 */
function pathLength(x, tree, currentHeight = 0) {
  if (tree.isExternal()) {
    // Use average path length adjustment for external nodes
    return currentHeight + averagePathLength(tree.size);
  }

  const attr = tree.splitAttr;
  if (x[attr] < tree.splitValue) {
    return pathLength(x, tree.left, currentHeight + 1);
  } else {
    return pathLength(x, tree.right, currentHeight + 1);
  }
}

/**
 * Average path length of unsuccessful search in BST
 * Used for path length adjustment
 */
function averagePathLength(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;

  // H(n-1) is harmonic number, approximated by ln(n-1) + Euler's constant
  const eulerConstant = 0.5772156649;
  return 2 * (Math.log(n - 1) + eulerConstant) - (2 * (n - 1) / n);
}

/**
 * Isolation Forest for outlier detection
 * Compatible with sklearn.ensemble.IsolationForest
 *
 * Detects outliers using ensemble of isolation trees.
 * Outliers are isolated closer to the root of the tree.
 *
 * @example
 * const iso = new IsolationForest({ contamination: 0.1, n_estimators: 100 });
 * iso.fit(X_train);
 * const predictions = iso.predict(X_test);  // -1 for outliers, 1 for inliers
 * const scores = iso.score_samples(X_test); // Anomaly scores
 */
export class IsolationForest {
  /**
   * @param {Object} options
   * @param {number} options.n_estimators - Number of trees (default: 100)
   * @param {number} options.max_samples - Samples to draw for each tree (default: 'auto' = min(256, n))
   * @param {number} options.contamination - Expected proportion of outliers (default: 0.1)
   * @param {number} options.max_features - Features to draw for each tree (default: 1.0 = all)
   * @param {number} options.random_state - Random seed (default: null)
   */
  constructor({
    n_estimators = 100,
    max_samples = 'auto',
    contamination = 0.1,
    max_features = 1.0,
    random_state = null
  } = {}) {
    this.n_estimators = n_estimators;
    this.max_samples = max_samples;
    this.contamination = contamination;
    this.max_features = max_features;
    this.random_state = random_state;

    this.trees_ = null;
    this.max_samples_ = null;
    this.offset_ = null;
    this.threshold_ = null;
    this.nFeatures_ = null;
    this._tableColumns = null;
  }

  /**
   * Fit the model
   * @param {Array<Array<number>>|Object} X - Training data
   * @returns {IsolationForest} this
   */
  fit(X) {
    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
      });
      this._tableColumns = X.columns || X.X;
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    const n = X.length;
    const nFeatures = X[0].length;
    this.nFeatures_ = nFeatures;

    // Determine max_samples
    if (this.max_samples === 'auto') {
      this.max_samples_ = Math.min(256, n);
    } else if (typeof this.max_samples === 'number' && this.max_samples <= 1) {
      this.max_samples_ = Math.floor(this.max_samples * n);
    } else {
      this.max_samples_ = this.max_samples;
    }

    // Height limit for trees
    const heightLimit = Math.ceil(Math.log2(this.max_samples_));

    // Build ensemble of isolation trees
    this.trees_ = [];
    for (let i = 0; i < this.n_estimators; i++) {
      // Sample data for this tree
      const indices = Array.from({ length: n }, (_, i) => i);
      const sampleIndices = [];
      for (let j = 0; j < this.max_samples_; j++) {
        sampleIndices.push(indices[randomInt(0, n)]);
      }
      const sample = sampleIndices.map(idx => X[idx]);

      // Build tree
      const tree = buildIsolationTree(sample, heightLimit);
      this.trees_.push(tree);
    }

    // Compute offset for normalization (average path length for max_samples)
    this.offset_ = averagePathLength(this.max_samples_);

    // Compute threshold based on contamination
    const scores = this.score_samples(X);
    const sortedScores = [...scores].sort((a, b) => a - b);

    // Number of expected outliers (at least 1 if contamination > 0)
    const nOutliers = this.contamination > 0 ?
      Math.max(1, Math.floor(this.contamination * n)) :
      0;

    // Threshold is set so that scores below it are outliers
    // If we want k outliers, threshold should be sortedScores[k]
    // (so indices 0 to k-1 are below threshold)
    if (nOutliers === 0 || nOutliers >= n) {
      // No outliers or all outliers
      this.threshold_ = nOutliers === 0 ? -Infinity : Infinity;
    } else {
      this.threshold_ = sortedScores[nOutliers];
    }

    return this;
  }

  /**
   * Compute anomaly scores for samples
   * Lower (more negative) scores indicate outliers
   * Scores range approximately from -1 to 0
   * @param {Array<Array<number>>|Object} X - Data
   * @returns {Array<number>} Anomaly scores (negative values)
   */
  score_samples(X) {
    if (this.trees_ === null) {
      throw new Error('Model must be fitted before scoring');
    }

    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X || this._tableColumns,
        data: X.data,
        omit_missing: false,
      });
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    if (X[0].length !== this.nFeatures_) {
      throw new Error(`X has ${X[0].length} features, but model expected ${this.nFeatures_}`);
    }

    const scores = [];
    for (let i = 0; i < X.length; i++) {
      const x = X[i];

      // Average path length across all trees
      let avgPathLength = 0;
      for (const tree of this.trees_) {
        avgPathLength += pathLength(x, tree);
      }
      avgPathLength /= this.n_estimators;

      // Anomaly score (normalized)
      // Outliers have SHORTER paths (isolated faster)
      // Raw score: s(x,n) = 2^(-E(h(x))/c(n))
      // We negate so outliers have LOWER scores
      const rawScore = Math.pow(2, -avgPathLength / this.offset_);
      const score = -rawScore; // Negate: outliers (high raw score) → low score
      scores.push(score);
    }

    return scores;
  }

  /**
   * Predict if samples are outliers
   * @param {Array<Array<number>>|Object} X - Data
   * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
   */
  predict(X) {
    if (this.threshold_ === null) {
      throw new Error('Model must be fitted before prediction');
    }

    const scores = this.score_samples(X);
    return scores.map(score => score < this.threshold_ ? -1 : 1);
  }

  /**
   * Fit and predict in one step
   * @param {Array<Array<number>>|Object} X - Data
   * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
   */
  fit_predict(X) {
    return this.fit(X).predict(X);
  }
}

// ============= LocalOutlierFactor =============

/**
 * Euclidean distance between two vectors
 */
function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Local Outlier Factor for outlier detection
 * Compatible with sklearn.neighbors.LocalOutlierFactor
 *
 * Detects outliers using local density deviation.
 * LOF > 1 indicates outlier (lower local density than neighbors).
 *
 * @example
 * const lof = new LocalOutlierFactor({ n_neighbors: 20, contamination: 0.1 });
 * lof.fit(X_train);
 * const predictions = lof.predict(X_test);  // -1 for outliers, 1 for inliers
 */
export class LocalOutlierFactor {
  /**
   * @param {Object} options
   * @param {number} options.n_neighbors - Number of neighbors (default: 20)
   * @param {string} options.algorithm - 'auto' (only option for now)
   * @param {Function} options.metric - Distance function (default: euclidean)
   * @param {number} options.contamination - Expected proportion of outliers (default: 0.1)
   * @param {string} options.novelty - If true, can predict on new data (default: false)
   */
  constructor({
    n_neighbors = 20,
    algorithm = 'auto',
    metric = null,
    contamination = 0.1,
    novelty = false
  } = {}) {
    if (n_neighbors <= 0) {
      throw new Error('n_neighbors must be positive');
    }

    this.n_neighbors = n_neighbors;
    this.algorithm = algorithm;
    this.metric = metric || euclideanDistance;
    this.contamination = contamination;
    this.novelty = novelty;

    this.X_ = null;
    this.negative_outlier_factor_ = null;
    this.offset_ = null;
    this.threshold_ = null;
    this.nFeatures_ = null;
    this._tableColumns = null;
  }

  /**
   * Fit the model
   * @param {Array<Array<number>>|Object} X - Training data
   * @returns {LocalOutlierFactor} this
   */
  fit(X) {
    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
      });
      this._tableColumns = X.columns || X.X;
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    const n = X.length;
    this.X_ = X.map(row => [...row]); // Store copy
    this.nFeatures_ = X[0].length;

    if (this.n_neighbors >= n) {
      throw new Error(`n_neighbors (${this.n_neighbors}) must be less than n_samples (${n})`);
    }

    // Compute pairwise distances
    const distances = this._pairwiseDistances(X);

    // For each point, find k-nearest neighbors and k-distance
    const kDistances = new Array(n);
    const kNeighbors = new Array(n);

    for (let i = 0; i < n; i++) {
      // Get distances to all other points
      const dists = distances[i]
        .map((d, j) => ({ dist: d, idx: j }))
        .filter((_, j) => j !== i); // Exclude self

      // Sort and take k nearest
      dists.sort((a, b) => a.dist - b.dist);
      const neighbors = dists.slice(0, this.n_neighbors);

      kNeighbors[i] = neighbors.map(n => n.idx);
      kDistances[i] = neighbors[this.n_neighbors - 1].dist; // k-distance
    }

    // Compute reachability distances
    const reachabilityDistances = new Array(n);
    for (let i = 0; i < n; i++) {
      reachabilityDistances[i] = new Array(n);
      for (let j = 0; j < n; j++) {
        if (i === j) {
          reachabilityDistances[i][j] = 0;
        } else {
          // reach-dist(A,B) = max(k-distance(B), dist(A,B))
          reachabilityDistances[i][j] = Math.max(kDistances[j], distances[i][j]);
        }
      }
    }

    // Compute Local Reachability Density (LRD)
    const lrd = new Array(n);
    for (let i = 0; i < n; i++) {
      let sumReachDist = 0;
      for (const neighborIdx of kNeighbors[i]) {
        sumReachDist += reachabilityDistances[i][neighborIdx];
      }
      lrd[i] = kNeighbors[i].length / sumReachDist;
    }

    // Compute Local Outlier Factor (LOF)
    this.negative_outlier_factor_ = new Array(n);
    for (let i = 0; i < n; i++) {
      let sumLrdRatio = 0;
      for (const neighborIdx of kNeighbors[i]) {
        sumLrdRatio += lrd[neighborIdx] / lrd[i];
      }
      const lof = sumLrdRatio / kNeighbors[i].length;
      this.negative_outlier_factor_[i] = -lof; // Negative for sklearn compatibility
    }

    // Compute threshold based on contamination
    const lofScores = this.negative_outlier_factor_.map(x => -x); // Convert back to positive
    const sortedScores = [...lofScores].sort((a, b) => b - a); // Descending (higher LOF = outlier)

    // Number of expected outliers (at least 1 if contamination > 0)
    const nOutliers = this.contamination > 0 ?
      Math.max(1, Math.floor(this.contamination * n)) :
      0;

    // Threshold: negative_outlier_factor_ values below this are outliers
    // Higher LOF (more positive) = outlier, so more negative = outlier in our negative scale
    if (nOutliers === 0 || nOutliers >= n) {
      this.threshold_ = nOutliers === 0 ? Infinity : -Infinity;
    } else {
      // sortedScores is LOF in descending order: [highest_LOF, ..., lowest_LOF]
      // We want the top nOutliers to be outliers
      // Cutoff should be between sortedScores[nOutliers-1] and sortedScores[nOutliers]
      // In negative scale: NOF < -sortedScores[nOutliers] are outliers
      this.threshold_ = -sortedScores[nOutliers];
    }

    this.offset_ = -1.0; // sklearn compatibility

    return this;
  }

  /**
   * Compute pairwise distances
   */
  _pairwiseDistances(X) {
    const n = X.length;
    const distances = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dist = this.metric(X[i], X[j]);
        distances[i][j] = dist;
        distances[j][i] = dist;
      }
    }

    return distances;
  }

  /**
   * Predict if samples are outliers
   * @param {Array<Array<number>>|Object} X - Data (must be training data if novelty=false)
   * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
   */
  predict(X) {
    if (this.negative_outlier_factor_ === null) {
      throw new Error('Model must be fitted before prediction');
    }

    if (!this.novelty) {
      // Can only predict on training data
      if (X !== this.X_ && (!Array.isArray(X) || X.length !== this.X_.length)) {
        throw new Error('LocalOutlierFactor with novelty=false can only predict on training data');
      }
    }

    // For non-novelty mode, use precomputed scores
    return this.negative_outlier_factor_.map(score =>
      score < this.threshold_ ? -1 : 1
    );
  }

  /**
   * Fit and predict in one step
   * @param {Array<Array<number>>|Object} X - Data
   * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
   */
  fit_predict(X) {
    return this.fit(X).predict(X);
  }

  /**
   * Get negative outlier factor for each sample
   * @returns {Array<number>} Negative outlier factors
   */
  get negative_outlier_factor() {
    return this.negative_outlier_factor_;
  }
}

// ============= MahalanobisDistance =============

/**
 * Mahalanobis distance-based outlier detection
 * Compatible with sklearn.covariance.EllipticEnvelope approach
 *
 * Detects outliers based on statistical distance from the mean,
 * accounting for covariance structure. Uses pseudoinverse to handle
 * singular/near-singular covariance matrices.
 *
 * @example
 * const md = new MahalanobisDistance({ contamination: 0.1 });
 * md.fit(X_train);
 * const predictions = md.predict(X_test);
 */
export class MahalanobisDistance {
  /**
   * @param {Object} options
   * @param {number} options.contamination - Expected proportion of outliers (default: 0.1)
   * @param {boolean} options.use_chi2 - Use chi-squared distribution for threshold (default: true)
   */
  constructor({ contamination = 0.1, use_chi2 = true } = {}) {
    if (contamination < 0 || contamination > 0.5) {
      throw new Error('contamination must be in [0, 0.5]');
    }

    this.contamination = contamination;
    this.use_chi2 = use_chi2;
    this.mean_ = null;
    this.precision_ = null; // Inverse covariance matrix
    this.threshold_ = null;
    this.nFeatures_ = null;
    this._tableColumns = null;
  }

  /**
   * Fit the detector on training data
   * @param {Array<Array<number>>|Object} X - Training data
   * @returns {MahalanobisDistance} this
   */
  fit(X) {
    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: true, // Remove missing values for robust estimation
      });
      this._tableColumns = X.columns || X.X;
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    const n = X.length;
    const nFeatures = X[0].length;
    this.nFeatures_ = nFeatures;

    if (n < nFeatures) {
      throw new Error('Number of samples must be >= number of features for robust estimation');
    }

    // Compute mean
    this.mean_ = new Array(nFeatures).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nFeatures; j++) {
        this.mean_[j] += X[i][j];
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      this.mean_[j] /= n;
    }

    // Compute covariance matrix
    const Xmat = new Matrix(X);
    const meanMat = Matrix.ones(n, 1).mmul(new Matrix([this.mean_]));
    const Xcentered = Xmat.sub(meanMat);
    const cov = Xcentered.transpose().mmul(Xcentered).div(n - 1);

    // Compute precision matrix (inverse covariance) using pseudoinverse
    // This handles singular/near-singular covariance matrices
    this.precision_ = pseudoInverse(cov);

    // Compute distances and threshold
    const distances = this._mahalanobis_distances(X);
    const sortedDistances = [...distances].sort((a, b) => a - b);

    if (this.use_chi2) {
      // Use chi-squared percentile for threshold
      // For now, use empirical contamination-based threshold
      const nOutliers = Math.max(1, Math.floor(this.contamination * n));
      this.threshold_ = nOutliers >= n ? Infinity : sortedDistances[n - nOutliers];
    } else {
      // Simple empirical threshold
      const nOutliers = Math.max(1, Math.floor(this.contamination * n));
      this.threshold_ = nOutliers >= n ? Infinity : sortedDistances[n - nOutliers];
    }

    return this;
  }

  /**
   * Compute Mahalanobis distances for samples
   * @param {Array<Array<number>>} X - Data
   * @returns {Array<number>} Mahalanobis distances
   */
  _mahalanobis_distances(X) {
    const n = X.length;
    const distances = [];

    for (let i = 0; i < n; i++) {
      // Compute (x - mean)
      const diff = [];
      for (let j = 0; j < this.nFeatures_; j++) {
        diff.push(X[i][j] - this.mean_[j]);
      }

      // Compute Mahalanobis distance: sqrt((x - μ)' * Σ^(-1) * (x - μ))
      const diffMat = new Matrix([diff]);
      const mahal_sq = diffMat.mmul(this.precision_).mmul(diffMat.transpose()).get(0, 0);
      const distance = Math.sqrt(Math.max(0, mahal_sq)); // Ensure non-negative

      distances.push(distance);
    }

    return distances;
  }

  /**
   * Compute Mahalanobis distances for samples
   * @param {Array<Array<number>>|Object} X - Data to score
   * @returns {Array<number>} Negative Mahalanobis distances (outliers have lower scores)
   */
  score_samples(X) {
    if (this.mean_ === null || this.precision_ === null) {
      throw new Error('Model must be fitted before scoring');
    }

    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X || this._tableColumns,
        data: X.data,
        omit_missing: true,
      });
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    if (X[0].length !== this.nFeatures_) {
      throw new Error(`X has ${X[0].length} features, but detector expected ${this.nFeatures_}`);
    }

    const distances = this._mahalanobis_distances(X);
    // Return negative distances (sklearn convention: lower = outlier)
    return distances.map(d => -d);
  }

  /**
   * Predict if samples are outliers
   * @param {Array<Array<number>>|Object} X - Data
   * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
   */
  predict(X) {
    if (this.threshold_ === null) {
      throw new Error('Model must be fitted before prediction');
    }

    const scores = this.score_samples(X);
    // Outliers have lower scores (more negative), i.e., higher distances
    return scores.map(score => -score >= this.threshold_ ? -1 : 1);
  }

  /**
   * Fit and predict in one step
   * @param {Array<Array<number>>|Object} X - Data
   * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
   */
  fit_predict(X) {
    return this.fit(X).predict(X);
  }

  /**
   * Get Mahalanobis distances for fitted data
   * @returns {Array<number>} Mahalanobis distances
   */
  get mahalanobis_distances() {
    if (this.mean_ === null) {
      throw new Error('Model must be fitted first');
    }
    return this._mahalanobis_distances_cache || [];
  }
}

// ============= Functional Exports =============

/**
 * Isolation Forest (functional interface)
 * @param {Array<Array<number>>} X - Data
 * @param {Object} options - IsolationForest options
 * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
 */
export function isolationForest(X, options = {}) {
  const iso = new IsolationForest(options);
  return iso.fit_predict(X);
}

/**
 * Local Outlier Factor (functional interface)
 * @param {Array<Array<number>>} X - Data
 * @param {Object} options - LOF options
 * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
 */
export function localOutlierFactor(X, options = {}) {
  const lof = new LocalOutlierFactor(options);
  return lof.fit_predict(X);
}

/**
 * Mahalanobis Distance (functional interface)
 * @param {Array<Array<number>>} X - Data
 * @param {Object} options - MahalanobisDistance options
 * @returns {Array<number>} Predictions: -1 for outliers, 1 for inliers
 */
export function mahalanobisDistance(X, options = {}) {
  const md = new MahalanobisDistance(options);
  return md.fit_predict(X);
}
