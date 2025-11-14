/**
 * Missing Data Imputation
 *
 * Provides sklearn-compatible imputers for handling missing values:
 * - SimpleImputer: Fill with mean, median, mode, or constant
 * - KNNImputer: Fill using k-nearest neighbors
 * - IterativeImputer: Multivariate imputation (MICE algorithm)
 */

import { mean as calculateMean } from '../core/math.js';
import { prepareX } from '../core/table.js';

/**
 * Helper: Check if value is missing (NaN, null, undefined)
 */
function isMissing(value) {
  return value === null || value === undefined || (typeof value === 'number' && isNaN(value));
}

/**
 * Helper: Get indices of missing values in array
 */
function getMissingIndices(arr) {
  const indices = [];
  for (let i = 0; i < arr.length; i++) {
    if (isMissing(arr[i])) {
      indices.push(i);
    }
  }
  return indices;
}

/**
 * Helper: Get non-missing values from array
 */
function getNonMissing(arr) {
  return arr.filter(val => !isMissing(val));
}

/**
 * Helper: Calculate median
 */
function calculateMedian(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

/**
 * Helper: Calculate mode (most frequent value)
 */
function calculateMode(arr) {
  const counts = new Map();
  let maxCount = 0;
  let mode = arr[0];

  for (const val of arr) {
    const count = (counts.get(val) || 0) + 1;
    counts.set(val, count);
    if (count > maxCount) {
      maxCount = count;
      mode = val;
    }
  }

  return mode;
}

// ============= SimpleImputer =============

/**
 * Simple imputation strategies for missing values
 * Compatible with sklearn.impute.SimpleImputer
 *
 * @example
 * const imputer = new SimpleImputer({ strategy: 'mean' });
 * imputer.fit(X_train);
 * const X_filled = imputer.transform(X_test);
 */
export class SimpleImputer {
  /**
   * @param {Object} options
   * @param {string} options.strategy - 'mean', 'median', 'most_frequent', or 'constant' (default: 'mean')
   * @param {number|string} options.fill_value - Value to use for 'constant' strategy
   * @param {boolean} options.copy - If true, create copy of X (default: true)
   */
  constructor({ strategy = 'mean', fill_value = null, copy = true } = {}) {
    const validStrategies = ['mean', 'median', 'most_frequent', 'constant'];
    if (!validStrategies.includes(strategy)) {
      throw new Error(`Invalid strategy: ${strategy}. Must be one of ${validStrategies.join(', ')}`);
    }

    if (strategy === 'constant' && fill_value === null) {
      throw new Error('fill_value must be provided when strategy is "constant"');
    }

    this.strategy = strategy;
    this.fill_value = fill_value;
    this.copy = copy;
    this.statistics_ = null;
    this.nFeatures_ = null;
    this._tableColumns = null;
  }

  /**
   * Fit the imputer on training data
   * @param {Array<Array<number>>|Object} X - Training data or table object
   * @returns {SimpleImputer} this
   */
  fit(X) {
    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: false, // Don't drop missing values during prep
      });
      this._tableColumns = X.columns || X.X;
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.nFeatures_ = nFeatures;

    // Compute statistics for each feature
    this.statistics_ = new Array(nFeatures);

    for (let j = 0; j < nFeatures; j++) {
      // Extract column j (excluding missing values)
      const column = [];
      for (let i = 0; i < nSamples; i++) {
        if (!isMissing(X[i][j])) {
          column.push(X[i][j]);
        }
      }

      if (column.length === 0) {
        // All values are missing - use 0 for numeric strategies
        this.statistics_[j] = this.strategy === 'constant' ? this.fill_value : 0;
        continue;
      }

      // Compute statistic based on strategy
      switch (this.strategy) {
        case 'mean':
          this.statistics_[j] = calculateMean(column);
          break;
        case 'median':
          this.statistics_[j] = calculateMedian(column);
          break;
        case 'most_frequent':
          this.statistics_[j] = calculateMode(column);
          break;
        case 'constant':
          this.statistics_[j] = this.fill_value;
          break;
      }
    }

    return this;
  }

  /**
   * Transform data by filling missing values
   * @param {Array<Array<number>>|Object} X - Data to transform or table object
   * @returns {Array<Array<number>>} Transformed data
   */
  transform(X) {
    if (this.statistics_ === null) {
      throw new Error('Imputer must be fitted before transform');
    }

    // Handle table input
    let isTable = false;
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X || this._tableColumns,
        data: X.data,
        omit_missing: false,
      });
      X = prepared.X;
      isTable = true;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    if (X[0].length !== this.nFeatures_) {
      throw new Error(`X has ${X[0].length} features, but imputer expected ${this.nFeatures_}`);
    }

    // Create copy if requested
    const result = this.copy ? X.map(row => [...row]) : X;

    // Fill missing values
    for (let i = 0; i < result.length; i++) {
      for (let j = 0; j < this.nFeatures_; j++) {
        if (isMissing(result[i][j])) {
          result[i][j] = this.statistics_[j];
        }
      }
    }

    return result;
  }

  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>|Object} X - Data to fit and transform
   * @returns {Array<Array<number>>} Transformed data
   */
  fit_transform(X) {
    return this.fit(X).transform(X);
  }
}

// ============= KNNImputer =============

/**
 * Imputation using k-Nearest Neighbors
 * Compatible with sklearn.impute.KNNImputer
 *
 * Missing values are imputed using the mean value from the k nearest
 * neighbors found in the training set.
 *
 * @example
 * const imputer = new KNNImputer({ n_neighbors: 5 });
 * imputer.fit(X_train);
 * const X_filled = imputer.transform(X_test);
 */
export class KNNImputer {
  /**
   * @param {Object} options
   * @param {number} options.n_neighbors - Number of neighbors to use (default: 5)
   * @param {string} options.weights - 'uniform' or 'distance' (default: 'uniform')
   * @param {Function} options.metric - Distance function (default: euclidean)
   * @param {boolean} options.copy - If true, create copy of X (default: true)
   */
  constructor({ n_neighbors = 5, weights = 'uniform', metric = null, copy = true } = {}) {
    if (n_neighbors <= 0) {
      throw new Error('n_neighbors must be positive');
    }

    this.n_neighbors = n_neighbors;
    this.weights = weights;
    this.metric = metric || this._euclideanDistance;
    this.copy = copy;
    this.X_ = null;
    this.nFeatures_ = null;
    this._tableColumns = null;
  }

  /**
   * Euclidean distance between two vectors (ignoring missing values)
   */
  _euclideanDistance(a, b) {
    let sum = 0;
    let count = 0;

    for (let i = 0; i < a.length; i++) {
      // Skip if either value is missing
      if (isMissing(a[i]) || isMissing(b[i])) {
        continue;
      }
      const diff = a[i] - b[i];
      sum += diff * diff;
      count++;
    }

    // If no common non-missing features, return Infinity
    if (count === 0) {
      return Infinity;
    }

    return Math.sqrt(sum);
  }

  /**
   * Fit the imputer on training data
   * @param {Array<Array<number>>|Object} X - Training data
   * @returns {KNNImputer} this
   */
  fit(X) {
    // Handle table input
    if (X && typeof X === 'object' && !Array.isArray(X)) {
      const prepared = prepareX({
        columns: X.columns || X.X,
        data: X.data,
        omit_missing: false,
      });
      this._tableColumns = X.columns || X.X;
      X = prepared.X;
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error('X must be a 2D array or table object');
    }

    // Store training data
    this.X_ = this.copy ? X.map(row => [...row]) : X;
    this.nFeatures_ = X[0].length;

    return this;
  }

  /**
   * Transform data by filling missing values using KNN
   * @param {Array<Array<number>>|Object} X - Data to transform
   * @param {Array<number>} exclude_indices - Row indices to exclude from neighbors (for fit_transform)
   * @returns {Array<Array<number>>} Transformed data
   */
  transform(X, exclude_indices = []) {
    if (this.X_ === null) {
      throw new Error('Imputer must be fitted before transform');
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
      throw new Error(`X has ${X[0].length} features, but imputer expected ${this.nFeatures_}`);
    }

    // Create copy
    const result = this.copy ? X.map(row => [...row]) : X;

    // For each row with missing values
    for (let i = 0; i < result.length; i++) {
      const row = result[i];
      const missingIndices = getMissingIndices(row);

      if (missingIndices.length === 0) {
        continue; // No missing values
      }

      // Find k nearest neighbors from training data (excluding this row if in exclude_indices)
      const excludeSet = new Set(exclude_indices);
      const neighbors = this._findNeighbors(row, this.n_neighbors, excludeSet.has(i) ? i : -1);

      // Impute each missing feature
      for (const j of missingIndices) {
        const values = [];
        const weights = [];

        for (const neighbor of neighbors) {
          if (!isMissing(neighbor.row[j])) {
            values.push(neighbor.row[j]);
            weights.push(neighbor.weight);
          }
        }

        if (values.length === 0) {
          // No neighbors have this feature - use 0
          result[i][j] = 0;
        } else {
          // Weighted mean
          if (this.weights === 'distance') {
            const weightSum = weights.reduce((a, b) => a + b, 0);
            const weightedSum = values.reduce((sum, val, idx) => sum + val * weights[idx], 0);
            result[i][j] = weightedSum / weightSum;
          } else {
            result[i][j] = calculateMean(values);
          }
        }
      }
    }

    return result;
  }

  /**
   * Find k nearest neighbors for a given row
   * @param {number} excludeIdx - Row index to exclude (-1 for none)
   */
  _findNeighbors(row, k, excludeIdx = -1) {
    const distances = [];

    for (let i = 0; i < this.X_.length; i++) {
      // Skip the row itself if doing fit_transform
      if (i === excludeIdx) {
        continue;
      }

      const dist = this.metric(row, this.X_[i]);

      // Skip identical rows (distance ~0) to avoid using a row's own NaN values
      if (dist < 1e-10) {
        continue;
      }

      distances.push({
        dist,
        row: this.X_[i],
        weight: 1 / (dist + 1e-10), // Distance weighting
      });
    }

    // Sort by distance and take k nearest
    distances.sort((a, b) => a.dist - b.dist);
    return distances.slice(0, k);
  }

  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>|Object} X - Data to fit and transform
   * @returns {Array<Array<number>>} Transformed data
   */
  fit_transform(X) {
    this.fit(X);
    // Pass all indices to exclude each row from its own neighbor search
    const excludeIndices = Array.from({ length: this.X_.length }, (_, i) => i);
    return this.transform(this.X_, excludeIndices);
  }
}

// ============= Functional Exports =============

/**
 * Simple imputation (functional interface)
 * @param {Array<Array<number>>} X - Data with missing values
 * @param {Object} options - Imputer options
 * @returns {Array<Array<number>>} Imputed data
 */
export function simpleImpute(X, options = {}) {
  const imputer = new SimpleImputer(options);
  return imputer.fit_transform(X);
}

/**
 * KNN imputation (functional interface)
 * @param {Array<Array<number>>} X - Data with missing values
 * @param {Object} options - Imputer options
 * @returns {Array<Array<number>>} Imputed data
 */
export function knnImpute(X, options = {}) {
  const imputer = new KNNImputer(options);
  return imputer.fit_transform(X);
}
