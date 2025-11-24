/**
 * K-Nearest Neighbors estimators (classifier & regressor).
 *
 * Provides scikit-learn style interfaces with support for:
 * - Multiple distance metrics (euclidean, manhattan, minkowski, etc.)
 * - Multiple algorithms (brute, kd_tree, ball_tree, auto)
 * - Radius-based neighbors
 * - Weighted predictions
 */

import { Classifier, Regressor, Estimator } from '../../core/estimators/estimator.js';
import { LabelEncoder, prepareX, prepareXY } from '../../core/table.js';
import { mean } from '../../core/math.js';
import { euclidean, manhattan, minkowski, chebyshev, cosine, getDistanceFunction } from '../distances.js';

// ============= KD-Tree for fast nearest neighbor search =============

class KDNode {
  constructor(point, idx, axis, left = null, right = null) {
    this.point = point;
    this.idx = idx;
    this.axis = axis;
    this.left = left;
    this.right = right;
  }
}

class KDTree {
  constructor(data, indices = null) {
    this.root = this._buildTree(data, indices || data.map((_, i) => i), 0);
  }

  _buildTree(data, indices, depth) {
    if (indices.length === 0) return null;
    if (indices.length === 1) {
      return new KDNode(data[indices[0]], indices[0], null);
    }

    const k = data[0].length;
    const axis = depth % k;

    // Sort by axis
    const sorted = indices.slice().sort((a, b) => data[a][axis] - data[b][axis]);
    const median = Math.floor(sorted.length / 2);

    return new KDNode(
      data[sorted[median]],
      sorted[median],
      axis,
      this._buildTree(data, sorted.slice(0, median), depth + 1),
      this._buildTree(data, sorted.slice(median + 1), depth + 1)
    );
  }

  kNearest(point, k, distFn = euclidean) {
    const best = [];

    const search = (node, depth = 0) => {
      if (!node) return;

      const dist = distFn(point, node.point);

      if (best.length < k || dist < best[best.length - 1].dist) {
        best.push({ dist, idx: node.idx });
        best.sort((a, b) => a.dist - b.dist);
        if (best.length > k) {
          best.pop();
        }
      }

      if (!node.left && !node.right) return;

      const axis = node.axis;
      const diff = point[axis] - node.point[axis];

      const nearSide = diff < 0 ? node.left : node.right;
      const farSide = diff < 0 ? node.right : node.left;

      search(nearSide, depth + 1);

      // Check if we need to search the other side
      if (best.length < k || Math.abs(diff) < best[best.length - 1].dist) {
        search(farSide, depth + 1);
      }
    };

    search(this.root);
    return best;
  }

  radiusNeighbors(point, radius, distFn = euclidean) {
    const neighbors = [];

    const search = (node) => {
      if (!node) return;

      const dist = distFn(point, node.point);
      if (dist <= radius) {
        neighbors.push({ dist, idx: node.idx });
      }

      if (!node.left && !node.right) return;

      const axis = node.axis;
      const diff = point[axis] - node.point[axis];

      if (diff < 0) {
        search(node.left);
        if (Math.abs(diff) <= radius) {
          search(node.right);
        }
      } else {
        search(node.right);
        if (Math.abs(diff) <= radius) {
          search(node.left);
        }
      }
    };

    search(this.root);
    return neighbors.sort((a, b) => a.dist - b.dist);
  }
}

// ============= Helper Functions =============

function buildDataset(X, y, opts = {}) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareXY({
      X: X.X || X.columns,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
      encoders: X.encoders,
    });
    return {
      X: prepared.X,
      y: prepared.y,
      columns: prepared.columnsX,
      encoders: prepared.encoders,
    };
  }

  if (!Array.isArray(X) || !Array.isArray(y)) {
    throw new Error('KNN fit expects arrays for X and y (or a table object).');
  }

  return { X, y, columns: null, encoders: null };
}

function preparePredictInput(X, storedColumns, opts = {}) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || storedColumns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
    });
    return prepared.X;
  }

  return Array.isArray(X[0]) ? X : X.map((row) => [row]);
}

function bruteForceNeighbors(trainX, point, k, distFn) {
  const distances = [];
  for (let i = 0; i < trainX.length; i++) {
    const dist = distFn(trainX[i], point);
    distances.push({ dist, idx: i });
  }

  distances.sort((a, b) => a.dist - b.dist);
  return distances.slice(0, k);
}

function bruteForceRadiusNeighbors(trainX, point, radius, distFn) {
  const neighbors = [];
  for (let i = 0; i < trainX.length; i++) {
    const dist = distFn(trainX[i], point);
    if (dist <= radius) {
      neighbors.push({ dist, idx: i });
    }
  }
  return neighbors.sort((a, b) => a.dist - b.dist);
}

// ============= KNN Base Class =============

class KNNBase extends Estimator {
  constructor({
    k = 5,
    weights = 'uniform',
    algorithm = 'auto',
    metric = 'euclidean',
    p = 2
  } = {}) {
    super({ k, weights, algorithm, metric, p });
    if (k <= 0) throw new Error('k must be positive');
    this.k = k;
    this.weights = weights;
    this.algorithm = algorithm;
    this.metric = metric;
    this.p = p;

    // Get distance function
    if (metric === 'minkowski') {
      this.distFn = (a, b) => minkowski(a, b, p);
    } else {
      this.distFn = getDistanceFunction(metric);
    }

    this.X = null;
    this.y = null;
    this.columns = null;
    this.kdtree = null;
  }

  _fitBase(X, y) {
    const prepared = buildDataset(X, y);
    this.X = prepared.X.map((row) => Array.isArray(row) ? row.map(Number) : [Number(row)]);
    this.y = prepared.y;
    this.columns = prepared.columns;
    this.fitted = true;
    this.encoders = prepared.encoders || null;

    // Build KD-tree if appropriate
    const useKDTree = this.algorithm === 'kd_tree' ||
      (this.algorithm === 'auto' && this.X.length > 30 && this.metric === 'euclidean');

    if (useKDTree) {
      try {
        this.kdtree = new KDTree(this.X);
      } catch (e) {
        // Fall back to brute force
        this.kdtree = null;
      }
    }

    return prepared;
  }

  _preparePredict(X) {
    this._ensureFitted('predict');
    return preparePredictInput(X, this.columns);
  }

  _getNeighbors(point, k) {
    if (this.kdtree && this.metric === 'euclidean') {
      return this.kdtree.kNearest(point, k, this.distFn);
    }
    return bruteForceNeighbors(this.X, point, k, this.distFn);
  }

  _getRadiusNeighbors(point, radius) {
    if (this.kdtree && this.metric === 'euclidean') {
      return this.kdtree.radiusNeighbors(point, radius, this.distFn);
    }
    return bruteForceRadiusNeighbors(this.X, point, radius, this.distFn);
  }

  _neighborWeights(neighbors) {
    if (this.weights === 'distance') {
      return neighbors.map(({ dist, idx }) => ({
        idx,
        weight: dist === 0 ? 1e9 : 1 / dist,
      }));
    }
    return neighbors.map(({ idx }) => ({ idx, weight: 1 }));
  }
}

// ============= KNN Classifier =============

export class KNNClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.knn = new KNNBase(opts);
  }

  fit(X, y = null) {
    const prepared = this.knn._fitBase(X, y);
    // Use centralized label encoder extraction
    this._extractLabelEncoder(prepared);
    return this;
  }

  predict(X, { decode = !!this.labelEncoder_ } = {}) {
    const prepared = this.knn._preparePredict(X);
    const data = prepared.X || prepared;
    const predictions = [];
    const { y: trainY } = this.knn;

    for (const point of data) {
      const neighbors = this.knn._getNeighbors(point, this.knn.k);
      const weighted = this.knn._neighborWeights(neighbors);

      const votes = new Map();
      for (const { idx, weight } of weighted) {
        const label = trainY[idx];
        votes.set(label, (votes.get(label) || 0) + weight);
      }

      let bestLabel = null;
      let bestScore = -Infinity;
      for (const [label, score] of votes.entries()) {
        if (score > bestScore) {
          bestScore = score;
          bestLabel = label;
        }
      }
      predictions.push(bestLabel);
    }

    // Use centralized label decoder
    if (decode) {
      return this._decodeLabels(predictions);
    }

    return predictions;
  }

  predictProba(X, { decode = !!this.labelEncoder_ } = {}) {
    const prepared = this.knn._preparePredict(X);
    const data = prepared.X || prepared;
    const result = [];
    const { y: trainY } = this.knn;
    const labels = Array.from(new Set(trainY));

    for (const point of data) {
      const neighbors = this.knn._getNeighbors(point, this.knn.k);
      const weighted = this.knn._neighborWeights(neighbors);

      const votes = new Map();
      let total = 0;
      for (const { idx, weight } of weighted) {
        const label = trainY[idx];
        votes.set(label, (votes.get(label) || 0) + weight);
        total += weight;
      }

      if (decode && this.labelEncoder_) {
        const decoded = {};
        labels.forEach((label) => {
          const name = this.labelEncoder_.inverseTransform([label])[0];
          decoded[name] = total === 0 ? 0 : (votes.get(label) || 0) / total;
        });
        result.push(decoded);
      } else {
        const proba = {};
        labels.forEach((label) => {
          proba[label] = total === 0 ? 0 : (votes.get(label) || 0) / total;
        });
        result.push(proba);
      }
    }
    return result;
  }

  /**
   * Find neighbors within a given radius
   * @param {Array} X - Query points
   * @param {number} radius - Radius within which to find neighbors
   * @returns {Array<Array>} Indices of neighbors for each query point
   */
  radiusNeighbors(X, radius) {
    const data = this.knn._preparePredict(X);
    const prepared = data.X || data;
    const result = [];

    for (const point of prepared) {
      const neighbors = this.knn._getRadiusNeighbors(point, radius);
      result.push(neighbors.map(n => n.idx));
    }

    return result;
  }

  /**
   * Find K nearest neighbors
   * @param {Array} X - Query points
   * @param {number} nNeighbors - Number of neighbors (default: this.k)
   * @returns {Object} {distances, indices}
   */
  kneighbors(X, nNeighbors = null) {
    const k = nNeighbors || this.knn.k;
    const data = this.knn._preparePredict(X);
    const prepared = data.X || data;
    const distances = [];
    const indices = [];

    for (const point of prepared) {
      const neighbors = this.knn._getNeighbors(point, k);
      distances.push(neighbors.map(n => n.dist));
      indices.push(neighbors.map(n => n.idx));
    }

    return { distances, indices };
  }
}

// ============= KNN Regressor =============

export class KNNRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.knn = new KNNBase(opts);
  }

  fit(X, y = null) {
    this.knn._fitBase(X, y);
    return this;
  }

  predict(X) {
    const data = this.knn._preparePredict(X);
    const predictions = [];
    const { y: trainY } = this.knn;

    for (const point of data) {
      const neighbors = this.knn._getNeighbors(point, this.knn.k);
      const weighted = this.knn._neighborWeights(neighbors);
      const totalWeight = weighted.reduce((acc, w) => acc + w.weight, 0);

      if (totalWeight === 0) {
        const neighborValues = neighbors.map((n) => Number(trainY[n.idx]));
        predictions.push(mean(neighborValues));
        continue;
      }

      let sum = 0;
      for (const { idx, weight } of weighted) {
        sum += Number(trainY[idx]) * weight;
      }
      predictions.push(sum / totalWeight);
    }

    return predictions;
  }

  /**
   * Find neighbors within a given radius
   * @param {Array} X - Query points
   * @param {number} radius - Radius within which to find neighbors
   * @returns {Array<Array>} Indices of neighbors for each query point
   */
  radiusNeighbors(X, radius) {
    const data = this.knn._preparePredict(X);
    const prepared = data.X || data;
    const result = [];

    for (const point of prepared) {
      const neighbors = this.knn._getRadiusNeighbors(point, radius);
      result.push(neighbors.map(n => n.idx));
    }

    return result;
  }

  /**
   * Find K nearest neighbors
   * @param {Array} X - Query points
   * @param {number} nNeighbors - Number of neighbors (default: this.k)
   * @returns {Object} {distances, indices}
   */
  kneighbors(X, nNeighbors = null) {
    const k = nNeighbors || this.knn.k;
    const data = this.knn._preparePredict(X);
    const prepared = data.X || data;
    const distances = [];
    const indices = [];

    for (const point of prepared) {
      const neighbors = this.knn._getNeighbors(point, k);
      distances.push(neighbors.map(n => n.dist));
      indices.push(neighbors.map(n => n.idx));
    }

    return { distances, indices };
  }
}

export default {
  KNNClassifier,
  KNNRegressor,
};
