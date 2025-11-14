/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
 * A density-based clustering algorithm that can find arbitrarily shaped clusters
 * and identify outliers as noise.
 */

import { toMatrix } from '../core/linalg.js';
import { prepareX } from '../core/table.js';

/**
 * Euclidean distance between two points
 * @param {Array<number>} a - First point
 * @param {Array<number>} b - Second point
 * @returns {number} Distance
 */
function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

/**
 * Find all neighbors within eps distance of a point
 * @param {Array<Array<number>>} data - Data points
 * @param {number} pointIdx - Index of the point
 * @param {number} eps - Maximum distance for neighborhood
 * @returns {Array<number>} Indices of neighbors
 */
function regionQuery(data, pointIdx, eps) {
  const neighbors = [];
  const point = data[pointIdx];

  for (let i = 0; i < data.length; i++) {
    if (euclideanDistance(point, data[i]) <= eps) {
      neighbors.push(i);
    }
  }

  return neighbors;
}

/**
 * Expand cluster from a core point
 * @param {Array<Array<number>>} data - Data points
 * @param {Array<number>} labels - Current cluster labels
 * @param {number} pointIdx - Index of core point
 * @param {number} clusterId - ID of current cluster
 * @param {number} eps - Maximum distance for neighborhood
 * @param {number} minSamples - Minimum samples for core point
 * @returns {boolean} True if cluster was expanded
 */
function expandCluster(data, labels, pointIdx, clusterId, eps, minSamples) {
  const seeds = regionQuery(data, pointIdx, eps);

  // Not a core point
  if (seeds.length < minSamples) {
    labels[pointIdx] = -1; // Mark as noise
    return false;
  }

  // All points in seeds are part of this cluster
  for (const seedIdx of seeds) {
    labels[seedIdx] = clusterId;
  }

  // Remove the initial point from seeds (already processed)
  const seedsSet = new Set(seeds);
  seedsSet.delete(pointIdx);

  // Process each seed point
  const seedsArray = Array.from(seedsSet);
  let i = 0;

  while (i < seedsArray.length) {
    const currentIdx = seedsArray[i];
    const neighbors = regionQuery(data, currentIdx, eps);

    // If current point is a core point
    if (neighbors.length >= minSamples) {
      for (const neighborIdx of neighbors) {
        // If neighbor is noise or unvisited
        if (labels[neighborIdx] === -1 || labels[neighborIdx] === 0) {
          if (labels[neighborIdx] === 0) {
            // Unvisited point - add to seeds
            seedsArray.push(neighborIdx);
          }
          labels[neighborIdx] = clusterId;
        }
      }
    }

    i++;
  }

  return true;
}

/**
 * Fit DBSCAN clustering model
 * @param {Array<Array<number>>|Matrix} X - Data matrix (n samples × d features)
 * @param {Object} options - {eps: neighborhood radius, minSamples: min points for core}
 * @returns {Object} {labels, nClusters, nNoise, coreSampleIndices}
 */
export function fit(
  X,
  {
    eps = 0.5,
    minSamples = 5,
    columns = null,
    data: data_in = null,
  } = {},
) {
  // Accept either:
  //  - legacy numeric input: fit(X_array_or_matrix, { eps, ... })
  //  - declarative options-object as first arg: fit({ data, columns, eps, ... })
  let data;
  if (
    X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.columns)
  ) {
    const opts = X;
    data_in = opts.data !== undefined ? opts.data : data_in;
    columns = opts.columns !== undefined ? opts.columns : columns;
    eps = opts.eps !== undefined ? opts.eps : eps;
    minSamples = opts.minSamples !== undefined ? opts.minSamples : minSamples;
  }

  // If declarative data provided, prepare numeric matrix via prepareX
  if (data_in) {
    const prepared = prepareX({ columns, data: data_in });
    data = prepared.X;
  } else if (Array.isArray(X) && X.length > 0 && typeof X[0] === 'object' && !Array.isArray(X[0])) {
    // Array of objects (table-like data) - use prepareX
    const prepared = prepareX({ columns, data: X });
    data = prepared.X;
  } else if (Array.isArray(X)) {
    // Array of arrays (numeric matrix)
    data = X.map((row) => Array.isArray(row) ? row : [row]);
  } else {
    const mat = toMatrix(X);
    data = [];
    for (let i = 0; i < mat.rows; i++) {
      const row = [];
      for (let j = 0; j < mat.columns; j++) {
        row.push(mat.get(i, j));
      }
      data.push(row);
    }
  }

  const n = data.length;

  // Initialize all points as unvisited (label = 0)
  const labels = new Array(n).fill(0);
  let clusterId = 0;
  const coreSampleIndices = [];

  for (let i = 0; i < n; i++) {
    // Skip if already processed
    if (labels[i] !== 0) continue;

    const neighbors = regionQuery(data, i, eps);

    // Check if this is a core point
    if (neighbors.length >= minSamples) {
      coreSampleIndices.push(i);
      clusterId++;
      expandCluster(data, labels, i, clusterId, eps, minSamples);
    } else {
      // Mark as noise (will be changed if density-reachable from core point)
      labels[i] = -1;
    }
  }

  // Count clusters (labels go from 1 to clusterId)
  const nClusters = clusterId;

  // Count noise points (label = -1)
  const nNoise = labels.filter(label => label === -1).length;

  return {
    labels,
    nClusters,
    nNoise,
    coreSampleIndices,
  };
}

/**
 * Predict cluster labels for new data points
 * Note: DBSCAN doesn't naturally support prediction on new points.
 * This implementation assigns new points to the cluster of their nearest core point
 * if within eps distance, otherwise marks as noise.
 *
 * @param {Object} model - Fitted model from fit()
 * @param {Array<Array<number>>} X - New data points
 * @param {Array<Array<number>>} X_train - Original training data
 * @param {number} eps - Maximum distance for neighborhood
 * @returns {Array<number>} Cluster labels
 */
export function predict(model, X, X_train, eps) {
  const { labels: trainLabels, coreSampleIndices } = model;
  const data = Array.isArray(X[0]) ? X : X.map((x) => [x]);

  const predictions = new Array(data.length);

  for (let i = 0; i < data.length; i++) {
    const point = data[i];
    let minDist = Infinity;
    let nearestLabel = -1;

    // Find nearest core point
    for (const coreIdx of coreSampleIndices) {
      const dist = euclideanDistance(point, X_train[coreIdx]);
      if (dist < minDist) {
        minDist = dist;
        nearestLabel = trainLabels[coreIdx];
      }
    }

    // Assign to cluster if within eps distance, otherwise noise
    predictions[i] = minDist <= eps ? nearestLabel : -1;
  }

  return predictions;
}
