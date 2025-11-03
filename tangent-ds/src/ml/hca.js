/**
 * Hierarchical Clustering Analysis (HCA)
 * Agglomerative clustering with various linkage methods
 */

import { mean } from "../core/math.js";
import { prepareX } from "../core/table.js";

/**
 * Compute Euclidean distance between two points
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
 * Compute pairwise distance matrix
 * @param {Array<Array<number>>} data - Data points
 * @returns {Array<Array<number>>} Distance matrix
 */
function computeDistanceMatrix(data) {
  const n = data.length;
  const distances = Array(n).fill(null).map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dist = euclideanDistance(data[i], data[j]);
      distances[i][j] = dist;
      distances[j][i] = dist;
    }
  }

  return distances;
}

/**
 * Compute mean vector for a cluster
 * @param {Array<number>} cluster - Indices in cluster
 * @param {Array<Array<number>>} data - Data matrix
 * @returns {Array<number>} Mean vector
 */
function clusterMean(cluster, data) {
  const dimension = data[0].length;
  const meanVector = Array(dimension).fill(0);

  for (const index of cluster) {
    const point = data[index];
    for (let i = 0; i < dimension; i++) {
      meanVector[i] += point[i];
    }
  }

  const size = cluster.length;
  for (let i = 0; i < dimension; i++) {
    meanVector[i] /= size;
  }

  return meanVector;
}

/**
 * Compute Ward linkage distance between two clusters
 * @param {Array<number>} cluster1 - Indices in cluster 1
 * @param {Array<number>} cluster2 - Indices in cluster 2
 * @param {Array<Array<number>>} data - Data matrix
 * @returns {number} Ward distance
 */
function wardDistance(cluster1, cluster2, data) {
  const size1 = cluster1.length;
  const size2 = cluster2.length;
  const mean1 = clusterMean(cluster1, data);
  const mean2 = clusterMean(cluster2, data);

  let squaredDiff = 0;
  for (let i = 0; i < mean1.length; i++) {
    const diff = mean1[i] - mean2[i];
    squaredDiff += diff * diff;
  }

  return (size1 * size2) / (size1 + size2) * squaredDiff;
}

/**
 * Find minimum distance between two clusters
 * @param {Array<number>} cluster1 - Indices in cluster 1
 * @param {Array<number>} cluster2 - Indices in cluster 2
 * @param {Array<Array<number>>} distances - Distance matrix
 * @param {string} linkage - Linkage method
 * @param {Array<Array<number>>} data - Data matrix (required for Ward)
 * @returns {number} Distance between clusters
 */
function clusterDistance(cluster1, cluster2, distances, linkage, data) {
  if (linkage === "ward") {
    if (!data) {
      throw new Error("Ward linkage requires access to the original data matrix.");
    }
    return wardDistance(cluster1, cluster2, data);
  }

  const dists = [];

  for (const i of cluster1) {
    for (const j of cluster2) {
      dists.push(distances[i][j]);
    }
  }

  if (linkage === "single") {
    return Math.min(...dists);
  } else if (linkage === "complete") {
    return Math.max(...dists);
  } else if (linkage === "average") {
    return mean(dists);
  }

  throw new Error(`Unknown linkage: ${linkage}`);
}

/**
 * Fit hierarchical clustering
 * @param {Array<Array<number>>} X - Data matrix
 * @param {Object} options - {linkage: 'single'|'complete'|'average'|'ward'}
 * @returns {Object} {dendrogram, distances}
 */
export function fit(X, { linkage = "average" } = {}) {
  let data;

  if (
    X && typeof X === "object" && !Array.isArray(X) && (X.data || X.columns)
  ) {
    const opts = X;
    const data_in = opts.data;
    const columns = opts.columns;
    const omit_missing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : true;

    if (!data_in) {
      throw new Error(
        "Declarative call requires a `data` property (array-of-objects or Arquero-like table)",
      );
    }

    const prepared = prepareX({ columns, data: data_in, omit_missing });
    data = prepared.X;

    linkage = opts.linkage !== undefined ? opts.linkage : linkage;
  } else if (Array.isArray(X)) {
    data = X.map((row) => Array.isArray(row) ? row : [row]);
  } else {
    throw new Error(
      "X must be an array of rows or an options object containing `data`/`columns`",
    );
  }

  const n = data.length;

  if (n < 2) {
    throw new Error("Need at least 2 samples for clustering");
  }

  const distances = computeDistanceMatrix(data);

  const clusters = data.map((_, i) => [i]);
  const merges = [];

  while (clusters.length > 1) {
    let minDist = Infinity;
    let minI = 0;
    let minJ = 1;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const dist = clusterDistance(
          clusters[i],
          clusters[j],
          distances,
          linkage,
          data,
        );
        if (dist < minDist) {
          minDist = dist;
          minI = i;
          minJ = j;
        }
      }
    }

    const merged = [...clusters[minI], ...clusters[minJ]];

    merges.push({
      cluster1: clusters[minI],
      cluster2: clusters[minJ],
      distance: minDist,
      size: merged.length,
    });

    const newClusters = [];
    for (let i = 0; i < clusters.length; i++) {
      if (i !== minI && i !== minJ) {
        newClusters.push(clusters[i]);
      }
    }
    newClusters.push(merged);
    clusters.splice(0, clusters.length, ...newClusters);
  }

  return {
    dendrogram: merges,
    linkage,
    n,
  };
}

export function cut(model, k) {
  const { dendrogram, n } = model;

  if (k < 1 || k > n) {
    throw new Error(`k must be between 1 and ${n}`);
  }

  if (k === n) {
    return Array.from({ length: n }, (_, i) => i);
  }

  const clusters = Array.from({ length: n }, (_, i) => [i]);

  for (let i = 0; i < dendrogram.length && clusters.length > k; i++) {
    const merge = dendrogram[i];
    const { cluster1, cluster2 } = merge;

    const idx1 = clusters.findIndex((cluster) =>
      cluster.length === cluster1.length &&
      cluster.every((value) => cluster1.includes(value))
    );
    const idx2 = clusters.findIndex((cluster) =>
      cluster.length === cluster2.length &&
      cluster.every((value) => cluster2.includes(value))
    );

    if (idx1 !== -1 && idx2 !== -1) {
      const merged = [...clusters[idx1], ...clusters[idx2]];
      clusters.splice(idx2, 1);
      clusters.splice(idx1, 1, merged);
    }
  }

  const labels = Array(n).fill(-1);
  clusters.forEach((cluster, idx) => {
    cluster.forEach((item) => {
      labels[item] = idx;
    });
  });

  return labels;
}

export function cutHeight(model, height) {
  const { dendrogram, n } = model;

  if (height < 0) {
    throw new Error("height must be non-negative");
  }

  const clusters = Array.from({ length: n }, (_, i) => [i]);

  for (let i = 0; i < dendrogram.length; i++) {
    const merge = dendrogram[i];
    const { cluster1, cluster2, distance } = merge;

    if (distance > height) {
      continue;
    }

    const idx1 = clusters.findIndex((cluster) =>
      cluster.length === cluster1.length &&
      cluster.every((value) => cluster1.includes(value))
    );
    const idx2 = clusters.findIndex((cluster) =>
      cluster.length === cluster2.length &&
      cluster.every((value) => cluster2.includes(value))
    );

    if (idx1 !== -1 && idx2 !== -1) {
      const merged = [...clusters[idx1], ...clusters[idx2]];
      clusters.splice(idx2, 1);
      clusters.splice(idx1, 1, merged);
    }
  }

  const labels = Array(n).fill(-1);
  clusters.forEach((cluster, idx) => {
    cluster.forEach((item) => {
      labels[item] = idx;
    });
  });

  return labels;
}

export default {
  fit,
  cut,
  cutHeight
};
