/**
 * Distance metrics for ML algorithms
 * Centralized distance functions used by KNN, DBSCAN, clustering, etc.
 */

/**
 * Euclidean distance (L2 norm)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Euclidean distance
 */
export function euclidean(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Manhattan distance (L1 norm, taxicab distance)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Manhattan distance
 */
export function manhattan(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

/**
 * Minkowski distance (generalized Lp norm)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @param {number} p - Order parameter (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)
 * @returns {number} Minkowski distance
 */
export function minkowski(a, b, p = 2) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  if (p === Infinity) {
    return chebyshev(a, b);
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(Math.abs(a[i] - b[i]), p);
  }
  return Math.pow(sum, 1 / p);
}

/**
 * Chebyshev distance (L∞ norm, maximum metric)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Chebyshev distance
 */
export function chebyshev(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) {
      max = diff;
    }
  }
  return max;
}

/**
 * Cosine distance (1 - cosine similarity)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Cosine distance (0 = identical direction, 2 = opposite)
 */
export function cosine(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 1; // Orthogonal
  }

  const cosineSimilarity = dotProduct / (normA * normB);
  return 1 - cosineSimilarity;
}

/**
 * Hamming distance (for categorical/binary data)
 * Counts the number of positions where elements differ
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {number} Hamming distance
 */
export function hamming(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  let count = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      count++;
    }
  }
  return count;
}

/**
 * Canberra distance (weighted version of Manhattan distance)
 * @param {Array<number>} a - First vector
 * @param {Array<number>} b - Second vector
 * @returns {number} Canberra distance
 */
export function canberra(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const absSum = Math.abs(a[i]) + Math.abs(b[i]);
    if (absSum > 0) {
      sum += Math.abs(a[i] - b[i]) / absSum;
    }
  }
  return sum;
}

/**
 * Gower distance for mixed-type data (numeric + categorical)
 * Handles missing values gracefully
 *
 * @param {Array} a - First vector (can contain numbers, strings, or null/undefined)
 * @param {Array} b - Second vector
 * @param {Object} options - Configuration
 * @param {Array<string>} options.types - Array indicating type of each feature: 'numeric' or 'categorical'
 * @param {Array<number>} options.ranges - Array of ranges for numeric features (max - min)
 * @returns {number} Gower distance (0 = identical, 1 = maximally different)
 */
export function gower(a, b, { types = null, ranges = null } = {}) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have same length");
  }

  const n = a.length;

  // Auto-detect types if not provided
  const featureTypes =
    types ||
    a.map((val, i) =>
      typeof val === "number" && typeof b[i] === "number"
        ? "numeric"
        : "categorical",
    );

  // Helper to check if value is missing
  const isMissing = (val) =>
    val === null ||
    val === undefined ||
    (typeof val === "number" && isNaN(val));

  let totalDistance = 0;
  let validCount = 0;

  for (let i = 0; i < n; i++) {
    const valA = a[i];
    const valB = b[i];

    // Skip if either value is missing
    if (isMissing(valA) || isMissing(valB)) {
      continue;
    }

    validCount++;

    if (featureTypes[i] === "numeric") {
      // Numeric feature: normalized absolute difference
      const range = ranges && ranges[i] ? ranges[i] : 1;
      const diff = Math.abs(valA - valB);
      totalDistance += range > 0 ? diff / range : 0;
    } else {
      // Categorical feature: 0 if same, 1 if different
      totalDistance += valA === valB ? 0 : 1;
    }
  }

  // Return average distance across valid (non-missing) features
  return validCount > 0 ? totalDistance / validCount : 0;
}

/**
 * Create a Gower distance function with pre-computed ranges
 * Useful for KNN when you want to compute ranges once from training data
 *
 * @param {Array<Array>} data - Training data to compute ranges from
 * @param {Array<string>} types - Feature types ('numeric' or 'categorical')
 * @returns {Function} Configured Gower distance function
 */
export function createGowerDistance(data, types = null) {
  const n = data.length;
  if (n === 0) return gower;

  const nFeatures = data[0].length;

  // Auto-detect types if not provided
  const featureTypes =
    types ||
    data[0].map((_, i) => {
      // Check if any non-missing value is non-numeric
      for (const row of data) {
        const val = row[i];
        if (val !== null && val !== undefined) {
          if (typeof val !== "number" || isNaN(val)) {
            return "categorical";
          }
        }
      }
      return "numeric";
    });

  // Compute ranges for numeric features
  const ranges = new Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) {
    if (featureTypes[i] === "numeric") {
      let min = Infinity;
      let max = -Infinity;

      for (const row of data) {
        const val = row[i];
        if (val !== null && val !== undefined && !isNaN(val)) {
          if (val < min) min = val;
          if (val > max) max = val;
        }
      }

      ranges[i] = max > min ? max - min : 1;
    } else {
      ranges[i] = 1; // Not used for categorical
    }
  }

  // Return a function with pre-computed configuration
  return (a, b) => gower(a, b, { types: featureTypes, ranges });
}

/**
 * Get distance function by name
 * @param {string|Function} metric - Metric name or custom function
 * @returns {Function} Distance function
 */
export function getDistanceFunction(metric) {
  if (typeof metric === "function") {
    return metric;
  }

  const metrics = {
    euclidean,
    manhattan,
    minkowski,
    chebyshev,
    cosine,
    hamming,
    canberra,
    gower,
  };

  if (!metrics[metric]) {
    throw new Error(
      `Unknown distance metric: ${metric}. Available: ${Object.keys(metrics).join(", ")}`,
    );
  }

  return metrics[metric];
}

// Default export
export default {
  euclidean,
  manhattan,
  minkowski,
  chebyshev,
  cosine,
  hamming,
  canberra,
  gower,
  createGowerDistance,
  getDistanceFunction,
};
