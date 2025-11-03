/**
 * Centralized scaling utilities for multivariate analysis (PCA, LDA, RDA)
 *
 * The scaling parameter controls how scores and loadings are scaled:
 * - scaling = 0: No additional scaling (default)
 * - scaling = 1: Scores divided by sqrt(n-1)
 * - scaling = 2: Correlation-based scaling (scores are U matrix, loadings multiplied by sqrt(eigenvalues))
 */

const EPSILON = 1e-10;

/**
 * Apply scaling transformation to scores
 *
 * @param {Array<Array<number>>} base - Base scores matrix
 * @param {Array<Array<number>>|null} u - U matrix from SVD (optional, used for scaling=2)
 * @param {Array<number>} singularValues - Singular values from SVD
 * @param {number} scaling - Scaling type (0, 1, or 2)
 * @param {number} sqrtNSamples - Square root of (n-1)
 * @returns {Array<Array<number>>} Scaled scores
 */
export function applyScalingToScores({
  base,
  u = null,
  singularValues = [],
  scaling = 0,
  sqrtNSamples = 1
}) {
  if (!Array.isArray(base) || base.length === 0) {
    return [];
  }

  const nComponents = base[0]?.length || 0;

  switch (scaling) {
    case 1: {
      // Divide by sqrt(n-1)
      const factor = sqrtNSamples > 0 ? 1 / sqrtNSamples : 1;
      return base.map(row => row.map(val => val * factor));
    }
    case 2: {
      // Correlation-based: use U matrix if available, otherwise divide by singular values
      if (Array.isArray(u) && u.length) {
        return u.map(row => row.slice(0, nComponents));
      }
      return base.map(row =>
        row.map((val, idx) => {
          const sv = singularValues[idx] ?? 1;
          return sv > EPSILON ? val / sv : 0;
        })
      );
    }
    default:
      // scaling = 0: no scaling
      return base;
  }
}

/**
 * Apply scaling transformation to loadings
 *
 * @param {Array<Array<number>>} components - Component vectors (columns)
 * @param {Array<number>} sqrtEigenvalues - Square roots of eigenvalues
 * @param {number} scaling - Scaling type (0, 1, or 2)
 * @param {Array<string>|null} featureNames - Feature/variable names
 * @returns {Object} { matrix: Array<Array<number>>, variableNames: Array<string> }
 */
export function applyScalingToLoadings({
  components,
  sqrtEigenvalues,
  scaling = 0,
  featureNames = null
}) {
  const nComponents = components.length;
  const p = components[0]?.length || 0;
  const variableNames = Array.isArray(featureNames) && featureNames.length === p
    ? featureNames.slice()
    : Array.from({ length: p }, (_, i) => `var${i + 1}`);

  const scaledColumns = components.map((col, j) => {
    // For scaling=2, multiply by sqrt(eigenvalue) to get correlation-based loadings
    const factor = scaling === 2 ? (sqrtEigenvalues[j] ?? 1) : 1;
    return col.map((val) => val * factor);
  });

  const matrix = columnsToRows(scaledColumns);
  return { matrix, variableNames };
}

/**
 * Convert column-oriented matrix to row-oriented matrix
 *
 * @param {Array<Array<number>>} columns - Array of column vectors
 * @returns {Array<Array<number>>} Row-oriented matrix
 */
function columnsToRows(columns) {
  if (!columns.length) return [];
  const rows = columns[0].length;
  const cols = columns.length;
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(columns[j][i]);
    }
    matrix.push(row);
  }
  return matrix;
}

/**
 * Convert scores matrix to array of objects with named components
 *
 * @param {Array<Array<number>>} matrix - Scores matrix
 * @param {string} prefix - Prefix for component names (e.g., 'pc', 'ld', 'rda')
 * @returns {Array<Object>} Array of score objects
 */
export function toScoreObjects(matrix, prefix) {
  return matrix.map((row) => {
    const entry = {};
    row.forEach((val, idx) => {
      entry[`${prefix}${idx + 1}`] = val;
    });
    return entry;
  });
}

/**
 * Convert loadings matrix to array of objects with named components and variables
 *
 * @param {Array<Array<number>>} matrix - Loadings matrix (rows = variables)
 * @param {Array<string>} variableNames - Variable names
 * @param {string} prefix - Prefix for component names (e.g., 'pc', 'ld', 'rda')
 * @returns {Array<Object>} Array of loading objects
 */
export function toLoadingObjects(matrix, variableNames, prefix) {
  return matrix.map((row, i) => {
    const entry = { variable: variableNames[i] || `var${i + 1}` };
    row.forEach((val, idx) => {
      entry[`${prefix}${idx + 1}`] = val;
    });
    return entry;
  });
}
