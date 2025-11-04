/**
 * Ordination scaling helpers following ter Braak / vegan conventions.
 *
 * Given an SVD of a centred (and optionally standardised) data matrix
 *   X = U * Lambda^{1/2} * V^T
 * where Lambda contains the eigenvalues of X^T X / (n - 1),
 * the site (sample) and variable (loading) scores obey:
 *
 *   site scores     = U * Lambda^{a}
 *   variable scores = V * Lambda^{1 - a}
 *
 * Scaling choices:
 *   scaling = 1  => a = 0.5  (distance-focused biplot)
 *   scaling = 2  => a = 0    (correlation-focused biplot)
 *
 * We expose helpers that convert raw orthonormal coordinates (U, V)
 * into scaled scores by applying the appropriate eigenvalue exponents.
 */

const SCALING_EXPONENT = {
  1: 0.5,
  2: 0,
};

/**
 * Sanitize scaling option (only 1 or 2 are supported).
 * Defaults to 2 (correlation biplot) when unspecified.
 */
export function normalizeScaling(value) {
  if (value === 1 || value === '1') return 1;
  return 2;
}

/**
 * Compute Lambda^power given eigenvalues.
 *
 * @param {number[]} eigenvalues - covariance eigenvalues (>= 0)
 * @param {number} power - exponent to apply
 * @returns {number[]} element-wise eigenvalue^power
 */
export function eigenvaluePowers(eigenvalues, power) {
  return eigenvalues.map((lambda) => {
    if (lambda <= 0) return 0;
    if (power === 0) return 1;
    if (power === 0.5) return Math.sqrt(lambda);
    if (power === 1) return lambda;
    return Math.pow(lambda, power);
  });
}

/**
 * Apply ordination scaling to site/variable coordinates.
 *
 * @param {Object} params
 * @param {Array<Array<number>>} params.rawSites
 * @param {Array<Array<number>>} params.rawLoadings
 * @param {number[]} params.eigenvalues
 * @param {number} params.scaling - 1 or 2
 * @returns {Object} { scores, loadings, siteFactors, loadingFactors, exponent }
 */
export function scaleOrdination({
  rawSites,
  rawLoadings,
  eigenvalues = [],
  singularValues = null,
  scaling = 2,
}) {
  const normalizedScaling = normalizeScaling(scaling);
  const componentCount = rawLoadings?.[0]?.length ?? eigenvalues.length ?? 0;

  const siteFactors = Array(componentCount).fill(1);
  const loadingFactors = Array(componentCount).fill(1);

  if (normalizedScaling === 1) {
    if (Array.isArray(singularValues) && singularValues.length) {
      for (let i = 0; i < componentCount; i++) {
        siteFactors[i] = singularValues[i] ?? 0;
      }
    } else {
      const fallback = eigenvaluePowers(eigenvalues, 0.5);
      for (let i = 0; i < componentCount; i++) {
        siteFactors[i] = fallback[i] ?? 0;
      }
    }
    // Loadings remain unscaled (geometric distances preserved)
  } else {
    const sqrtEigen = eigenvaluePowers(eigenvalues, 0.5);
    for (let i = 0; i < componentCount; i++) {
      loadingFactors[i] = sqrtEigen[i] ?? 0;
    }
  }

  const scores = rawSites.map((row) =>
    row.map((val, idx) => val * (siteFactors[idx] ?? 0))
  );
  const loadings = rawLoadings.map((row) =>
    row.map((val, idx) => val * (loadingFactors[idx] ?? 0))
  );

  return {
    scores,
    loadings,
    siteFactors,
    loadingFactors,
    exponent: SCALING_EXPONENT[normalizedScaling],
    scaling: normalizedScaling,
  };
}

/**
 * Apply ordination scaling to predictor/constraint (biplot) scores.
 * They follow the same exponent as variable scores (1 - a).
 *
 * @param {Array<Array<number>>} rawConstraints
 * @param {Object} options
 * @param {Array<number>} [options.loadingFactors]
 * @param {Array<number>} [options.eigenvalues]
 * @param {number} [options.scaling]
 * @returns {Array<Array<number>>}
 */
export function scaleConstraintScores(
  rawConstraints,
  { loadingFactors = null, eigenvalues = [], scaling = 2 } = {}
) {
  const normalizedScaling = normalizeScaling(scaling);
  let factors = Array.isArray(loadingFactors) && loadingFactors.length
    ? loadingFactors
    : null;

  if (!factors) {
    const fallbackExponent = normalizedScaling === 1 ? 0 : 0.5;
    factors = eigenvaluePowers(eigenvalues, fallbackExponent);
  }

  return rawConstraints.map((row) =>
    row.map((val, idx) => val * factors[idx])
  );
}

/**
 * Convert matrix of scores into array-of-objects with component names.
 *
 * @param {Array<Array<number>>} matrix - rows correspond to observations
 * @param {string} prefix - component prefix (e.g., 'pc', 'ld', 'rda')
 * @param {Object} [extraPerRow] - optional extra properties per row index
 * @returns {Array<Object>}
 */
export function toScoreObjects(matrix, prefix, extraPerRow = null) {
  return matrix.map((row, idx) => {
    const entry = {};
    if (extraPerRow && typeof extraPerRow === 'function') {
      Object.assign(entry, extraPerRow(idx));
    } else if (extraPerRow && typeof extraPerRow === 'object') {
      Object.assign(entry, extraPerRow);
    }
    row.forEach((val, compIdx) => {
      entry[`${prefix}${compIdx + 1}`] = val;
    });
    return entry;
  });
}

/**
 * Convert matrix of loadings (rows = variables) to annotated objects.
 *
 * @param {Array<Array<number>>} matrix - row-major loadings
 * @param {Array<string>} variableNames - names per row
 * @param {string} prefix - component prefix (e.g., 'pc', 'ld', 'rda')
 * @returns {Array<Object>}
 */
export function toLoadingObjects(matrix, variableNames, prefix) {
  return matrix.map((row, idx) => {
    const entry = { variable: variableNames[idx] || `var${idx + 1}` };
    row.forEach((val, compIdx) => {
      entry[`${prefix}${compIdx + 1}`] = val;
    });
    return entry;
  });
}

/**
 * Utility to transpose column-major component array into row-major matrix.
 *
 * @param {Array<Array<number>>} columns - component vectors (column-major)
 * @returns {Array<Array<number>>} row-major matrix
 */
export function columnsToRows(columns) {
  if (!columns || !columns.length) return [];
  const rowCount = columns[0].length;
  const colCount = columns.length;
  const result = Array.from({ length: rowCount }, () => Array(colCount).fill(0));
  for (let j = 0; j < colCount; j++) {
    for (let i = 0; i < rowCount; i++) {
      result[i][j] = columns[j][i];
    }
  }
  return result;
}
