/**
 * Core mathematical utilities and constants
 */

// Mathematical constants
export const EPSILON = 1e-10;
export const PI = Math.PI;
export const E = Math.E;

/**
 * Approximate equality comparison for floating point numbers
 * @param {number} a - First number
 * @param {number} b - Second number
 * @param {number} tolerance - Tolerance for comparison
 * @returns {boolean} True if approximately equal
 */
export function approxEqual(a, b, tolerance = EPSILON) {
  return Math.abs(a - b) < tolerance;
}

/**
 * Guard against non-finite values
 * @param {number} value - Value to check
 * @param {string} name - Name for error message
 * @returns {number} The value if valid
 * @throws {Error} If value is not finite
 */
export function guardFinite(value, name = 'value') {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
  return value;
}

/**
 * Guard against negative values
 * @param {number} value - Value to check
 * @param {string} name - Name for error message
 * @returns {number} The value if valid
 * @throws {Error} If value is negative
 */
export function guardPositive(value, name = 'value') {
  if (value <= 0) {
    throw new Error(`${name} must be positive, got ${value}`);
  }
  return value;
}

/**
 * Guard against values outside [0, 1]
 * @param {number} value - Value to check
 * @param {string} name - Name for error message
 * @returns {number} The value if valid
 * @throws {Error} If value is outside [0, 1]
 */
export function guardProbability(value, name = 'value') {
  if (value < 0 || value > 1) {
    throw new Error(`${name} must be between 0 and 1, got ${value}`);
  }
  return value;
}

/**
 * Sum of array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Sum
 */
export function sum(arr, options = {}) {
  if (!Array.isArray(arr)) {
    throw new Error('Expected an array of numbers');
  }
  if (!options || Object.keys(options).length === 0) {
    return arr.reduce((a, b) => a + b, 0);
  }
  const data = sanitizeNumericArray(arr, options);
  return data.reduce((a, b) => a + b, 0);
}

/**
 * Mean of array
 * @param {number[]} arr - Array of numbers
 * @param {Object} [options] - { naOmit?: boolean }
 * @returns {number} Mean
 */
export function mean(arr, options = {}) {
  if (!Array.isArray(arr)) {
    throw new Error('Expected an array of numbers');
  }
  if (!options || Object.keys(options).length === 0) {
    if (arr.length === 0) return NaN;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }
  const data = sanitizeNumericArray(arr, options);
  return meanFromSanitized(data);
}

/**
 * Variance of array
 * @param {number[]} arr - Array of numbers
 * @param {boolean} sample - If true, use sample variance (n-1)
 * @param {Object} [options] - { naOmit?: boolean }
 * @returns {number} Variance
 */
export function variance(arr, sample = true, options = {}) {
  if (!Array.isArray(arr)) {
    throw new Error('Expected an array of numbers');
  }

  let opts = options;
  let useSample = sample;
  if (typeof sample === 'object' && sample !== null) {
    opts = sample;
    useSample = sample.sample !== undefined ? !!sample.sample : true;
  }

  const needsSanitizing = opts && Object.keys(opts).length > 0;
  const data = needsSanitizing ? sanitizeNumericArray(arr, opts) : arr;
  if (!data.length) {
    return NaN;
  }
  const m = meanFromSanitized(data);
  if (!Number.isFinite(m)) {
    return NaN;
  }
  const squaredDiffs = data.map(x => (x - m) ** 2);
  const divisor = useSample ? data.length - 1 : data.length;
  if (divisor <= 0) {
    return NaN;
  }
  const total = squaredDiffs.reduce((a, b) => a + b, 0);
  return total / divisor;
}

/**
 * Standard deviation of array
 * @param {number[]} arr - Array of numbers
 * @param {boolean} sample - If true, use sample variance (n-1)
 * @param {Object} [options] - { naOmit?: boolean }
 * @returns {number} Standard deviation
 */
export function stddev(arr, sample = true, options = {}) {
  return Math.sqrt(variance(arr, sample, options));
}

function sanitizeNumericArray(arr, options = {}) {
  if (!Array.isArray(arr)) {
    throw new Error('Expected an array of numbers');
  }
  const { naOmit = false } = options;
  const result = [];
  arr.forEach((value, index) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      if (!naOmit) {
        throw new Error(`Non-numeric value at index ${index}: ${value}`);
      }
      return;
    }
    const num = Number(value);
    if (!Number.isFinite(num)) {
      if (!naOmit) {
        throw new Error(`Non-finite value at index ${index}: ${value}`);
      }
      return;
    }
    result.push(num);
  });
  return result;
}

function meanFromSanitized(arr) {
  if (!arr.length) {
    return NaN;
  }
  const total = arr.reduce((sum, val) => sum + val, 0);
  return total / arr.length;
}

export function quantile(arr, p, options = {}) {
  if (!Array.isArray(arr)) {
    throw new Error('Expected an array of numbers');
  }
  const { naOmit = false, method = 'linear' } = options;
  const data = naOmit ? sanitizeNumericArray(arr, { naOmit }) : arr.filter((v) => Number.isFinite(v));
  if (!naOmit && data.length !== arr.length) {
    throw new Error('Encountered non-finite values. Pass { naOmit: true } to ignore them.');
  }
  if (!data.length) return NaN;
  const sorted = data.slice().sort((a, b) => a - b);
  if (p <= 0) return sorted[0];
  if (p >= 1) return sorted[sorted.length - 1];
  const index = (sorted.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  if (method === 'linear') {
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }
  return sorted[weight > 0.5 ? upper : lower];
}

export function median(arr, options = {}) {
  return quantile(arr, 0.5, options);
}

export function percentile(arr, value, options = {}) {
  const data = sanitizeNumericArray(arr, options);
  if (!data.length) return NaN;
  const sorted = data.slice().sort((a, b) => a - b);
  let count = 0;
  for (const v of sorted) {
    if (v <= value) count += 1;
  }
  return count / sorted.length;
}

export function summaryQuantiles(arr, probs = [0, 0.25, 0.5, 0.75, 1], options = {}) {
  const out = {};
  for (const p of probs) {
    out[p] = quantile(arr, p, options);
  }
  return out;
}
