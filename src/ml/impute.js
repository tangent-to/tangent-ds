/**
 * Missing Data Imputation
 *
 * Provides sklearn-compatible imputers for handling missing values:
 * - SimpleImputer: Fill with mean, median, mode, or constant
 * - KNNImputer: Fill using k-nearest neighbors
 * - IterativeImputer: Multivariate imputation (MICE algorithm)
 */

import { mean as calculateMean } from "../core/math.js";
import { normalize, applyColumns } from "../core/table.js";
import { Matrix, pseudoInverse } from "ml-matrix";

/**
 * Helper: Check if value is missing (NaN, null, undefined)
 */
function isMissing(value) {
  return (
    value === null ||
    value === undefined ||
    (typeof value === "number" && isNaN(value))
  );
}

/**
 * Helper: Convert {data, columns} format to 2D array, preserving missing values
 * @param {Object} options - {data, columns}
 * @returns {Object} {X: Array<Array<number>>, columns: Array<string>}
 */
function tableToMatrix({ data, columns }) {
  const rows = normalize(data);

  if (rows.length === 0) {
    throw new Error("Cannot prepare matrix from empty data");
  }

  // If no columns specified, auto-detect numeric columns
  let selectedColumns = columns;
  if (!selectedColumns) {
    const firstRow = rows[0];
    selectedColumns = Object.keys(firstRow).filter((key) => {
      const val = firstRow[key];
      // Include column if it's numeric or missing (we'll handle missing values during imputation)
      return typeof val === "number" || isMissing(val);
    });
  } else if (typeof selectedColumns === "string") {
    selectedColumns = [selectedColumns];
  }

  // Convert to 2D array, allowing missing values
  const X = rows.map((row) =>
    selectedColumns.map((col) => {
      const val = row[col];
      // Convert missing values to NaN for consistency
      return isMissing(val) ? NaN : val;
    }),
  );

  return { X, columns: selectedColumns };
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
  return arr.filter((val) => !isMissing(val));
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
  constructor({ strategy = "mean", fill_value = null, copy = true } = {}) {
    const validStrategies = ["mean", "median", "most_frequent", "constant"];
    if (!validStrategies.includes(strategy)) {
      throw new Error(
        `Invalid strategy: ${strategy}. Must be one of ${validStrategies.join(", ")}`,
      );
    }

    if (strategy === "constant" && fill_value === null) {
      throw new Error(
        'fill_value must be provided when strategy is "constant"',
      );
    }

    this.strategy = strategy;
    this.fill_value = fill_value;
    this.copy = copy;
    this.statistics_ = null;
    this.nFeatures_ = null;
    this._tableColumns = null;
    this._groupModels = null;
    this._originalData = null;
  }

  /**
   * Fit the imputer on training data
   * @param {Array<Array<number>>|Object} X - Training data, table object, or {data, columns, group} format
   * @returns {SimpleImputer} this
   */
  fit(X) {
    // Handle table input: either {data, columns, group} format or array of objects
    if (X && typeof X === "object" && !Array.isArray(X)) {
      // Check if it's {data, columns} format or array of objects
      if (X.data || X.columns) {
        const groupBy = X.group;
        this._originalData = X.data;

        // Group-based imputation
        if (groupBy) {
          this._groupModels = new Map();
          const rows = normalize(X.data);
          const groups = new Map();

          // Group rows
          rows.forEach((row, idx) => {
            const groupKey = row[groupBy];
            if (!groups.has(groupKey)) {
              groups.set(groupKey, []);
            }
            groups.get(groupKey).push({ row, idx });
          });

          // Fit a model for each group
          for (const [groupKey, groupData] of groups) {
            const groupRows = groupData.map((d) => d.row);
            const result = tableToMatrix({
              data: groupRows,
              columns: X.columns,
            });

            if (result.X.length > 0) {
              const model = this._fitSingleModel(result.X);
              this._groupModels.set(groupKey, {
                model,
                indices: groupData.map((d) => d.idx),
                columns: result.columns,
              });
            }
          }

          this._tableColumns = X.columns;
          return this;
        }

        // Non-grouped imputation
        const result = tableToMatrix({ data: X.data, columns: X.columns });
        this._tableColumns = result.columns;
        X = result.X;
      } else {
        // Assume it's an array of objects (not wrapped in {data, columns})
        throw new Error(
          "X must be a 2D array, {data, columns} object, or array of objects",
        );
      }
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error("X must be a 2D array or table object");
    }

    this._fitSingleModel(X);
    return this;
  }

  /**
   * Internal method to fit a single imputation model
   * @param {Array<Array<number>>} X - 2D array of numeric data
   * @returns {Object} Model statistics
   */
  _fitSingleModel(X) {
    const nSamples = X.length;
    const nFeatures = X[0].length;
    this.nFeatures_ = nFeatures;

    // Compute statistics for each feature
    const statistics = new Array(nFeatures);

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
        statistics[j] = this.strategy === "constant" ? this.fill_value : 0;
        continue;
      }

      // Compute statistic based on strategy
      switch (this.strategy) {
        case "mean":
          statistics[j] = calculateMean(column);
          break;
        case "median":
          statistics[j] = calculateMedian(column);
          break;
        case "most_frequent":
          statistics[j] = calculateMode(column);
          break;
        case "constant":
          statistics[j] = this.fill_value;
          break;
      }
    }

    this.statistics_ = statistics;
    return { statistics, nFeatures };
  }

  /**
   * Transform data by filling missing values
   * @param {Array<Array<number>>|Object} X - Data to transform, table object, or {data, columns, group} format
   * @returns {Array<Array<number>>|Array<Object>} Transformed data (array if input was table)
   */
  transform(X) {
    if (this.statistics_ === null && !this._groupModels) {
      throw new Error("Imputer must be fitted before transform");
    }

    // Handle table input with groups
    if (X && typeof X === "object" && !Array.isArray(X)) {
      if (X.data || X.columns) {
        const groupBy = X.group;
        const rows = normalize(X.data);

        // Group-based imputation
        if (groupBy && this._groupModels) {
          const imputedRows = rows.map((row) => ({ ...row }));

          for (const [groupKey, groupInfo] of this._groupModels) {
            const groupIndices = [];
            const groupRows = [];

            rows.forEach((row, idx) => {
              if (row[groupBy] === groupKey) {
                groupIndices.push(idx);
                groupRows.push(row);
              }
            });

            if (groupRows.length === 0) continue;

            const result = tableToMatrix({
              data: groupRows,
              columns: groupInfo.columns,
            });

            const imputed = this._transformWithModel(
              result.X,
              groupInfo.model.statistics,
            );

            // Apply back to original rows
            groupIndices.forEach((originalIdx, i) => {
              groupInfo.columns.forEach((col, colIdx) => {
                imputedRows[originalIdx][col] = imputed[i][colIdx];
              });
            });
          }

          return imputedRows;
        }

        // Non-grouped table imputation
        const result = tableToMatrix({
          data: X.data,
          columns: X.columns || this._tableColumns,
        });

        const imputed = this._transformWithModel(result.X, this.statistics_);
        return applyColumns(rows, result.columns, imputed);
      } else {
        throw new Error(
          "X must be a 2D array, {data, columns} object, or array of objects",
        );
      }
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error("X must be a 2D array or table object");
    }

    if (X[0].length !== this.nFeatures_) {
      throw new Error(
        `X has ${X[0].length} features, but imputer expected ${this.nFeatures_}`,
      );
    }

    return this._transformWithModel(X, this.statistics_);
  }

  /**
   * Internal method to transform data with specific statistics
   * @param {Array<Array<number>>} X - Data to transform
   * @param {Array<number>} statistics - Statistics to use for imputation
   * @returns {Array<Array<number>>} Transformed data
   */
  _transformWithModel(X, statistics) {
    // Create copy if requested
    const result = this.copy ? X.map((row) => [...row]) : X;

    // Fill missing values
    for (let i = 0; i < result.length; i++) {
      for (let j = 0; j < statistics.length; j++) {
        if (isMissing(result[i][j])) {
          result[i][j] = statistics[j];
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
  constructor({
    n_neighbors = 5,
    weights = "uniform",
    metric = null,
    copy = true,
  } = {}) {
    if (n_neighbors <= 0) {
      throw new Error("n_neighbors must be positive");
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
   * @param {Array<Array<number>>|Object} X - Training data, table object, or {data, columns} format
   * @returns {KNNImputer} this
   */
  fit(X) {
    // Handle table input: either {data, columns} format or array of objects
    if (X && typeof X === "object" && !Array.isArray(X)) {
      if (X.data || X.columns) {
        const result = tableToMatrix({ data: X.data, columns: X.columns });
        this._tableColumns = result.columns;
        X = result.X;
      } else {
        throw new Error(
          "X must be a 2D array, {data, columns} object, or array of objects",
        );
      }
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error("X must be a 2D array or table object");
    }

    // Store training data
    this.X_ = this.copy ? X.map((row) => [...row]) : X;
    this.nFeatures_ = X[0].length;

    return this;
  }

  /**
   * Transform data by filling missing values using KNN
   * @param {Array<Array<number>>|Object} X - Data to transform, table object, or {data, columns} format
   * @param {Array<number>} exclude_indices - Row indices to exclude from neighbors (for fit_transform)
   * @returns {Array<Array<number>>|Array<Object>} Transformed data (array if input was table)
   */
  transform(X, exclude_indices = []) {
    if (this.X_ === null) {
      throw new Error("Imputer must be fitted before transform");
    }

    // Handle table input
    let originalData = null;
    let selectedColumns = null;
    if (X && typeof X === "object" && !Array.isArray(X)) {
      if (X.data || X.columns) {
        originalData = X.data;
        const result = tableToMatrix({
          data: X.data,
          columns: X.columns || this._tableColumns,
        });
        selectedColumns = result.columns;
        X = result.X;
      } else {
        throw new Error(
          "X must be a 2D array, {data, columns} object, or array of objects",
        );
      }
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error("X must be a 2D array or table object");
    }

    if (X[0].length !== this.nFeatures_) {
      throw new Error(
        `X has ${X[0].length} features, but imputer expected ${this.nFeatures_}`,
      );
    }

    // Create copy
    const result = this.copy ? X.map((row) => [...row]) : X;

    // For each row with missing values
    for (let i = 0; i < result.length; i++) {
      const row = result[i];
      const missingIndices = getMissingIndices(row);

      if (missingIndices.length === 0) {
        continue; // No missing values
      }

      // Find k nearest neighbors from training data (excluding this row if in exclude_indices)
      const excludeSet = new Set(exclude_indices);
      const neighbors = this._findNeighbors(
        row,
        this.n_neighbors,
        excludeSet.has(i) ? i : -1,
      );

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
          if (this.weights === "distance") {
            const weightSum = weights.reduce((a, b) => a + b, 0);
            const weightedSum = values.reduce(
              (sum, val, idx) => sum + val * weights[idx],
              0,
            );
            result[i][j] = weightedSum / weightSum;
          } else {
            result[i][j] = calculateMean(values);
          }
        }
      }
    }

    // If input was table format, return full table with imputed values
    if (originalData !== null) {
      return applyColumns(normalize(originalData), selectedColumns, result);
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
   * @returns {Array<Array<number>>|Array<Object>} Transformed data
   */
  fit_transform(X) {
    // Store original input to preserve table format
    const originalInput = X;
    this.fit(X);
    // Pass all indices to exclude each row from its own neighbor search
    const excludeIndices = Array.from({ length: this.X_.length }, (_, i) => i);

    // If input was table format, pass it through to transform
    if (
      originalInput &&
      typeof originalInput === "object" &&
      !Array.isArray(originalInput) &&
      (originalInput.data || originalInput.columns)
    ) {
      // Create a modified input for transform that uses the stored data but preserves format
      return this.transform(originalInput, excludeIndices);
    }

    // Array format - use stored training data
    return this.transform(this.X_, excludeIndices);
  }
}

// ============= IterativeImputer =============

/**
 * Multivariate imputation using chained equations (MICE algorithm)
 * Compatible with sklearn.impute.IterativeImputer
 *
 * Models each feature with missing values as a function of other features,
 * and uses that estimate for imputation. It does so in an iterated round-robin
 * fashion: at each step, a feature column is designated as output y and the other
 * feature columns are treated as inputs X. A regressor is fit on (X, y) for known
 * values and used to predict missing values of y.
 *
 * @example
 * const imputer = new IterativeImputer({ max_iter: 10 });
 * imputer.fit(X_train);
 * const X_filled = imputer.transform(X_test);
 */
export class IterativeImputer {
  /**
   * @param {Object} options
   * @param {string} options.initial_strategy - Initial imputation strategy (default: 'mean')
   * @param {number} options.max_iter - Maximum number of imputation rounds (default: 10)
   * @param {number} options.tol - Tolerance for convergence (default: 1e-3)
   * @param {number} options.min_value - Minimum possible imputed value (default: -Infinity)
   * @param {number} options.max_value - Maximum possible imputed value (default: Infinity)
   * @param {boolean} options.verbose - Print progress (default: false)
   * @param {boolean} options.copy - If true, create copy of X (default: true)
   */
  constructor({
    initial_strategy = "mean",
    max_iter = 10,
    tol = 1e-3,
    min_value = -Infinity,
    max_value = Infinity,
    verbose = false,
    copy = true,
  } = {}) {
    this.initial_strategy = initial_strategy;
    this.max_iter = max_iter;
    this.tol = tol;
    this.min_value = min_value;
    this.max_value = max_value;
    this.verbose = verbose;
    this.copy = copy;
    this.nFeatures_ = null;
    this.initial_imputer_ = null;
    this._tableColumns = null;
    this.n_iter_ = null;
  }

  /**
   * Fit a simple linear regression using pseudoinverse
   * @param {Array<Array<number>>} X - Features
   * @param {Array<number>} y - Target
   * @returns {Object} Model with coefficients and predict function
   */
  _fitLinearRegression(X, y) {
    // Add intercept column
    const n = X.length;
    const Xmat = new Matrix(X);
    const ones = Matrix.ones(n, 1);
    const XwithIntercept = Matrix.columnMatrix(
      ones.getColumn(0),
      ...Xmat.transpose().to2DArray(),
    );

    const yMat = new Matrix([y]).transpose();

    // Use pseudoinverse for robust estimation: β = (X'X)^(-1) X'y = pinv(X) * y
    const XpInv = pseudoInverse(XwithIntercept);
    const beta = XpInv.mmul(yMat);
    const coef = beta.to2DArray().map((row) => row[0]);

    return {
      intercept: coef[0],
      coefficients: coef.slice(1),
      predict: (Xnew) => {
        const predictions = [];
        for (let i = 0; i < Xnew.length; i++) {
          let pred = coef[0]; // intercept
          for (let j = 0; j < Xnew[i].length; j++) {
            pred += coef[j + 1] * Xnew[i][j];
          }
          predictions.push(pred);
        }
        return predictions;
      },
    };
  }

  /**
   * Impute a single feature using other features
   * @param {Array<Array<number>>} X - Data matrix
   * @param {number} featureIdx - Index of feature to impute
   * @returns {Array<number>} Imputed values for this feature
   */
  _imputeFeature(X, featureIdx) {
    const n = X.length;
    const nFeatures = X[0].length;

    // Separate rows with and without missing values for this feature
    const knownRows = [];
    const missingRows = [];
    const knownY = [];

    for (let i = 0; i < n; i++) {
      if (!isMissing(X[i][featureIdx])) {
        knownRows.push(i);
        knownY.push(X[i][featureIdx]);
      } else {
        missingRows.push(i);
      }
    }

    // If no missing values, return as-is
    if (missingRows.length === 0) {
      return X.map((row) => row[featureIdx]);
    }

    // If all values missing, use initial strategy
    if (knownRows.length === 0) {
      const fillValue = this.initial_imputer_.statistics_[featureIdx];
      return new Array(n).fill(fillValue);
    }

    // Build training data (other features)
    const X_train = [];
    for (const idx of knownRows) {
      const row = [];
      for (let j = 0; j < nFeatures; j++) {
        if (j !== featureIdx) {
          row.push(X[idx][j]);
        }
      }
      X_train.push(row);
    }

    // Fit regression model
    const model = this._fitLinearRegression(X_train, knownY);

    // Predict missing values
    const X_missing = [];
    for (const idx of missingRows) {
      const row = [];
      for (let j = 0; j < nFeatures; j++) {
        if (j !== featureIdx) {
          row.push(X[idx][j]);
        }
      }
      X_missing.push(row);
    }

    const predictions = model.predict(X_missing);

    // Combine known and predicted values
    const result = new Array(n);
    for (let i = 0; i < knownRows.length; i++) {
      result[knownRows[i]] = X[knownRows[i]][featureIdx];
    }
    for (let i = 0; i < missingRows.length; i++) {
      // Clip to min/max
      let pred = predictions[i];
      pred = Math.max(this.min_value, Math.min(this.max_value, pred));
      result[missingRows[i]] = pred;
    }

    return result;
  }

  /**
   * Fit the imputer on training data
   * @param {Array<Array<number>>|Object} X - Training data, table object, or {data, columns} format
   * @returns {IterativeImputer} this
   */
  fit(X) {
    // Handle table input: either {data, columns} format or array of objects
    if (X && typeof X === "object" && !Array.isArray(X)) {
      if (X.data || X.columns) {
        const result = tableToMatrix({ data: X.data, columns: X.columns });
        this._tableColumns = result.columns;
        X = result.X;
      } else {
        throw new Error(
          "X must be a 2D array, {data, columns} object, or array of objects",
        );
      }
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error("X must be a 2D array or table object");
    }

    this.nFeatures_ = X[0].length;

    // Initial imputation using simple strategy
    this.initial_imputer_ = new SimpleImputer({
      strategy: this.initial_strategy,
    });
    this.initial_imputer_.fit(X);

    return this;
  }

  /**
   * Transform data by filling missing values using MICE
   * @param {Array<Array<number>>|Object} X - Data to transform, table object, or {data, columns} format
   * @returns {Array<Array<number>>|Array<Object>} Transformed data (array if input was table)
   */
  transform(X) {
    if (this.initial_imputer_ === null) {
      throw new Error("Imputer must be fitted before transform");
    }

    // Handle table input
    let originalData = null;
    let selectedColumns = null;
    if (X && typeof X === "object" && !Array.isArray(X)) {
      if (X.data || X.columns) {
        originalData = X.data;
        const result = tableToMatrix({
          data: X.data,
          columns: X.columns || this._tableColumns,
        });
        selectedColumns = result.columns;
        X = result.X;
      } else {
        throw new Error(
          "X must be a 2D array, {data, columns} object, or array of objects",
        );
      }
    }

    if (!Array.isArray(X) || !Array.isArray(X[0])) {
      throw new Error("X must be a 2D array or table object");
    }

    if (X[0].length !== this.nFeatures_) {
      throw new Error(
        `X has ${X[0].length} features, but imputer expected ${this.nFeatures_}`,
      );
    }

    const n = X.length;
    const nFeatures = this.nFeatures_;

    // Initial imputation
    let X_filled = this.initial_imputer_.transform(X);

    // Store which values were originally missing
    const missing_mask = [];
    for (let i = 0; i < n; i++) {
      missing_mask[i] = [];
      for (let j = 0; j < nFeatures; j++) {
        missing_mask[i][j] = isMissing(X[i][j]);
      }
    }

    // Iterative imputation
    let prev_X = X_filled.map((row) => [...row]);

    for (let iter = 0; iter < this.max_iter; iter++) {
      // Impute each feature in round-robin fashion
      for (let featureIdx = 0; featureIdx < nFeatures; featureIdx++) {
        // Check if this feature has any missing values
        let hasMissing = false;
        for (let i = 0; i < n; i++) {
          if (missing_mask[i][featureIdx]) {
            hasMissing = true;
            break;
          }
        }

        if (!hasMissing) {
          continue; // Skip features without missing values
        }

        // Impute this feature
        const imputed = this._imputeFeature(X_filled, featureIdx);

        // Update only the originally missing values
        for (let i = 0; i < n; i++) {
          if (missing_mask[i][featureIdx]) {
            X_filled[i][featureIdx] = imputed[i];
          }
        }
      }

      // Check for convergence
      let maxChange = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < nFeatures; j++) {
          if (missing_mask[i][j]) {
            const change = Math.abs(X_filled[i][j] - prev_X[i][j]);
            maxChange = Math.max(maxChange, change);
          }
        }
      }

      if (this.verbose) {
        console.log(`Iteration ${iter + 1}: max change = ${maxChange}`);
      }

      if (maxChange < this.tol) {
        this.n_iter_ = iter + 1;
        if (this.verbose) {
          console.log(`Converged after ${this.n_iter_} iterations`);
        }
        break;
      }

      // Update prev_X
      prev_X = X_filled.map((row) => [...row]);
      this.n_iter_ = iter + 1;
    }

    // If input was table format, return full table with imputed values
    if (originalData !== null) {
      return applyColumns(normalize(originalData), selectedColumns, X_filled);
    }

    return X_filled;
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

/**
 * Iterative imputation (functional interface)
 * @param {Array<Array<number>>} X - Data with missing values
 * @param {Object} options - Imputer options
 * @returns {Array<Array<number>>} Imputed data
 */
export function iterativeImpute(X, options = {}) {
  const imputer = new IterativeImputer(options);
  return imputer.fit_transform(X);
}
