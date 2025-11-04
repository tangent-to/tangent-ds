/**
 * Unified interface for table-like data
 * Accepts arrays of objects or Arquero-like tables
 */

import { Matrix } from "./linalg.js";

/**
 * Check if input has Arquero-like interface
 * @param {*} data - Input data
 * @returns {boolean} True if has .objects() method
 */
function isArqueroLike(data) {
  return data && typeof data.objects === "function";
}

/**
 * Normalize input to array of objects
 * @param {Array<Object>|Object} data - Input data
 * @returns {Array<Object>} Array of row objects
 */
export function normalize(data) {
  if (isArqueroLike(data)) {
    return data.objects();
  }
  if (Array.isArray(data)) {
    return data;
  }
  throw new Error("Data must be array of objects or Arquero-like table");
}

/**
 * Convert table data to matrix
 * @param {Array<Object>|Object} data - Input data
 * @param {Array<string>} columns - Column names to extract
 * @returns {Matrix} Matrix with selected columns
 */
export function toMatrix(data, columns) {
  const rows = normalize(data);

  if (rows.length === 0) {
    throw new Error("Cannot create matrix from empty data");
  }

  // If no columns specified, use all numeric columns
  if (!columns) {
    const firstRow = rows[0];
    columns = Object.keys(firstRow).filter((key) =>
      typeof firstRow[key] === "number"
    );
  }

  const matrix = rows.map((row) =>
    columns.map((col) => {
      const val = row[col];
      if (typeof val !== "number") {
        throw new Error(`Column ${col} contains non-numeric value: ${val}`);
      }
      return val;
    })
  );

  return new Matrix(matrix);
}

/**
 * Convert table column to vector
 * @param {Array<Object>|Object} data - Input data
 * @param {string} column - Column name
 * @returns {Array<number>} 1D array
 */
export function toVector(data, column) {
  const rows = normalize(data);

  if (rows.length === 0) {
    throw new Error("Cannot create vector from empty data");
  }

  return rows.map((row) => {
    const val = row[column];
    if (typeof val !== "number") {
      throw new Error(`Column ${column} contains non-numeric value: ${val}`);
    }
    return val;
  });
}

/**
 * Extract multiple columns as arrays
 * @param {Array<Object>|Object} data - Input data
 * @param {Array<string>} columns - Column names
 * @returns {Object} Object with column names as keys and arrays as values
 */
export function toColumns(data, columns) {
  const rows = normalize(data);
  const result = {};

  for (const col of columns) {
    result[col] = toVector(rows, col);
  }

  return result;
}

/**
 * Get column names from data
 * @param {Array<Object>|Object} data - Input data
 * @returns {Array<string>} Column names
 */
export function getColumns(data) {
  const rows = normalize(data);
  if (rows.length === 0) {
    return [];
  }
  return Object.keys(rows[0]);
}

/**
 * Filter rows based on predicate
 * @param {Array<Object>|Object} data - Input data
 * @param {Function} predicate - Filter function
 * @returns {Array<Object>} Filtered rows
 */
export function filter(data, predicate) {
  return normalize(data).filter(predicate);
}

/**
 * Select specific columns
 * @param {Array<Object>|Object} data - Input data
 * @param {Array<string>} columns - Columns to select
 * @returns {Array<Object>} Rows with selected columns
 */
export function select(data, columns) {
  const rows = normalize(data);
  return rows.map((row) => {
    const newRow = {};
    for (const col of columns) {
      newRow[col] = row[col];
    }
    return newRow;
  });
}

/**
 * Helper: check if a value is numeric (finite number)
 * @param {*} v
 * @returns {boolean}
 */
function isNumeric(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * Helper: check if a value is considered missing
 * @param {*} v
 * @returns {boolean}
 */
function isMissing(v) {
  return v === null || v === undefined ||
    (typeof v === "number" && Number.isNaN(v));
}

/**
 * Simple Label Encoder for categorical labels -> integers
 */
export class LabelEncoder {
  constructor() {
    this.classes_ = [];
    this.classIndex = new Map();
  }

  fit(values = []) {
    this.classes_ = [];
    this.classIndex = new Map();
    for (const v of values) {
      if (isMissing(v)) continue;
      if (!this.classIndex.has(v)) {
        this.classIndex.set(v, this.classes_.length);
        this.classes_.push(v);
      }
    }
    return this;
  }

  transform(values = []) {
    return values.map((v) => {
      if (isMissing(v)) return NaN;
      const idx = this.classIndex.get(v);
      if (idx === undefined) {
        // unseen category -> register as new class (useful for small pipelines)
        const newIdx = this.classes_.length;
        this.classIndex.set(v, newIdx);
        this.classes_.push(v);
        return newIdx;
      }
      return idx;
    });
  }

  fitTransform(values = []) {
    this.fit(values);
    return this.transform(values);
  }

  inverseTransform(indices = []) {
    return indices.map((i) => this.classes_[i]);
  }

  toJSON() {
    return { __class__: "LabelEncoder", classes: this.classes_ };
  }

  static fromJSON(obj = {}) {
    const le = new LabelEncoder();
    if (Array.isArray(obj.classes)) {
      le.classes_ = obj.classes.slice();
      le.classIndex = new Map(obj.classes.map((c, i) => [c, i]));
    }
    return le;
  }
}

/**
 * Simple OneHotEncoder for a single categorical column
 * Note: this encoder returns an array of arrays (one-hot vectors)
 */
export class OneHotEncoder {
  constructor({ handleUnknown = "ignore" } = {}) {
    this.categories_ = [];
    this.catIndex = new Map();
    this.handleUnknown = handleUnknown; // "ignore" or "error"
  }

  fit(values = []) {
    this.categories_ = [];
    this.catIndex = new Map();
    for (const v of values) {
      if (isMissing(v)) continue;
      if (!this.catIndex.has(v)) {
        this.catIndex.set(v, this.categories_.length);
        this.categories_.push(v);
      }
    }
    return this;
  }

  transform(values = []) {
    const k = this.categories_.length;
    return values.map((v) => {
      if (isMissing(v)) {
        // represent missing as all zeros
        return new Array(k).fill(0);
      }
      const idx = this.catIndex.get(v);
      if (idx === undefined) {
        if (this.handleUnknown === "ignore") {
          return new Array(k).fill(0);
        } else {
          throw new Error(`Unknown category encountered during transform: ${v}`);
        }
      }
      const vec = new Array(k).fill(0);
      vec[idx] = 1;
      return vec;
    });
  }

  fitTransform(valuesOrOptions = []) {
    // Declarative API: fitTransform({ data, columns })
    if (valuesOrOptions && typeof valuesOrOptions === 'object' && !Array.isArray(valuesOrOptions)) {
      if ('data' in valuesOrOptions || 'columns' in valuesOrOptions) {
        return this._fitTransformDeclarative(valuesOrOptions);
      }
    }

    // Array API: fitTransform(values)
    this.fit(valuesOrOptions);
    return this.transform(valuesOrOptions);
  }

  /**
   * Declarative API for fit
   * @param {Object} options - { data, columns }
   * @returns {OneHotEncoder} this
   */
  _fitDeclarative({ data, columns }) {
    if (!columns) {
      throw new Error('OneHotEncoder: columns parameter is required for declarative API');
    }

    const columnList = Array.isArray(columns) ? columns : [columns];
    const rows = normalize(data);

    // Store column configuration
    this._columns = columnList;
    this._encoders = new Map();

    // Create an encoder for each column
    for (const col of columnList) {
      const encoder = new OneHotEncoder({ handleUnknown: this.handleUnknown });
      const values = rows.map(row => row[col]);
      encoder.fit(values);
      this._encoders.set(col, encoder);
    }

    return this;
  }

  /**
   * Declarative API for transform
   * @param {Object} options - { data }
   * @returns {Array<Object>} Array of objects with one-hot encoded columns
   */
  _transformDeclarative({ data }) {
    if (!this._encoders) {
      throw new Error('OneHotEncoder: must call fit before transform');
    }

    const rows = normalize(data);

    return rows.map(row => {
      const encoded = {};

      for (const [col, encoder] of this._encoders.entries()) {
        const value = row[col];
        const oneHotVec = encoder.transform([value])[0];
        const featureNames = encoder.getFeatureNames(`${col}_`);

        // Add each one-hot feature as a separate property
        featureNames.forEach((name, idx) => {
          encoded[name] = oneHotVec[idx];
        });
      }

      return encoded;
    });
  }

  /**
   * Declarative API for fitTransform
   * @param {Object} options - { data, columns }
   * @returns {Array<Object>} Array of objects with one-hot encoded columns
   */
  _fitTransformDeclarative(options) {
    this._fitDeclarative(options);
    return this._transformDeclarative({ data: options.data });
  }

  /**
   * Get all feature names for declarative API
   * @returns {Array<string>} All feature names across all columns
   */
  getFeatureNames(prefix = "") {
    // If using declarative API, return all feature names
    if (this._encoders) {
      const names = [];
      for (const [col, encoder] of this._encoders.entries()) {
        names.push(...encoder.getFeatureNames(`${col}_`));
      }
      return names;
    }

    // Original single-column API
    return this.categories_.map((c) => `${prefix}${c}`);
  }

  toJSON() {
    return { __class__: "OneHotEncoder", categories: this.categories_ };
  }

  static fromJSON(obj = {}) {
    const enc = new OneHotEncoder();
    if (Array.isArray(obj.categories)) {
      enc.categories_ = obj.categories.slice();
      enc.catIndex = new Map(enc.categories_.map((c, i) => [c, i]));
    }
    return enc;
  }
}

/**
 * Prepare feature matrix X from table-like data.
 * Supports optional categorical encoding:
 *   prepareX({ columns, data, omit_missing = true, encode = null })
 *
 * encode can be:
 *   - null/false: no encoding (default)
 *   - true: auto-label encode any non-numeric columns
 *   - { colName: 'label' | 'onehot' } mapping per-column
 *
 * Returns:
 *   { X, columns, n, rows, encoders } where encoders is a mapping of column->encoder used
 */
export function prepareX({ columns, data, omit_missing = true, encode = null } = {}) {
  if (!columns) {
    // If no columns specified, select numeric columns from first row
    const rows0 = normalize(data);
    if (rows0.length === 0) {
      throw new Error("Cannot prepare X from empty data");
    }
    columns = Object.keys(rows0[0]).filter((k) => isNumeric(rows0[0][k]));
  } else if (typeof columns === "string") {
    columns = [columns];
  }

  const rows = normalize(data);

  // Validate that requested columns exist
  for (const c of columns) {
    if (!rows.length) break;
    if (!(c in rows[0])) {
      throw new Error(`Column ${c} not found in data`);
    }
  }

  // If omit_missing is true, drop rows with missing in any requested columns
  const preFiltered = omit_missing
    ? rows.filter((r) => columns.every((c) => !isMissing(r[c])))
    : rows.slice();

  // Determine per-column encoders if requested
  const encoders = {}; // columnName -> encoder instance
  const finalColumnNames = []; // will hold expanded column names (for one-hot)
  // For onehot columns we will expand the columns into multiple feature names

  // Helper to detect non-numeric column in the filtered rows
  function columnIsNumeric(col) {
    for (const r of preFiltered) {
      const v = r[col];
      if (v === null || v === undefined) continue;
      if (!isNumeric(v)) return false;
    }
    return true;
  }

  // Decide encoders based on `encode` param
  for (const col of columns) {
    const numeric = columnIsNumeric(col);
    if (numeric) {
      encoders[col] = null;
      finalColumnNames.push(col);
      continue;
    }

    // Non-numeric column
    if (!encode) {
      // If user did not request encoding, throw informative error
      throw new Error(`Column ${col} contains non-numeric values; pass encode option to prepareX to encode categorical columns`);
    }

    // Determine requested encoding for this column
    let mode = null;
    if (encode === true) {
      mode = "label";
    } else if (typeof encode === "string") {
      mode = encode; // 'label' or 'onehot'
    } else if (encode && typeof encode === "object") {
      mode = encode[col] || null;
    }

    // Default fallback
    if (!mode) mode = "label";

    if (mode === "label") {
      const le = new LabelEncoder();
      const vals = preFiltered.map((r) => r[col]);
      le.fit(vals);
      encoders[col] = le;
      finalColumnNames.push(col); // label encoder produces one numeric column with same name
    } else if (mode === "onehot" || mode === "ohe") {
      const ohe = new OneHotEncoder();
      const vals = preFiltered.map((r) => r[col]);
      ohe.fit(vals);
      encoders[col] = ohe;
      // expand finalColumnNames with category-specific names
      for (const cname of ohe.getFeatureNames(`${col}_`)) {
        finalColumnNames.push(cname);
      }
    } else {
      throw new Error(`Unknown encode mode for column ${col}: ${mode}`);
    }
  }

  // Build matrix (array of arrays) using encoders where necessary
  const X = preFiltered.map((row) => {
    const outRow = [];
    for (const col of columns) {
      const enc = encoders[col];
      const v = row[col];
      if (!enc) {
        // Expect numeric
        if (!isNumeric(v)) {
          throw new Error(`Column ${col} contains non-numeric value: ${v}`);
        }
        outRow.push(v);
      } else if (enc instanceof LabelEncoder) {
        // single numeric value
        const val = enc.classIndex.has(v) ? enc.classIndex.get(v) : NaN;
        outRow.push(isMissing(val) ? NaN : val);
      } else if (enc instanceof OneHotEncoder) {
        // expand into multiple entries
        const vec = enc.transform([v])[0];
        for (const e of vec) outRow.push(e);
      } else {
        throw new Error(`Unsupported encoder for column ${col}`);
      }
    }
    return outRow;
  });

  return { X, columns: finalColumnNames, n: X.length, rows: preFiltered, encoders };
}

/**
 * Utility to one-hot encode columns in a table-like object.
 *
 * @param {Object} options
 * @param {Array|Object} options.data - Array of row objects or table-like input
 * @param {string|Array<string>} options.columns - Column or columns to encode
 * @param {boolean} [options.dropFirst=true] - Drop first dummy (gives D-1 columns)
 * @param {boolean} [options.keepOriginal=false] - Keep the original categorical column
 * @param {boolean} [options.prefix=true] - Prefix generated column names with original column name
 * @param {string} [options.handleUnknown="ignore"] - Behaviour for unseen categories
 * @returns {Object} { data, dummyInfo }
 */
export function oneHotEncodeTable({
  data,
  columns,
  dropFirst = true,
  keepOriginal = false,
  prefix = true,
  handleUnknown = "ignore"
} = {}) {
  if (!data) {
    throw new Error("oneHotEncodeTable: data parameter is required");
  }

  const rows = normalize(data);
  const columnList = Array.isArray(columns) ? columns : [columns];
  const resultRows = rows.map((row) => ({ ...row }));
  const dummyInfo = new Map();

  for (const column of columnList) {
    if (!rows.length || !(column in rows[0])) {
      throw new Error(`oneHotEncodeTable: column ${column} not found in data`);
    }

    const encoder = new OneHotEncoder({ handleUnknown });
    const values = rows.map((row) => row[column]);
    encoder.fit(values);

    const categories = encoder.categories_.slice();
    const includedIndices = [];
    const columnNames = [];

    categories.forEach((cat, idx) => {
      if (dropFirst && idx === 0) return;
      includedIndices.push(idx);
      const colName = prefix ? `${column}_${String(cat)}` : String(cat);
      columnNames.push(colName);
    });

    resultRows.forEach((row, rowIndex) => {
      const encodedVec = encoder.transform([values[rowIndex]])[0];
      includedIndices.forEach((catIdx, idx) => {
        const colName = columnNames[idx];
        row[colName] = encodedVec[catIdx];
      });
      if (!keepOriginal) {
        delete row[column];
      }
    });

    dummyInfo.set(column, {
      column,
      categories,
      dropFirst,
      columnNames,
      encoder
    });
  }

  return {
    data: resultRows,
    dummyInfo
  };
}

/**
 * Prepare feature matrix X and response vector y from table-like data.
 * Supports categorical encoding for X and y via `encode` option:
 *   prepareXY({ X, y, data, omit_missing = true, encode = null })
 *
 * encode semantics same as prepareX. For y, only 'label' encoding is supported
 * (maps categories to integer class labels).
 *
 * Returns:
 *   { X, y, columnsX, n, rows, encoders } where encoders may include encoders.y
 */
export function prepareXY({ X, y, data, omit_missing = true, encode = null } = {}) {
  if (!data) {
    throw new Error(
      "Data argument is required when using column names for X/y",
    );
  }

  // Normalize X param to columns array
  const columnsX = (typeof X === "string") ? [X] : Array.isArray(X) ? X : null;

  if (!columnsX || typeof y !== "string") {
    throw new Error(
      "prepareXY expects X (string or array of strings) and y (string)",
    );
  }

  const rows = normalize(data);

  // Validate presence of columns
  if (rows.length > 0) {
    for (const c of [...columnsX, y]) {
      if (!(c in rows[0])) {
        throw new Error(`Column ${c} not found in data`);
      }
    }
  }

  // If omit_missing: filter rows where any of the X columns or y are missing
  const preFiltered = omit_missing
    ? rows.filter((r) => {
      return columnsX.every((c) => !isMissing(r[c])) && !isMissing(r[y]);
    })
    : rows.slice();

  // For X we will reuse prepareX logic by delegating to prepareX with the same encode map
  const xPrep = prepareX({
    columns: columnsX,
    data: preFiltered,
    omit_missing: false, // already filtered above
    encode: encode,
  });

  // Handle y: if numeric, simply extract; if non-numeric and encode indicates label encoding (or encode===true), apply LabelEncoder
  const yValsRaw = preFiltered.map((row) => row[y]);
  let yvec = [];
  let encoders = xPrep.encoders || {};
  // decide y encoder
  const yNeedsEncoding = !yValsRaw.every((v) => isNumeric(v));
  if (yNeedsEncoding) {
    // Determine if user requested encoding for y
    let yMode = null;
    if (encode === true) yMode = "label";
    else if (typeof encode === "string") yMode = encode;
    else if (encode && typeof encode === "object") yMode = encode[y] || null;
    if (!yMode) {
      // default to label encoding for y if non-numeric and no explicit instruction
      yMode = "label";
    }

    if (yMode === "label") {
      const le = new LabelEncoder();
      le.fit(yValsRaw);
      yvec = le.transform(yValsRaw);
      encoders = { ...encoders, y: le };
    } else {
      throw new Error("prepareXY: only 'label' encoding is supported for y (response)");
    }
  } else {
    yvec = yValsRaw.map((v) => Number(v));
  }

  if (xPrep.X.length !== yvec.length) {
    throw new Error("Mismatch between prepared X rows and y length");
  }

  return { X: xPrep.X, y: yvec, columnsX: xPrep.columns, n: xPrep.n, rows: preFiltered, encoders };
}
