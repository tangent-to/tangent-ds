/**
 * Decision Tree estimators (classification & regression) using CART-style splits.
 */

import { Classifier, Regressor, Estimator } from "../../core/estimators/estimator.js";
import { prepareXY, prepareX } from "../../core/table.js";
import {
  gini,
  entropy,
  variance,
  mse,
  mae,
  getCriterionFunction,
} from "../criteria.js";

function toNumericMatrix(X) {
  return X.map((row) => (Array.isArray(row) ? row.map(Number) : [Number(row)]));
}

function toArray(y) {
  return Array.isArray(y) ? y.slice() : Array.from(y);
}

function majorityVote(labels) {
  const counts = new Map();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  let bestLabel = null;
  let bestCount = -Infinity;
  for (const [label, count] of counts.entries()) {
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }
  return bestLabel;
}

function meanValue(values) {
  if (values.length === 0) return 0;
  const sum = values.reduce((a, b) => a + b, 0);
  return sum / values.length;
}

function buildPreparedData(X, y) {
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareXY({
      X: X.X || X.columns,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
      encoders: X.encoders, // Pass encoders for label encoding
    });
    return {
      X: toNumericMatrix(prepared.X),
      y: prepared.y,
      columns: prepared.columnsX,
      encoders: prepared.encoders, // Return encoders
    };
  }

  return {
    X: toNumericMatrix(X),
    y: toArray(y),
    columns: null,
    encoders: null,
  };
}

function preparePredictInput(X, storedColumns) {
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || storedColumns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
    });
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

function featureSubset(allFeatures, maxFeatures, random) {
  if (!maxFeatures || maxFeatures >= allFeatures.length) {
    return allFeatures;
  }
  const features = allFeatures.slice();
  for (let i = features.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [features[i], features[j]] = [features[j], features[i]];
  }
  return features.slice(0, maxFeatures);
}

class DecisionTreeBase extends Estimator {
  constructor({
    maxDepth = 10,
    minSamplesSplit = 2,
    minSamplesLeaf = 1,
    minGain = 1e-7,
    maxLeafNodes = null,
    criterion = null,
    task = "classification",
    maxFeatures = null,
    random = Math.random,
  } = {}) {
    super({ maxDepth, minSamplesSplit, minSamplesLeaf, minGain, maxLeafNodes, criterion, task, maxFeatures });
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minGain = minGain;
    this.maxLeafNodes = maxLeafNodes;
    this.task = task;
    this.maxFeatures = maxFeatures;
    this.random = random;

    // Set default criterion based on task
    if (criterion === null) {
      this.criterion = task === "classification" ? "gini" : "mse";
    } else {
      this.criterion = criterion;
    }

    this.criterionFn = getCriterionFunction(this.criterion, task);

    this.root = null;
    this.columns = null;
    this.nFeatures = null;
    this._featureImportances = null;
    this.nLeaves = 0;
    this.nNodes = 0;
  }

  fit(X, y) {
    const prepared = buildPreparedData(X, y);
    return this._fitPrepared(prepared.X, prepared.y, prepared.columns);
  }

  _fitPrepared(X, y, columns) {
    this.columns = columns;
    this.nFeatures = X[0].length;
    this.trainX = X;
    this.trainY = y;

    // Initialize feature importances
    this._featureImportances = new Array(this.nFeatures).fill(0);
    this.nLeaves = 0;
    this.nNodes = 0;

    const features = Array.from({ length: this.nFeatures }, (_, i) => i);
    this.root = this._buildTree(X, y, 0, features, X.length);

    // Normalize feature importances
    const totalImportance = this._featureImportances.reduce((a, b) => a + b, 0);
    if (totalImportance > 0) {
      this._featureImportances = this._featureImportances.map(
        (x) => x / totalImportance,
      );
    }

    this.fitted = true;
    return this;
  }

  _buildTree(X, y, depth, allFeatures, totalSamples) {
    this.nNodes++;

    // Check stopping criteria
    if (
      depth >= this.maxDepth ||
      X.length < this.minSamplesSplit ||
      new Set(y).size === 1 ||
      (this.maxLeafNodes && this.nLeaves >= this.maxLeafNodes)
    ) {
      this.nLeaves++;
      return this._createLeaf(y);
    }

    const subset = featureSubset(allFeatures, this.maxFeatures, this.random);
    const { feature, threshold, gain, impurityDecrease } = this._bestSplit(
      X,
      y,
      subset,
    );

    if (gain < this.minGain || feature === null) {
      this.nLeaves++;
      return this._createLeaf(y);
    }

    const { leftX, leftY, rightX, rightY } = this._splitDataset(
      X,
      y,
      feature,
      threshold,
    );

    // Check min_samples_leaf constraint
    if (
      leftX.length < this.minSamplesLeaf ||
      rightX.length < this.minSamplesLeaf
    ) {
      this.nLeaves++;
      return this._createLeaf(y);
    }

    if (leftX.length === 0 || rightX.length === 0) {
      this.nLeaves++;
      return this._createLeaf(y);
    }

    // Track feature importance (weighted by number of samples)
    const sampleWeight = X.length / totalSamples;
    this._featureImportances[feature] += sampleWeight * impurityDecrease;

    return {
      type: "internal",
      feature,
      threshold,
      impurity: this.criterionFn(y),
      nSamples: X.length,
      left: this._buildTree(leftX, leftY, depth + 1, allFeatures, totalSamples),
      right: this._buildTree(
        rightX,
        rightY,
        depth + 1,
        allFeatures,
        totalSamples,
      ),
    };
  }

  _splitDataset(X, y, feature, threshold) {
    const leftX = [];
    const leftY = [];
    const rightX = [];
    const rightY = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return { leftX, leftY, rightX, rightY };
  }

  _bestSplit(X, y, features) {
    let bestFeature = null;
    let bestThreshold = null;
    let bestGain = -Infinity;
    let bestImpurityDecrease = 0;

    const parentImpurity = this.criterionFn(y);

    for (const feature of features) {
      const values = X.map((row) => row[feature]);
      const uniqueValues = Array.from(new Set(values)).sort((a, b) => a - b);
      if (uniqueValues.length <= 1) continue;

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const { leftY, rightY } = this._splitDataset(X, y, feature, threshold);

        if (leftY.length === 0 || rightY.length === 0) continue;

        const leftImpurity = this.criterionFn(leftY);
        const rightImpurity = this.criterionFn(rightY);

        const weightedImpurity =
          (leftY.length / y.length) * leftImpurity +
          (rightY.length / y.length) * rightImpurity;

        const gain = parentImpurity - weightedImpurity;
        const impurityDecrease = (y.length / this.trainY.length) * gain;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feature;
          bestThreshold = threshold;
          bestImpurityDecrease = impurityDecrease;
        }
      }
    }

    return {
      feature: bestFeature,
      threshold: bestThreshold,
      gain: bestGain,
      impurityDecrease: bestImpurityDecrease,
    };
  }

  _createLeaf(y) {
    if (this.task === "classification") {
      const counts = new Map();
      y.forEach((label) => counts.set(label, (counts.get(label) || 0) + 1));
      const total = y.length;
      const distribution = {};
      counts.forEach((count, label) => {
        distribution[label] = count / total;
      });
      return {
        type: "leaf",
        value: majorityVote(y),
        distribution,
        impurity: this.criterionFn(y),
        nSamples: y.length,
      };
    }
    return {
      type: "leaf",
      value: meanValue(y),
      impurity: this.criterionFn(y),
      nSamples: y.length,
    };
  }

  predict(X) {
    this._ensureFitted('predict');
    const data = preparePredictInput(X, this.columns);
    return data.map((row) => this._predictRow(row, this.root));
  }

  _predictRow(row, node) {
    if (node.type === "leaf") {
      return node.value;
    }
    if (row[node.feature] <= node.threshold) {
      return this._predictRow(row, node.left);
    }
    return this._predictRow(row, node.right);
  }

  /**
   * Apply tree to X, return leaf indices
   * @param {Array} X - Input data
   * @returns {Array<number>} Leaf indices for each sample
   */
  apply(X) {
    this._ensureFitted('apply');
    const data = preparePredictInput(X, this.columns);
    return data.map((row) => this._getLeafIndex(row, this.root, 0));
  }

  _getLeafIndex(row, node, idx) {
    if (node.type === "leaf") {
      return idx;
    }
    if (row[node.feature] <= node.threshold) {
      return this._getLeafIndex(row, node.left, idx * 2 + 1);
    }
    return this._getLeafIndex(row, node.right, idx * 2 + 2);
  }

  /**
   * Return decision path in the tree
   * @param {Array} X - Input data
   * @returns {Array<Array>} Decision path for each sample
   */
  decisionPath(X) {
    this._ensureFitted('decisionPath');
    const data = preparePredictInput(X, this.columns);
    return data.map((row) => {
      const path = [];
      this._recordPath(row, this.root, path);
      return path;
    });
  }

  _recordPath(row, node, path) {
    if (!node) return;
    path.push(node);
    if (node.type === "leaf") return;

    if (row[node.feature] <= node.threshold) {
      this._recordPath(row, node.left, path);
    } else {
      this._recordPath(row, node.right, path);
    }
  }

  /**
   * Get maximum depth of the tree
   * @returns {number} Maximum depth
   */
  getDepth() {
    this._ensureFitted('getDepth');
    return this._computeDepth(this.root);
  }

  _computeDepth(node) {
    if (!node || node.type === "leaf") {
      return 0;
    }
    return (
      1 +
      Math.max(this._computeDepth(node.left), this._computeDepth(node.right))
    );
  }

  /**
   * Get number of leaves in the tree
   * @returns {number} Number of leaves
   */
  getNLeaves() {
    this._ensureFitted('getNLeaves');
    return this.nLeaves;
  }

  /**
   * Export tree structure to DOT format for visualization
   * @param {Array<string>} featureNames - Optional feature names
   * @param {Array<string>} classNames - Optional class names
   * @returns {string} DOT format string
   */
  exportTree(featureNames = null, classNames = null) {
    this._ensureFitted('exportTree');

    const features =
      featureNames ||
      Array.from({ length: this.nFeatures }, (_, i) => `X[${i}]`);
    let nodeId = 0;
    let dot =
      'digraph Tree {\nnode [shape=box, style="rounded", fontname="helvetica"];\n';

    const buildDot = (node, parentId = null, edge = "") => {
      const currentId = nodeId++;

      if (node.type === "leaf") {
        let label = `samples = ${node.nSamples}\\n`;
        label += `impurity = ${node.impurity.toFixed(4)}\\n`;

        if (this.task === "classification") {
          const className = classNames
            ? classNames[node.value]
            : `class ${node.value}`;
          label += `value = ${className}`;
          dot += `${currentId} [label="${label}", fillcolor="#e5f5e0", style="filled,rounded"];\n`;
        } else {
          label += `value = ${node.value.toFixed(4)}`;
          dot += `${currentId} [label="${label}", fillcolor="#fef0d9", style="filled,rounded"];\n`;
        }
      } else {
        const featureName = features[node.feature];
        let label = `${featureName} <= ${node.threshold.toFixed(4)}\\n`;
        label += `impurity = ${node.impurity.toFixed(4)}\\n`;
        label += `samples = ${node.nSamples}`;

        dot += `${currentId} [label="${label}"];\n`;

        buildDot(node.left, currentId, "<=");
        buildDot(node.right, currentId, ">");
      }

      if (parentId !== null) {
        dot += `${parentId} -> ${currentId} [label="${edge}"];\n`;
      }

      return currentId;
    };

    buildDot(this.root);
    dot += "}";
    return dot;
  }

  /**
   * Export tree as ASCII text
   * @param {Array<string>} featureNames - Optional feature names
   * @returns {string} ASCII representation
   */
  exportText(featureNames = null) {
    this._ensureFitted('exportText');

    const features =
      featureNames ||
      Array.from({ length: this.nFeatures }, (_, i) => `X[${i}]`);
    let text = "";

    const buildText = (node, depth = 0, prefix = "") => {
      const indent = "  ".repeat(depth);

      if (node.type === "leaf") {
        if (this.task === "classification") {
          text += `${indent}${prefix}class: ${node.value} (samples=${node.nSamples}, impurity=${node.impurity.toFixed(4)})\n`;
        } else {
          text += `${indent}${prefix}value: ${node.value.toFixed(4)} (samples=${node.nSamples}, impurity=${node.impurity.toFixed(4)})\n`;
        }
      } else {
        const featureName = features[node.feature];
        text += `${indent}${prefix}${featureName} <= ${node.threshold.toFixed(4)} (samples=${node.nSamples}, impurity=${node.impurity.toFixed(4)})\n`;
        buildText(node.left, depth + 1, "├─ ");
        buildText(node.right, depth + 1, "└─ ");
      }
    };

    buildText(this.root);
    return text;
  }

  /**
   * Get feature importances
   * @returns {Array<number>} Feature importance scores
   */
  get featureImportances() {
    this._ensureFitted('featureImportances');
    return this._featureImportances;
  }
}

export class DecisionTreeClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.tree = new DecisionTreeBase({ ...opts, task: "classification" });
  }

  fit(X, y = null) {
    const prepared = buildPreparedData(X, y);

    // Use centralized label encoder extraction
    this._extractLabelEncoder(prepared);

    // Use centralized class extraction to handle encoded labels
    const { numericY, classes } = this._getClasses(prepared.y, false);

    // Store classes on both this instance and the tree
    this.classes_ = classes;
    this.tree.classes_ = classes;

    // Fit tree with numeric labels using prepared data
    this.tree._fitPrepared(prepared.X, numericY, prepared.columns);

    this.fitted = true;
    return this;
  }

  predict(X) {
    const predictions = this.tree.predict(X);
    // Use centralized label decoder
    return this._decodeLabels(predictions);
  }

  predictProba(X) {
    this._ensureFitted('predict');
    const data = preparePredictInput(X, this.tree.columns);
    return data.map((row) => {
      let node = this.tree.root;
      while (node.type !== "leaf") {
        if (row[node.feature] <= node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      }
      return node.distribution;
    });
  }

  /** Get feature importances */
  get featureImportances() {
    return this.tree.featureImportances;
  }

  /** Apply tree to X, return leaf indices */
  apply(X) {
    return this.tree.apply(X);
  }

  /** Return decision path */
  decisionPath(X) {
    return this.tree.decisionPath(X);
  }

  /** Get maximum depth of tree */
  getDepth() {
    return this.tree.getDepth();
  }

  /** Get number of leaves */
  getNLeaves() {
    return this.tree.getNLeaves();
  }

  /** Export tree to DOT format */
  exportTree(featureNames = null, classNames = null) {
    return this.tree.exportTree(featureNames, classNames);
  }

  /** Export tree as ASCII text */
  exportText(featureNames = null) {
    return this.tree.exportText(featureNames);
  }
}

export class DecisionTreeRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.tree = new DecisionTreeBase({ ...opts, task: "regression" });
  }

  fit(X, y = null) {
    this.tree.fit(X, y);
    this.fitted = true;
    return this;
  }

  predict(X) {
    return this.tree.predict(X);
  }

  /** Get feature importances */
  get featureImportances() {
    return this.tree.featureImportances;
  }

  /** Apply tree to X, return leaf indices */
  apply(X) {
    return this.tree.apply(X);
  }

  /** Return decision path */
  decisionPath(X) {
    return this.tree.decisionPath(X);
  }

  /** Get maximum depth of tree */
  getDepth() {
    return this.tree.getDepth();
  }

  /** Get number of leaves */
  getNLeaves() {
    return this.tree.getNLeaves();
  }

  /** Export tree to DOT format */
  exportTree(featureNames = null) {
    return this.tree.exportTree(featureNames);
  }

  /** Export tree as ASCII text */
  exportText(featureNames = null) {
    return this.tree.exportText(featureNames);
  }
}

export default {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
};
