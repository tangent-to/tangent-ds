/**
 * Random Forest estimators built on top of Decision Trees.
 */

import { Classifier, Regressor, Estimator } from "../../core/estimators/estimator.js";
import { prepareXY, prepareX } from "../../core/table.js";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
} from "./DecisionTree.js";

function toNumericMatrix(X) {
  return X.map((row) => (Array.isArray(row) ? row.map(Number) : [Number(row)]));
}

function bootstrapSample(X, y, random, maxSamples = null) {
  const n = X.length;
  const sampleSize = maxSamples !== null ? Math.min(maxSamples, n) : n;
  const XSample = [];
  const ySample = [];
  const indices = [];
  const oobIndices = new Set(Array.from({ length: n }, (_, i) => i));

  for (let i = 0; i < sampleSize; i++) {
    const idx = Math.floor(random() * n);
    XSample.push(X[idx]);
    ySample.push(y[idx]);
    indices.push(idx);
    oobIndices.delete(idx);
  }

  return { XSample, ySample, indices, oobIndices: Array.from(oobIndices) };
}

function createRandomGenerator(seed) {
  if (seed === null || seed === undefined) {
    return Math.random;
  }
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function prepareDataset(X, y) {
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
    y: Array.isArray(y) ? y.slice() : Array.from(y),
    columns: null,
    encoders: null,
  };
}

function preparePredict(X, columns) {
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || columns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
    });
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

function computeFeatureImportances(tree, nFeatures, nSamples) {
  const importances = new Array(nFeatures).fill(0);

  function traverse(node, nNodeSamples) {
    if (!node || node.type === "leaf") return;

    // For internal nodes, compute weighted impurity decrease
    const feature = node.feature;
    const threshold = node.threshold;

    // Estimate samples in left and right based on split
    // We don't have exact counts, so we'll use a simplified version
    // In practice, we'd need to track sample counts during tree building
    traverse(node.left, nNodeSamples);
    traverse(node.right, nNodeSamples);

    // Simplified importance: just count splits per feature
    importances[feature] += 1;
  }

  traverse(tree.root, nSamples);
  return importances;
}

function applyClassWeights(y, classWeight) {
  if (!classWeight) return null;

  const weights = new Array(y.length).fill(1);
  if (classWeight === "balanced") {
    const counts = new Map();
    y.forEach((label) => counts.set(label, (counts.get(label) || 0) + 1));
    const nSamples = y.length;
    const nClasses = counts.size;
    const weightMap = new Map();
    counts.forEach((count, label) => {
      weightMap.set(label, nSamples / (nClasses * count));
    });
    y.forEach((label, i) => {
      weights[i] = weightMap.get(label);
    });
  } else if (typeof classWeight === "object") {
    y.forEach((label, i) => {
      weights[i] = classWeight[label] || 1;
    });
  }
  return weights;
}

function applySampleWeights(y, sampleWeight, classWeights) {
  if (!sampleWeight && !classWeights) return null;

  const n = y.length;
  const weights = new Array(n).fill(1);

  if (classWeights) {
    for (let i = 0; i < n; i++) {
      weights[i] *= classWeights[i];
    }
  }

  if (sampleWeight) {
    for (let i = 0; i < n; i++) {
      weights[i] *= sampleWeight[i];
    }
  }

  return weights;
}

function weightedBootstrapSample(X, y, weights, random, maxSamples = null) {
  const n = X.length;
  const sampleSize = maxSamples !== null ? Math.min(maxSamples, n) : n;

  if (!weights) {
    return bootstrapSample(X, y, random, maxSamples);
  }

  // Compute cumulative weights for weighted sampling
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  const cumWeights = new Array(n);
  cumWeights[0] = weights[0] / totalWeight;
  for (let i = 1; i < n; i++) {
    cumWeights[i] = cumWeights[i - 1] + weights[i] / totalWeight;
  }

  const XSample = [];
  const ySample = [];
  const indices = [];
  const oobIndices = new Set(Array.from({ length: n }, (_, i) => i));

  for (let i = 0; i < sampleSize; i++) {
    const r = random();
    let idx = 0;
    for (let j = 0; j < n; j++) {
      if (r <= cumWeights[j]) {
        idx = j;
        break;
      }
    }
    XSample.push(X[idx]);
    ySample.push(y[idx]);
    indices.push(idx);
    oobIndices.delete(idx);
  }

  return { XSample, ySample, indices, oobIndices: Array.from(oobIndices) };
}

class RandomForestBase extends Estimator {
  constructor({
    nEstimators = 50,
    maxDepth = 10,
    minSamplesSplit = 2,
    maxFeatures = null,
    minImpurityDecrease = 0.0,
    maxSamples = null,
    classWeight = null,
    warmStart = false,
    oobScore = false,
    task = "classification",
    seed = null,
  } = {}) {
    super({ nEstimators, maxDepth, minSamplesSplit, maxFeatures, minImpurityDecrease, maxSamples, classWeight, warmStart, oobScore, task, seed });
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxFeatures = maxFeatures;
    this.minImpurityDecrease = minImpurityDecrease;
    this.maxSamples = maxSamples;
    this.classWeight = classWeight;
    this.warmStart = warmStart;
    this.oobScore = oobScore;
    this.task = task;
    this.seed = seed;
    this.trees = [];
    this.columns = null;
    this.random = createRandomGenerator(seed);
    this._featureImportances = null;
    this._oobScore = null;
    this._oobDecisionFunction = null;
  }

  fit(X, y, sampleWeight = null) {
    const prepared = prepareDataset(X, y);
    return this._fitPrepared(
      prepared.X,
      prepared.y,
      prepared.columns,
      sampleWeight,
    );
  }

  _fitPrepared(X, y, columns, sampleWeight = null) {
    this.columns = columns;

    const featureCount = X[0].length;
    const nSamples = X.length;
    const defaultMaxFeatures =
      this.task === "classification"
        ? Math.max(1, Math.floor(Math.sqrt(featureCount)))
        : Math.max(1, Math.floor(featureCount / 3));
    const featureBagSize = this.maxFeatures || defaultMaxFeatures;
    this.classes =
      this.task === "classification" ? Array.from(new Set(y)) : null;

    // Apply class weights if specified
    const classWeights =
      this.task === "classification"
        ? applyClassWeights(y, this.classWeight)
        : null;

    // Combine class weights and sample weights
    const combinedWeights = applySampleWeights(y, sampleWeight, classWeights);

    // Warm start: keep existing trees if enabled
    const startIdx = this.warmStart ? this.trees.length : 0;
    if (!this.warmStart) {
      this.trees = [];
    }

    // Initialize feature importances
    const featureImportances = new Array(featureCount).fill(0);

    // Initialize OOB tracking
    const oobPredictions = this.oobScore
      ? new Array(nSamples).fill(null).map(() => [])
      : null;

    for (let i = startIdx; i < this.nEstimators; i++) {
      const { XSample, ySample, oobIndices } = weightedBootstrapSample(
        X,
        y,
        combinedWeights,
        this.random,
        this.maxSamples,
      );

      const treeOpts = {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minGain: this.minImpurityDecrease,
        maxFeatures: featureBagSize,
        random: this.random,
      };

      const tree =
        this.task === "classification"
          ? new DecisionTreeClassifier(treeOpts)
          : new DecisionTreeRegressor(treeOpts);

      tree.fit(XSample, ySample);
      this.trees.push(tree);

      // Compute feature importances for this tree
      const treeImportances = computeFeatureImportances(
        tree.tree,
        featureCount,
        nSamples,
      );
      for (let f = 0; f < featureCount; f++) {
        featureImportances[f] += treeImportances[f];
      }

      // Compute OOB predictions if enabled
      if (this.oobScore && oobIndices.length > 0) {
        const oobX = oobIndices.map((idx) => X[idx]);
        const oobPreds = tree.predict(oobX);
        oobIndices.forEach((idx, j) => {
          oobPredictions[idx].push(oobPreds[j]);
        });
      }
    }

    // Normalize feature importances
    const totalImportance = featureImportances.reduce((a, b) => a + b, 0);
    if (totalImportance > 0) {
      this._featureImportances = featureImportances.map(
        (x) => x / totalImportance,
      );
    } else {
      this._featureImportances = featureImportances;
    }

    // Compute OOB score if enabled
    if (this.oobScore) {
      this._computeOOBScore(oobPredictions, y);
    }

    this.fitted = true;
    return this;
  }

  _computeOOBScore(oobPredictions, yTrue) {
    const n = yTrue.length;
    let validCount = 0;
    let correct = 0;
    let mse = 0;

    if (this.task === "classification") {
      for (let i = 0; i < n; i++) {
        if (oobPredictions[i].length === 0) continue;
        validCount++;

        // Majority vote
        const counts = new Map();
        oobPredictions[i].forEach((pred) => {
          counts.set(pred, (counts.get(pred) || 0) + 1);
        });
        let bestLabel = null;
        let bestCount = -Infinity;
        for (const [label, count] of counts.entries()) {
          if (count > bestCount) {
            bestCount = count;
            bestLabel = label;
          }
        }

        if (bestLabel === yTrue[i]) {
          correct++;
        }
      }

      this._oobScore = validCount > 0 ? correct / validCount : null;
    } else {
      // Regression: compute MSE
      for (let i = 0; i < n; i++) {
        if (oobPredictions[i].length === 0) continue;
        validCount++;

        const meanPred =
          oobPredictions[i].reduce((a, b) => a + b, 0) /
          oobPredictions[i].length;
        const error = meanPred - yTrue[i];
        mse += error * error;
      }

      // For regression, OOB score is R^2 score
      if (validCount > 0) {
        const yMean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
        let totalSS = 0;
        for (let i = 0; i < n; i++) {
          if (oobPredictions[i].length === 0) continue;
          const error = yTrue[i] - yMean;
          totalSS += error * error;
        }
        this._oobScore = totalSS > 0 ? 1 - mse / totalSS : null;
      } else {
        this._oobScore = null;
      }
    }

    this._oobDecisionFunction = oobPredictions;
  }

  _predictRaw(X) {
    const data = preparePredict(X, this.columns);
    const predictions = [];

    for (const row of data) {
      if (this.task === "classification") {
        const votes = new Map();
        this.trees.forEach((tree) => {
          const pred = tree.predict([row])[0];
          votes.set(pred, (votes.get(pred) || 0) + 1);
        });
        let bestLabel = null;
        let bestCount = -Infinity;
        for (const [label, count] of votes.entries()) {
          if (count > bestCount) {
            bestCount = count;
            bestLabel = label;
          }
        }
        predictions.push(bestLabel);
      } else {
        let sum = 0;
        this.trees.forEach((tree) => {
          sum += tree.predict([row])[0];
        });
        predictions.push(sum / this.trees.length);
      }
    }

    return predictions;
  }

  /**
   * Apply trees in the forest to X, return leaf indices.
   * Returns array of shape [n_samples, n_estimators].
   */
  apply(X) {
    this._ensureFitted('apply');

    const data = preparePredict(X, this.columns);
    const result = [];

    for (const row of data) {
      const leafIndices = [];
      for (const tree of this.trees) {
        const leafIdx = this._getLeafIndex(row, tree.tree.root, 0);
        leafIndices.push(leafIdx);
      }
      result.push(leafIndices);
    }

    return result;
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
   * Return the decision path in the forest.
   * Returns indicator matrix of shape [n_samples, n_nodes].
   */
  decisionPath(X) {
    this._ensureFitted('decisionPath');

    const data = preparePredict(X, this.columns);
    const paths = [];

    for (const row of data) {
      const treePaths = [];
      for (const tree of this.trees) {
        const path = [];
        this._recordPath(row, tree.tree.root, path);
        treePaths.push(path);
      }
      paths.push(treePaths);
    }

    return paths;
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

  get featureImportances() {
    this._ensureFitted('featureImportances');
    return this._featureImportances;
  }

  get oobScoreValue() {
    this._ensureFitted('oobScoreValue');
    if (!this.oobScore) {
      throw new Error(
        "RandomForest: oobScore=true must be set to compute OOB score.",
      );
    }
    return this._oobScore;
  }
}

export class RandomForestClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.forest = new RandomForestBase({ ...opts, task: "classification" });
  }

  fit(X, y = null, sampleWeight = null) {
    const prepared = prepareDataset(X, y);

    // Use centralized label encoder extraction
    this._extractLabelEncoder(prepared);

    // Use centralized class extraction to handle encoded labels
    const { numericY, classes } = this._getClasses(prepared.y, false);

    // Store classes on both this instance and the forest
    this.classes_ = classes;
    this.forest.classes_ = classes;
    this.forest.classes = classes;

    // Fit forest with numeric labels using prepared data
    this.forest._fitPrepared(
      prepared.X,
      numericY,
      prepared.columns,
      sampleWeight,
    );

    this.fitted = true;
    return this;
  }

  predict(X) {
    const predictions = this.forest._predictRaw(X);
    // Use centralized label decoder
    return this._decodeLabels(predictions);
  }

  predictProba(X) {
    this._ensureFitted('predictProba');
    const data = preparePredict(X, this.forest.columns);
    const proba = [];
    const labels = this.forest.classes;

    for (const row of data) {
      const votes = new Map();
      this.forest.trees.forEach((tree) => {
        const leafProba = tree.predictProba([row])[0];
        Object.keys(leafProba).forEach((label) => {
          votes.set(label, (votes.get(label) || 0) + leafProba[label]);
        });
      });
      const total = this.forest.trees.length;
      const dist = {};
      labels.forEach((label) => {
        dist[label] = (votes.get(label) || 0) / total;
      });
      proba.push(dist);
    }
    return proba;
  }

  /**
   * Get feature importances (MDI - Mean Decrease in Impurity)
   */
  get featureImportances() {
    return this.forest.featureImportances;
  }

  /**
   * Get out-of-bag score (accuracy for classification)
   */
  get oobScore() {
    return this.forest.oobScoreValue;
  }

  /**
   * Apply trees in forest to X, return leaf indices
   */
  apply(X) {
    return this.forest.apply(X);
  }

  /**
   * Return decision path through the forest
   */
  decisionPath(X) {
    return this.forest.decisionPath(X);
  }
}

export class RandomForestRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.forest = new RandomForestBase({ ...opts, task: "regression" });
  }

  fit(X, y = null, sampleWeight = null) {
    this.forest.fit(X, y, sampleWeight);
    this.fitted = true;
    return this;
  }

  predict(X) {
    return this.forest._predictRaw(X);
  }

  /**
   * Get feature importances (MDI - Mean Decrease in Impurity)
   */
  get featureImportances() {
    return this.forest.featureImportances;
  }

  /**
   * Get out-of-bag score (R^2 score for regression)
   */
  get oobScore() {
    return this.forest.oobScoreValue;
  }

  /**
   * Apply trees in forest to X, return leaf indices
   */
  apply(X) {
    return this.forest.apply(X);
  }

  /**
   * Return decision path through the forest
   */
  decisionPath(X) {
    return this.forest.decisionPath(X);
  }
}

export default {
  RandomForestClassifier,
  RandomForestRegressor,
};
