/**
 * Pipeline utilities for chaining preprocessing and models
 */

import { Estimator } from '../core/estimators/estimator.js';
import { prepareXY } from '../core/table.js';

function isTableDescriptor(input) {
  return (
    input &&
    typeof input === 'object' &&
    !Array.isArray(input) &&
    (input.data || input.rows)
  );
}

/**
 * Pipeline class for chaining transformers and estimators
 */
export class Pipeline extends Estimator {
  /**
   * Create a pipeline
   * @param {Array<Object>} steps - Array of transformers/estimators
   */
  constructor(steps) {
    super({ steps });
    if (!Array.isArray(steps) || steps.length === 0) {
      throw new Error('Pipeline requires at least one step');
    }

    this.steps = steps;
  }

  /**
   * Fit all steps
   * @param {Array} X - Feature matrix
   * @param {Array} y - Target values (optional)
   * @returns {Pipeline} this
   */
  fit(X, y = null) {
    let XTransformed = X;

    // Fit and transform through all but last step
    for (let i = 0; i < this.steps.length - 1; i++) {
      const step = this.steps[i];

      if (typeof step.fit !== 'function') {
        throw new Error(`Step ${i} must have a fit() method`);
      }

      step.fit(XTransformed, y);

      if (typeof step.transform === 'function') {
        XTransformed = step.transform(XTransformed);
      }
    }

    // Fit final step
    const finalStep = this.steps[this.steps.length - 1];
    if (typeof finalStep.fit === 'function') {
      finalStep.fit(XTransformed, y);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform data through all steps
   * @param {Array} X - Feature matrix
   * @returns {Array} Transformed data
   */
  transform(X) {
    this._ensureFitted('transform');

    let XTransformed = X;

    for (const step of this.steps) {
      if (typeof step.transform === 'function') {
        XTransformed = step.transform(XTransformed);
      }
    }

    return XTransformed;
  }

  /**
   * Fit and transform in one step
   * @param {Array} X - Feature matrix
   * @param {Array} y - Target values (optional)
   * @returns {Array} Transformed data
   */
  fitTransform(X, y = null) {
    this.fit(X, y);
    return this.transform(X);
  }

  /**
   * Predict using final estimator
   * @param {Array} X - Feature matrix
   * @returns {Array} Predictions
   */
  predict(X) {
    this._ensureFitted('predict');

    let XTransformed = X;

    // Transform through all but last step
    for (let i = 0; i < this.steps.length - 1; i++) {
      const step = this.steps[i];
      if (typeof step.transform === 'function') {
        XTransformed = step.transform(XTransformed);
      }
    }

    // Predict with final step
    const finalStep = this.steps[this.steps.length - 1];
    if (typeof finalStep.predict !== 'function') {
      throw new Error('Final step must have a predict() method');
    }

    return finalStep.predict(XTransformed);
  }

  /**
   * Get final estimator
   * @returns {Object} Final step
   */
  getFinalEstimator() {
    return this.steps[this.steps.length - 1];
  }
}

/**
 * Simple GridSearchCV for hyperparameter tuning
 */
export class GridSearchCV extends Estimator {
  /**
   * Create grid search
   * @param {Function} estimatorFn - Function that creates estimator: (params) => estimator
   * @param {Object} paramGrid - Grid of parameters: {param1: [val1, val2], ...}
   * @param {Function} scoreFn - Scoring function: (yTrue, yPred) => score
   * @param {number} cv - Number of cross-validation folds
   */
  constructor(estimatorFn, paramGrid, scoreFn, cv = 5) {
    super({ cv });
    this.estimatorFn = estimatorFn;
    this.paramGrid = paramGrid;
    this.scoreFn = scoreFn;
    this.cv = cv;
    this.bestParams = null;
    this.bestScore = -Infinity;
    this.bestEstimator = null;
    this.cvResults = [];
  }

  /**
   * Generate all parameter combinations
   * @private
   */
  _generateParamCombinations() {
    const keys = Object.keys(this.paramGrid);
    const values = keys.map((k) => this.paramGrid[k]);

    const combinations = [];

    const generate = (current, depth) => {
      if (depth === keys.length) {
        combinations.push({ ...current });
        return;
      }

      for (const value of values[depth]) {
        current[keys[depth]] = value;
        generate(current, depth + 1);
      }
    };

    generate({}, 0);
    return combinations;
  }

  /**
   * Perform grid search with cross-validation
   * @param {Array} X - Feature matrix
   * @param {Array} y - Target values
   * @returns {GridSearchCV} this
   */
  fit(X, y) {
    let dataX = X;
    let dataY = y;

    if (isTableDescriptor(X)) {
      if (!X.y) {
        throw new Error('GridSearchCV.fit requires a y column when using table descriptors');
      }
      const prepared = prepareXY({
        X: X.X || X.columns,
        y: X.y,
        data: X.data || X.rows,
        naOmit: X.naOmit,
        omit_missing: X.omit_missing,
        encode: X.encode,
      });
      dataX = prepared.X;
      dataY = prepared.y;
    }

    if (!Array.isArray(dataX) || !Array.isArray(dataY)) {
      throw new Error('GridSearchCV.fit expects array inputs for X and y');
    }

    const combinations = this._generateParamCombinations();
    const n = dataX.length;
    const foldSize = Math.floor(n / this.cv);

    for (const params of combinations) {
      const scores = [];

      // Cross-validation
      for (let fold = 0; fold < this.cv; fold++) {
        const testStart = fold * foldSize;
        const testEnd = fold === this.cv - 1 ? n : (fold + 1) * foldSize;

        const XTrain = [...dataX.slice(0, testStart), ...dataX.slice(testEnd)];
        const yTrain = [...dataY.slice(0, testStart), ...dataY.slice(testEnd)];
        const XTest = dataX.slice(testStart, testEnd);
        const yTest = dataY.slice(testStart, testEnd);

        const estimator = this.estimatorFn(params);
        estimator.fit(XTrain, yTrain);
        const yPred = estimator.predict(XTest);
        const score = this.scoreFn(yTest, yPred);
        scores.push(score);
      }

      const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;

      this.cvResults.push({
        params,
        meanScore,
        scores,
      });

      if (meanScore > this.bestScore) {
        this.bestScore = meanScore;
        this.bestParams = params;
      }
    }

    // Fit best estimator on full data
    this.bestEstimator = this.estimatorFn(this.bestParams);
    this.bestEstimator.fit(dataX, dataY);

    this.fitted = true;
    return this;
  }

  /**
   * Predict using best estimator
   * @param {Array} X - Feature matrix
   * @returns {Array} Predictions
   */
  predict(X) {
    this._ensureFitted('predict');
    return this.bestEstimator.predict(X);
  }

  /**
   * Get results sorted by score
   * @returns {Array<Object>} Results
   */
  getResults() {
    return [...this.cvResults].sort((a, b) => b.meanScore - a.meanScore);
  }
}
