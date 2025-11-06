/**
 * Generalized Additive Models (regression & classification) using
 * cubic spline basis functions with preselected knots.
 */

import { Regressor, Classifier } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';
import { fitGLM, predictGLM } from '../../stats/glm.js';

// Minimal lm/logit namespaces for compatibility
const lm = {
  fit: (X, y, opts) => fitGLM(X, y, { ...opts, family: 'gaussian' }),
  predict: (coefficients, X, opts) => {
    const model = { coefficients, family: 'gaussian', link: 'identity', intercept: opts?.intercept !== false, p: coefficients.length };
    return predictGLM(model, X, opts);
  },
  summary: (model) => ({
    coefficients: model.coefficients,
    rSquared: model.pseudoR2 || model.rSquared,
    adjRSquared: model.adjRSquared
  })
};

const logit = {
  fit: (X, y, opts) => fitGLM(X, y, { ...opts, family: 'binomial', link: 'logit' }),
  predict: (coefficients, X, opts) => {
    const model = { coefficients, family: 'binomial', link: 'logit', intercept: opts?.intercept !== false, p: coefficients.length };
    return predictGLM(model, X, opts);
  }
};

function toNumericMatrix(X) {
  return X.map((row) => Array.isArray(row) ? row.map(Number) : [Number(row)]);
}

function prepareDataset(X, y) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareXY({
      X: X.X || X.columns,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return {
      X: toNumericMatrix(prepared.X),
      y: Array.isArray(prepared.y) ? prepared.y.slice() : Array.from(prepared.y),
      columns: prepared.columnsX
    };
  }
  return {
    X: toNumericMatrix(X),
    y: Array.isArray(y) ? y.slice() : Array.from(y),
    columns: null
  };
}

function preparePredictInput(X, columns) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || columns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

function computeKnots(values, nSplines) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const knots = [];
  for (let i = 1; i < nSplines; i++) {
    const idx = Math.floor((i / (nSplines + 1)) * (sorted.length - 1));
    knots.push(sorted[idx]);
  }
  return knots;
}

function splineBasis(value, knots) {
  const basis = [value];
  for (const knot of knots) {
    const diff = value - knot;
    basis.push(diff > 0 ? diff * diff * diff : 0);
  }
  return basis;
}

function buildDesignMatrix(X, featureConfigs, includeIntercept = true) {
  const design = [];
  for (const row of X) {
    const features = [];
    if (includeIntercept) features.push(1);
    for (let j = 0; j < row.length; j++) {
      const basis = splineBasis(row[j], featureConfigs[j]);
      features.push(...basis);
    }
    design.push(features);
  }
  return design;
}

class GAMBase {
  constructor({
    nSplines = 4,
    includeLinear = true,
    task = 'regression',
    maxIter = 100,
    tol = 1e-6
  } = {}) {
    this.nSplines = nSplines;
    this.includeLinear = includeLinear;
    this.task = task;
    this.maxIter = maxIter;
    this.tol = tol;

    this.featureConfigs = null;
    this.columns = null;
    this.coef = null;
    this.classMap = null;
  }

  _buildFeatures(X) {
    const p = X[0].length;
    const configs = [];
    for (let j = 0; j < p; j++) {
      const values = X.map((row) => row[j]);
      configs.push(computeKnots(values, this.nSplines));
    }
    this.featureConfigs = configs;
  }

  _designMatrix(X) {
    return buildDesignMatrix(X, this.featureConfigs, true);
  }
}

export class GAMRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.gam = new GAMBase({ ...opts, task: 'regression' });
  }

  fit(X, y = null) {
    const prepared = prepareDataset(X, y);
    this.gam.columns = prepared.columns;
    this.gam._buildFeatures(prepared.X);
    const design = this.gam._designMatrix(prepared.X);
    const lmResult = lm.fit(design, prepared.y, { intercept: false });
    this.gam.coef = lmResult.coefficients;
    this.gam.summary = lm.summary(lmResult);
    this.fitted = true;
    return this;
  }

  predict(X) {
    if (!this.fitted) throw new Error('GAMRegressor: estimator not fitted.');
    const data = preparePredictInput(X, this.gam.columns);
    const design = this.gam._designMatrix(data);
    const predictions = [];
    for (const row of design) {
      let sum = 0;
      for (let i = 0; i < row.length; i++) {
        sum += row[i] * this.gam.coef[i];
      }
      predictions.push(sum);
    }
    return predictions;
  }

  summary() {
    if (!this.fitted) {
      throw new Error('GAMRegressor: estimator not fitted.');
    }
    return this.gam.summary;
  }
}

export class GAMClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.gam = new GAMBase({ ...opts, task: 'classification' });
  }

  fit(X, y = null) {
    const prepared = prepareDataset(X, y);
    const classes = Array.from(new Set(prepared.y));
    if (classes.length !== 2) {
      throw new Error('GAMClassifier currently supports exactly 2 classes.');
    }
    this.gam.classMap = {
      [classes[0]]: 0,
      [classes[1]]: 1
    };
    this.gam.classes = classes;
    const numericY = prepared.y.map((label) => this.gam.classMap[label]);
    this.gam.columns = prepared.columns;
    this.gam._buildFeatures(prepared.X);
    const design = this.gam._designMatrix(prepared.X);
    const logitResult = logit.fit(design, numericY, {
      intercept: false,
      maxIter: this.gam.maxIter,
      tol: this.gam.tol
    });
    this.gam.coef = logitResult.coefficients;
    this.gam.model = logitResult;
    this.fitted = true;
    return this;
  }

  _decisionFunction(X) {
    const design = this.gam._designMatrix(X);
    const scores = [];
    for (const row of design) {
      let sum = 0;
      for (let i = 0; i < row.length; i++) {
        sum += row[i] * this.gam.coef[i];
      }
      scores.push(sum);
    }
    return scores;
  }

  predictProba(X) {
    if (!this.fitted) throw new Error('GAMClassifier: estimator not fitted.');
    const data = preparePredictInput(X, this.gam.columns);
    const scores = this._decisionFunction(data);
    const probs = scores.map((score) => {
      const p = 1 / (1 + Math.exp(-score));
      return { [this.gam.classes[0]]: 1 - p, [this.gam.classes[1]]: p };
    });
    return probs;
  }

  predict(X) {
    const probs = this.predictProba(X);
    const [negLabel, posLabel] = this.gam.classes;
    return probs.map((dist) => (dist[posLabel] >= 0.5 ? posLabel : negLabel));
  }
}

export default {
  GAMRegressor,
  GAMClassifier
};
