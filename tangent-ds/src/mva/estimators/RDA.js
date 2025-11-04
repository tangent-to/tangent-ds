import { Transformer } from '../../core/estimators/estimator.js';
import { normalize } from '../../core/table.js';
import * as rda from '../rda.js';
import { normalizeScaling, toLoadingObjects, toScoreObjects } from '../scaling.js';

const DEFAULT_PARAMS = {
  scale: false,
  omit_missing: true,
  scaling: 2,
  constrained: true,
};

function prepareMatricesFromTable({
  data,
  response,
  predictors,
  omit_missing = true
}) {
  if (!data) {
    throw new Error('RDA declarative usage requires a `data` property.');
  }
  if (!response || !predictors) {
    throw new Error('RDA declarative usage requires `response` and `predictors` keys.');
  }

  const responseCols = Array.isArray(response) ? response : [response];
  const predictorCols = Array.isArray(predictors) ? predictors : [predictors];
  const rows = normalize(data);

  const filtered = omit_missing
    ? rows.filter(
        (row) =>
          responseCols.every(
            (c) => typeof row[c] === 'number' && !Number.isNaN(row[c])
          ) &&
          predictorCols.every(
            (c) => typeof row[c] === 'number' && !Number.isNaN(row[c])
          )
      )
    : rows;

  const Y = filtered.map((row) => responseCols.map((c) => row[c]));
  const X = filtered.map((row) => predictorCols.map((c) => row[c]));

  if (Y.length === 0 || X.length === 0) {
    throw new Error('No valid rows available after filtering missing values.');
  }

  return {
    Y,
    X,
    responseNames: responseCols.map((c) => String(c)),
    predictorNames: predictorCols.map((c) => String(c))
  };
}

export class RDA extends Transformer {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
  }

  fit(Y, X = null, opts = {}) {
    let responses = Y;
    let predictors = X;
    let scale = opts.scale !== undefined ? opts.scale : this.params.scale;
    let scaling = opts.scaling !== undefined ? opts.scaling : this.params.scaling;
    let constrained = opts.constrained !== undefined ? opts.constrained : this.params.constrained;
    let omitMissing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : this.params.omit_missing;

    if (
      Y &&
      typeof Y === 'object' &&
      !Array.isArray(Y) &&
      (Y.data || Y.response || Y.predictors)
    ) {
      const callOpts = { ...DEFAULT_PARAMS, ...this.params, ...Y };
      const prepared = prepareMatricesFromTable({
        data: callOpts.data,
        response: callOpts.response || callOpts.Y,
        predictors: callOpts.predictors || callOpts.X,
        omit_missing: callOpts.omit_missing
      });
      responses = prepared.Y;
      predictors = prepared.X;
      scale = callOpts.scale;
      scaling = callOpts.scaling !== undefined ? callOpts.scaling : scaling;
      constrained = callOpts.constrained !== undefined ? callOpts.constrained : constrained;
      omitMissing = callOpts.omit_missing;
      opts = {
        ...opts,
        responseNames: prepared.responseNames,
        predictorNames: prepared.predictorNames
      };
    }

    if (!responses || !predictors) {
      throw new Error('RDA.fit requires response and predictor matrices.');
    }

    const result = rda.fit(responses, predictors, {
      scale,
      scaling,
      constrained,
      responseNames: opts.responseNames,
      predictorNames: opts.predictorNames
    });
    this.model = { ...result, omit_missing: omitMissing };
    this.params.scale = scale;
    this.params.omit_missing = omitMissing;
    this.params.scaling = normalizeScaling(result.scaling ?? scaling);
    this.params.constrained = result.constrained ?? constrained;
    this.params.constrained = result.constrained ?? constrained;
    this.fitted = true;
    return this;
  }

  transform(Y, X, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('RDA: estimator not fitted. Call fit() before transform().');
    }

    if (
      Y &&
      typeof Y === 'object' &&
      !Array.isArray(Y) &&
      (Y.data || Y.response || Y.predictors)
    ) {
      const callOpts = { ...opts, ...Y };
      const prepared = prepareMatricesFromTable({
        data: callOpts.data,
        response: callOpts.response || callOpts.Y,
        predictors: callOpts.predictors || callOpts.X,
        omit_missing: callOpts.omit_missing !== undefined
          ? callOpts.omit_missing
          : this.params.omit_missing
      });
      return rda.transform(this.model, prepared.Y, prepared.X);
    }

    return rda.transform(this.model, Y, X);
  }

  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('RDA: estimator not fitted.');
    }
    const {
      constrainedVariance,
      eigenvalues,
      varianceExplained,
      n,
      p,
      q,
      scaling,
      constrained
    } = this.model;
    return {
      constrainedVariance,
      eigenvalues,
      varianceExplained,
      samples: n,
      predictors: p,
      responses: q,
      scaling: scaling ?? this.params.scaling,
      constrained: constrained ?? this.params.constrained,
    };
  }

  /**
   * Retrieve site (scores), response loadings, or predictor constraint scores.
   * @param {'sites'|'samples'|'responses'|'variables'|'loadings'|'constraints'|'predictors'} type
   * @param {boolean} [scaled=true]
   */
  getScores(type = 'sites', scaled = true) {
    if (!this.fitted || !this.model) {
      throw new Error('RDA: estimator not fitted. Call fit() before getScores().');
    }

    const {
      scores,
      loadings,
      constraintScores,
      rawScores,
      rawLoadings,
      rawConstraintScores,
      responseNames,
      predictorNames,
    } = this.model;

    const lower = type.toLowerCase();
    if (lower === 'sites' || lower === 'samples') {
      if (scaled) return scores;
      return toScoreObjects(rawScores, 'rda');
    }

    if (lower === 'responses' || lower === 'variables' || lower === 'loadings') {
      if (scaled) return loadings;
      return toLoadingObjects(rawLoadings, responseNames, 'rda');
    }

    if (lower === 'constraints' || lower === 'predictors') {
      if (scaled) return constraintScores;
      return toLoadingObjects(rawConstraintScores, predictorNames, 'rda');
    }

    throw new Error(`Unknown score type "${type}". Expected "sites", "responses", or "constraints".`);
  }

  toJSON() {
    return {
      __class__: 'RDA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new RDA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default RDA;
