/**
 * PolynomialRegressor - scikit-like wrapper around ml/polynomial utilities
 */

import { Regressor } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';
import * as poly from '../polynomial.js';

const DEFAULT_PARAMS = {
  degree: 2,
  intercept: true,
  omit_missing: true
};

export class PolynomialRegressor extends Regressor {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
    this.coef = null;
  }

  fit(X, y = null, opts = {}) {
    let dataX = X;
    let dataY = y;
    let effective = { ...this.params, ...opts };

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.X || X.columns)
    ) {
      const callOpts = { ...DEFAULT_PARAMS, ...this.params, ...X };
      const prepared = prepareXY({
        X: callOpts.X || callOpts.columns,
        y: callOpts.y,
        data: callOpts.data,
        omit_missing: callOpts.omit_missing
      });
      dataX = prepared.X;
      dataY = prepared.y;
      effective = {
        degree: callOpts.degree,
        intercept: callOpts.intercept,
        omit_missing: callOpts.omit_missing
      };
    }

    if (!dataX || !dataY) {
      throw new Error('PolynomialRegressor.fit requires X and y.');
    }

    const result = poly.fit(dataX, dataY, {
      degree: effective.degree,
      intercept: effective.intercept
    });

    this.model = result;
    this.coef = result.coefficients;
    this.fitted = true;
    this.params.degree = result.degree;
    this.params.intercept = effective.intercept;

    return this;
  }

  predict(X, { intercept = undefined } = {}) {
    this._ensureFitted('predict');

    let matrix = X;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.X || X.columns)
    ) {
      const prepared = prepareX({
        columns: X.X || X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined
          ? X.omit_missing
          : this.params.omit_missing
      });
      matrix = prepared.X;
    }

    const useIntercept = intercept === undefined ? this.params.intercept : intercept;
    return poly.predict(
      { ...this.model, degree: this.params.degree, coefficients: this.coef },
      matrix,
      { intercept: useIntercept }
    );
  }

  summary() {
    this._ensureFitted('summary');
    return { ...this.model };
  }

  toJSON() {
    return {
      __class__: 'PolynomialRegressor',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
      coefficients: this.coef
    };
  }

  static fromJSON(obj = {}) {
    const inst = new PolynomialRegressor(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.coef = obj.coefficients || (obj.model && obj.model.coefficients) || null;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default PolynomialRegressor;
