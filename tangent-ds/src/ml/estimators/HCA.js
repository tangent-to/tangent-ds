import { Estimator } from '../../core/estimators/estimator.js';
import { prepareX } from '../../core/table.js';
import * as hca from '../hca.js';

const DEFAULT_PARAMS = {
  linkage: 'average',
  omit_missing: true
};

export class HCA extends Estimator {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
  }

  fit(X, opts = {}) {
    let data = X;
    let linkage = opts.linkage !== undefined ? opts.linkage : this.params.linkage;
    let omitMissing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : this.params.omit_missing;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.columns)
    ) {
      const callOpts = { ...DEFAULT_PARAMS, ...this.params, ...X };
      const prepared = prepareX({
        columns: callOpts.columns || callOpts.X,
        data: callOpts.data,
        omit_missing: callOpts.omit_missing
      });
      data = prepared.X;
      linkage = callOpts.linkage;
      omitMissing = callOpts.omit_missing;
    }

    if (!Array.isArray(data)) {
      throw new Error('HCA.fit expects an array of observations or an options object.');
    }

    const result = hca.fit(data, { linkage });
    this.model = { ...result, omit_missing: omitMissing };
    this.params.linkage = linkage;
    this.params.omit_missing = omitMissing;
    this.fitted = true;
    return this;
  }

  cut(k) {
    if (!this.fitted || !this.model) {
      throw new Error('HCA: estimator not fitted.');
    }
    return hca.cut(this.model, k);
  }

  cutHeight(height) {
    if (!this.fitted || !this.model) {
      throw new Error('HCA: estimator not fitted.');
    }
    return hca.cutHeight(this.model, height);
  }

  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('HCA: estimator not fitted.');
    }
    const { linkage, n, dendrogram } = this.model;
    return {
      linkage,
      n,
      merges: dendrogram.length,
      maxDistance: dendrogram.reduce(
        (max, merge) => Math.max(max, merge.distance),
        0
      )
    };
  }

  toJSON() {
    return {
      __class__: 'HCA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new HCA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default HCA;
