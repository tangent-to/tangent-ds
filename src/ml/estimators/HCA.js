import { Estimator } from '../../core/estimators/estimator.js';
import { prepareX } from '../../core/table.js';
import * as hca from '../hca.js';

const DEFAULT_PARAMS = {
  linkage: 'average',
  omit_missing: true,
  k: null  // Optional: number of clusters to cut dendrogram
};

export class HCA extends Estimator {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
    this.labels = null;
  }

  fit(X, opts = {}) {
    let data = X;
    let linkage = opts.linkage !== undefined ? opts.linkage : this.params.linkage;
    let omitMissing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : this.params.omit_missing;
    let k = opts.k !== undefined ? opts.k : this.params.k;

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
      k = callOpts.k !== undefined ? callOpts.k : k;
    }

    if (!Array.isArray(data)) {
      throw new Error('HCA.fit expects an array of observations or an options object.');
    }

    const result = hca.fit(data, { linkage });
    this.model = { ...result, omit_missing: omitMissing };
    this.params.linkage = linkage;
    this.params.omit_missing = omitMissing;
    this.params.k = k;
    this.fitted = true;

    // Auto-cut if k is specified
    if (k !== null && k !== undefined) {
      this.labels = this.cut(k);
    } else {
      this.labels = null;
    }

    return this;
  }

  cut(k) {
    this._ensureFitted('cut');
    return hca.cut(this.model, k);
  }

  cutHeight(height) {
    this._ensureFitted('cutHeight');
    return hca.cutHeight(this.model, height);
  }

  summary() {
    this._ensureFitted('summary');
    const { linkage, n, dendrogram } = this.model;
    const summary = {
      linkage,
      n,
      merges: dendrogram.length,
      maxDistance: dendrogram.reduce(
        (max, merge) => Math.max(max, merge.distance),
        0
      )
    };

    // Add cluster info if k was specified
    if (this.params.k !== null && this.params.k !== undefined && this.labels) {
      const uniqueLabels = [...new Set(this.labels)];
      summary.k = this.params.k;
      summary.nClusters = uniqueLabels.length;
    }

    return summary;
  }

  toJSON() {
    return {
      __class__: 'HCA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
      labels: this.labels
    };
  }

  static fromJSON(obj = {}) {
    const inst = new HCA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
      inst.labels = obj.labels || null;
    }
    return inst;
  }
}

export default HCA;
