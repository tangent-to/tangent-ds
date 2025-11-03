// Deprecated shim: use src/ml/estimators/HCA.js instead
import { HCA as MLHCA } from '../../ml/estimators/HCA.js';

let warned = false;
function warnOnce() {
  if (!warned && typeof console !== 'undefined' && console.warn) {
    console.warn('[tangent-ds] mva.HCA is deprecated. Please use ml.HCA instead.');
    warned = true;
  }
}

export class HCA extends MLHCA {
  constructor(params = {}) {
    warnOnce();
    super(params);
  }

  static fromJSON(obj = {}) {
    warnOnce();
    const inst = new HCA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default HCA;
