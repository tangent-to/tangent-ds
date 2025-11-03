// Deprecated shim: use src/ml/hca.js instead
import { fit as mlFit, cut as mlCut, cutHeight as mlCutHeight } from '../ml/hca.js';

let warned = false;
function warnOnce() {
  if (!warned && typeof console !== 'undefined' && console.warn) {
    console.warn('[tangent-ds] mva.hca is deprecated. Please import from ml.hca instead.');
    warned = true;
  }
}

export function fit(...args) {
  warnOnce();
  return mlFit(...args);
}

export function cut(...args) {
  warnOnce();
  return mlCut(...args);
}

export function cutHeight(...args) {
  warnOnce();
  return mlCutHeight(...args);
}

export default {
  fit,
  cut,
  cutHeight
};
