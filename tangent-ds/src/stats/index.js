/**
 * Stats module exports
 */

import { normal, uniform, gamma, beta } from './distribution.js';
import {
  oneSampleTTest as oneSampleTTestFn,
  twoSampleTTest as twoSampleTTestFn,
  chiSquareTest as chiSquareTestFn,
  oneWayAnova as oneWayAnovaFn
} from './tests.js';

import { GLM } from './estimators/GLM.js';
import {
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova
} from './estimators/tests.js';

// Alias classes under camelCase names for ergonomic construction
const oneSampleTTest = OneSampleTTest;
const twoSampleTTest = TwoSampleTTest;
const chiSquareTest = ChiSquareTest;
const oneWayAnova = OneWayAnova;

// Preserve functional helpers grouped under a namespace for direct usage
const hypothesis = {
  oneSampleTTest: oneSampleTTestFn,
  twoSampleTTest: twoSampleTTestFn,
  chiSquareTest: chiSquareTestFn,
  oneWayAnova: oneWayAnovaFn
};

export {
  // Distributions
  normal,
  uniform,
  gamma,
  beta,

  // Hypothesis test helper namespace (functional)
  hypothesis,

  // Generalized Linear Models (GLM and GLMM)
  GLM,

  // Hypothesis test estimator classes
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova,
  oneSampleTTest,
  twoSampleTTest,
  chiSquareTest,
  oneWayAnova
};
