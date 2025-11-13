/**
 * Stats module exports
 */

import { normal, uniform, gamma, beta } from './distribution.js';
import {
  oneSampleTTest as oneSampleTTestFn,
  twoSampleTTest as twoSampleTTestFn,
  chiSquareTest as chiSquareTestFn,
  oneWayAnova as oneWayAnovaFn,
  tukeyHSD as tukeyHSDFn
} from './tests.js';

import { GLM } from './estimators/GLM.js';
import {
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova,
  TukeyHSD
} from './estimators/tests.js';

import {
  compareModels,
  likelihoodRatioTest,
  pairwiseLRT,
  modelSelectionPlot,
  aicWeightPlot,
  coefficientComparisonPlot,
  crossValidate,
  crossValidateModels
} from './model_comparison.js';

// Alias classes under camelCase names for ergonomic construction
const oneSampleTTest = OneSampleTTest;
const twoSampleTTest = TwoSampleTTest;
const chiSquareTest = ChiSquareTest;
const oneWayAnova = OneWayAnova;
const tukeyHSD = TukeyHSD;

// Preserve functional helpers grouped under a namespace for direct usage
const hypothesis = {
  oneSampleTTest: oneSampleTTestFn,
  twoSampleTTest: twoSampleTTestFn,
  chiSquareTest: chiSquareTestFn,
  oneWayAnova: oneWayAnovaFn,
  tukeyHSD: tukeyHSDFn
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

  // Model comparison and selection
  compareModels,
  likelihoodRatioTest,
  pairwiseLRT,
  modelSelectionPlot,
  aicWeightPlot,
  coefficientComparisonPlot,
  crossValidate,
  crossValidateModels,

  // Hypothesis test estimator classes
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova,
  TukeyHSD,
  oneSampleTTest,
  twoSampleTTest,
  chiSquareTest,
  oneWayAnova,
  tukeyHSD
};
