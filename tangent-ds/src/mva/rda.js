/**
 * Redundancy Analysis (RDA)
 * Constrained ordination - PCA on fitted values from multiple regression
 */

import { toMatrix, solveLeastSquares, Matrix } from '../core/linalg.js';
import * as pca from './pca.js';
import { mean } from '../core/math.js';
import { prepareX } from '../core/table.js';
import {
  normalizeScaling,
  scaleOrdination,
  scaleConstraintScores,
  toScoreObjects,
  toLoadingObjects,
  eigenvaluePowers,
} from './scaling.js';

/**
 * Fit RDA model.
 *
 * @param {Array<Array<number>>} Y - Response matrix (n x q)
 * @param {Array<Array<number>>} X - Explanatory matrix (n x p)
 * @param {Object} options
 * @param {boolean} [options.scale=false] - Standardise response variables before regression.
 * @param {boolean} [options.constrained=true] - When true, perform PCA on fitted values (constrained ordination); when false, perform PCA on residuals (unconstrained ordination).
 * @returns {Object} RDA model
 */
export function fit(Y, X, options = {}) {
  let scale = options.scale ?? false;
  let scaling = options.scaling ?? 2;
  let constrained = options.constrained ?? true;
  let responseMatrix = Y;
  let predictorMatrix = X;
  let responseNames = Array.isArray(options.responseNames)
    ? options.responseNames.map((name) => String(name))
    : null;
  let predictorNames = Array.isArray(options.predictorNames)
    ? options.predictorNames.map((name) => String(name))
    : null;

  if (
    Y &&
    typeof Y === "object" &&
    !Array.isArray(Y) &&
    (Y.data || Y.response || Y.responses || Y.predictors || Y.Y)
  ) {
    const opts = Y;
    const data = opts.data;
    const responseCols = opts.response || opts.responses || opts.Y;
    const predictorCols = opts.predictors || opts.X;
    if (!data || !responseCols || !predictorCols) {
      throw new Error("RDA.fit requires data, response columns, and predictor columns.");
    }
    const omitMissing = opts.omit_missing !== undefined ? opts.omit_missing : true;
    scale = opts.scale !== undefined ? opts.scale : scale;
    scaling = opts.scaling !== undefined ? opts.scaling : scaling;
    constrained = opts.constrained !== undefined ? opts.constrained : constrained;

    const responseList = Array.isArray(responseCols) ? responseCols : [responseCols];
    const predictorList = Array.isArray(predictorCols) ? predictorCols : [predictorCols];

    const responsePrepInitial = prepareX({
      columns: responseList,
      data,
      omit_missing: omitMissing,
    });
    const predictorPrep = prepareX({
      columns: predictorList,
      data: responsePrepInitial.rows,
      omit_missing: omitMissing,
    });
    const responsePrepAligned = prepareX({
      columns: responseList,
      data: predictorPrep.rows,
      omit_missing: false,
    });

    responseMatrix = responsePrepAligned.X;
    predictorMatrix = predictorPrep.X;
    responseNames = responsePrepAligned.columns.map((name) => String(name));
    predictorNames = predictorPrep.columns.map((name) => String(name));
  }

  if (!responseMatrix || !predictorMatrix) {
    throw new Error('RDA.fit requires response and predictor matrices.');
  }

  const appliedScaling = normalizeScaling(scaling);

  const responseData = responseMatrix.map(row => Array.isArray(row) ? row : [row]);
  const explData = predictorMatrix.map(row => Array.isArray(row) ? row : [row]);

  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;

  if (n !== explData.length) {
    throw new Error('Y and X must have same number of rows');
  }

  if (n < p + 2) {
    throw new Error('Need more samples than explanatory variables');
  }

  const YMeans = [];
  const XMeans = [];

  for (let j = 0; j < q; j++) {
    const col = responseData.map(row => row[j]);
    YMeans.push(mean(col));
  }

  for (let j = 0; j < p; j++) {
    const col = explData.map(row => row[j]);
    XMeans.push(mean(col));
  }

  let YCentered = responseData.map(row =>
    row.map((val, j) => val - YMeans[j])
  );

  const XCentered = explData.map(row =>
    row.map((val, j) => val - XMeans[j])
  );

  // Apply scaling to Y if requested
  let YSds = null;
  if (scale) {
    YSds = [];
    for (let j = 0; j < q; j++) {
      const col = YCentered.map(row => row[j]);
      const sd = col.reduce((sum, val) => sum + val * val, 0) / n;
      YSds.push(Math.sqrt(sd));
    }
    YCentered = YCentered.map(row =>
      row.map((val, j) => YSds[j] > 0 ? val / YSds[j] : 0)
    );
  }

  const YFitted = [];
  const YResiduals = [];
  const coefficients = [];

  for (let j = 0; j < q; j++) {
    const yCol = YCentered.map(row => row[j]);

    const XMat = new Matrix(XCentered);
    const yVec = Matrix.columnVector(yCol);
    const betaVec = solveLeastSquares(XMat, yVec);
    const beta = betaVec.to1DArray();

    coefficients.push(beta);

    const fitted = [];
    const residuals = [];
    for (let i = 0; i < n; i++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * beta[k];
      }
      fitted.push(yhat);
      residuals.push(yCol[i] - yhat);
    }

    YFitted.push(fitted);
    YResiduals.push(residuals);
  }

const fittedMatrix = [];
const residualMatrix = [];
for (let i = 0; i < n; i++) {
  const fittedRow = [];
  const residualRow = [];
  for (let j = 0; j < q; j++) {
    fittedRow.push(YFitted[j][i]);
    residualRow.push(YResiduals[j][i]);
  }
  fittedMatrix.push(fittedRow);
  residualMatrix.push(residualRow);
}

const targetMatrix = constrained ? fittedMatrix : residualMatrix;

const pcaModel = pca.fit(targetMatrix, {
    scale: false,
    center: false,
    scaling: appliedScaling,
    columns: responseNames || undefined,
  });

  const rawSiteMatrix = pcaModel.rawScores;
  const rawLoadingsMatrix = pcaModel.rawLoadings;

  const ordination = scaleOrdination({
    rawSites: rawSiteMatrix,
    rawLoadings: rawLoadingsMatrix,
    eigenvalues: pcaModel.eigenvalues,
    singularValues: pcaModel.singularValues,
    scaling: appliedScaling,
  });

  const responseNamesFinal = responseNames && responseNames.length === rawLoadingsMatrix.length
    ? responseNames
    : Array.from({ length: rawLoadingsMatrix.length }, (_, idx) => responseNames?.[idx] ?? `Resp${idx + 1}`);

  const predictorNamesFinal = predictorNames && predictorNames.length === p
    ? predictorNames
    : Array.from({ length: p }, (_, idx) => predictorNames?.[idx] ?? `Pred${idx + 1}`);

  const scoresObjects = toScoreObjects(ordination.scores, 'rda');
  const loadingsObjects = toLoadingObjects(ordination.loadings, responseNamesFinal, 'rda');

let rawConstraintMatrix = [];
let scaledConstraintMatrix = [];
let constraintObjects = [];
if (constrained) {
  rawConstraintMatrix = solveLeastSquares(XCentered, rawSiteMatrix).to2DArray();
  scaledConstraintMatrix = scaleConstraintScores(rawConstraintMatrix, {
    loadingFactors: ordination.loadingFactors,
    eigenvalues: pcaModel.eigenvalues,
    scaling: appliedScaling,
  });
  constraintObjects = toLoadingObjects(scaledConstraintMatrix, predictorNamesFinal, 'rda');
}

  let totalInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      totalInertia += YCentered[i][j] ** 2;
    }
  }
  totalInertia /= n;

  let explainedInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      explainedInertia += fittedMatrix[i][j] ** 2;
    }
  }
  explainedInertia /= n;

  const constrainedVariance = explainedInertia / totalInertia;

const predictorCorrelations = constraintObjects;

const model = {
  scores: scoresObjects,
  loadings: loadingsObjects,
  constraintScores: constraintObjects,
  eigenvalues: pcaModel.eigenvalues,
  varianceExplained: pcaModel.varianceExplained,
  constrainedVariance,
  coefficients,
  YMeans,
  XMeans,
  n,
  p,
  q,
  rawScores: rawSiteMatrix,
  rawLoadings: rawLoadingsMatrix,
  rawFitted: fittedMatrix,
  rawResiduals: residualMatrix,
  rawConstraintScores: rawConstraintMatrix,
  siteFactors: ordination.siteFactors,
  loadingFactors: ordination.loadingFactors,
  scaling: appliedScaling,
  exponent: ordination.exponent,
  singularValues: pcaModel.singularValues,
  components: pcaModel.components,
  constrained: !!constrained,
};

  model.responseNames = responseNamesFinal;
  model.predictorNames = predictorNamesFinal;

  model.canonicalScores = scoresObjects;
  model.canonicalLoadings = loadingsObjects;
  model.predictorCorrelations = predictorCorrelations;

  return model;
}

/**
 * Transform new data using fitted RDA model
 * @param {Object} model - Fitted RDA model
 * @param {Array<Array<number>>} Y - New response data
 * @param {Array<Array<number>>} X - New explanatory data
 * @returns {Array<Object>} Canonical scores
 */
export function transform(model, Y, X) {
  const {
    coefficients,
    YMeans,
    XMeans,
    components,
    singularValues,
    siteFactors,
    scaling,
    eigenvalues,
  } = model;
  
  const responseData = Y.map(row => Array.isArray(row) ? row : [row]);
  const explData = X.map(row => Array.isArray(row) ? row : [row]);
  
  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;
  
  // Center data
  const YCentered = responseData.map(row => 
    row.map((val, j) => val - YMeans[j])
  );
  
  const XCentered = explData.map(row => 
    row.map((val, j) => val - XMeans[j])
  );
  
  // Compute fitted values
  const fittedMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < q; j++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * coefficients[j][k];
      }
      row.push(yhat);
    }
    fittedMatrix.push(row);
  }
  
  // Extract loading matrix
  const nAxes = components.length;

  const baseScores = [];
  for (const row of fittedMatrix) {
    const entry = [];
    for (let j = 0; j < nAxes; j++) {
      let sum = 0;
      for (let k = 0; k < q; k++) {
        sum += row[k] * components[j][k];
      }
      entry.push(sum);
    }
    baseScores.push(entry);
  }

  const rawScores = baseScores.map((row) =>
    row.map((val, idx) => {
      const sv = singularValues[idx] ?? 0;
      return sv === 0 ? 0 : val / sv;
    })
  );

  const exponent = scaling === 1 ? 0.5 : 0;
  const siteScaling = siteFactors && siteFactors.length
    ? siteFactors
    : eigenvaluePowers(eigenvalues, exponent);
  const scaledScores = rawScores.map((row) =>
    row.map((val, idx) => val * (siteScaling[idx] ?? 1))
  );
  
  return toScoreObjects(scaledScores, 'rda');
}
