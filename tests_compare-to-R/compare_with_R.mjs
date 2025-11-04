#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { pca, lda, rda } from '../tangent-ds/src/mva/index.js';
import { prepareX } from '../tangent-ds/src/core/table.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const dataset = [
  { species: 'Adelie', island: 'Torgersen', bill_length_mm: 39.1, bill_depth_mm: 18.7, flipper_length_mm: 181, body_mass_g: 3750 },
  { species: 'Adelie', island: 'Torgersen', bill_length_mm: 39.5, bill_depth_mm: 17.4, flipper_length_mm: 186, body_mass_g: 3800 },
  { species: 'Adelie', island: 'Dream', bill_length_mm: 40.3, bill_depth_mm: 18.0, flipper_length_mm: 195, body_mass_g: 3250 },
  { species: 'Adelie', island: 'Dream', bill_length_mm: 36.7, bill_depth_mm: 19.3, flipper_length_mm: 193, body_mass_g: 3450 },
  { species: 'Adelie', island: 'Biscoe', bill_length_mm: 38.9, bill_depth_mm: 17.8, flipper_length_mm: 181, body_mass_g: 3625 },
  { species: 'Chinstrap', island: 'Dream', bill_length_mm: 46.5, bill_depth_mm: 17.9, flipper_length_mm: 195, body_mass_g: 3500 },
  { species: 'Chinstrap', island: 'Dream', bill_length_mm: 50.0, bill_depth_mm: 19.5, flipper_length_mm: 196, body_mass_g: 3900 },
  { species: 'Chinstrap', island: 'Dream', bill_length_mm: 51.3, bill_depth_mm: 18.2, flipper_length_mm: 193, body_mass_g: 3775 },
  { species: 'Chinstrap', island: 'Dream', bill_length_mm: 45.4, bill_depth_mm: 17.0, flipper_length_mm: 190, body_mass_g: 3250 },
  { species: 'Chinstrap', island: 'Dream', bill_length_mm: 52.7, bill_depth_mm: 19.8, flipper_length_mm: 197, body_mass_g: 3725 },
  { species: 'Gentoo', island: 'Biscoe', bill_length_mm: 46.1, bill_depth_mm: 13.2, flipper_length_mm: 211, body_mass_g: 4500 },
  { species: 'Gentoo', island: 'Biscoe', bill_length_mm: 50.0, bill_depth_mm: 16.3, flipper_length_mm: 230, body_mass_g: 5700 },
  { species: 'Gentoo', island: 'Biscoe', bill_length_mm: 48.7, bill_depth_mm: 14.1, flipper_length_mm: 210, body_mass_g: 4450 },
  { species: 'Gentoo', island: 'Biscoe', bill_length_mm: 45.2, bill_depth_mm: 14.8, flipper_length_mm: 215, body_mass_g: 5000 },
  { species: 'Gentoo', island: 'Biscoe', bill_length_mm: 49.5, bill_depth_mm: 15.6, flipper_length_mm: 222, body_mass_g: 5250 },
];

const numericColumns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'];
const classColumn = 'species';
const responseColumns = numericColumns;
const predictorColumns = ['species', 'island'];

const payload = {
  data: dataset,
  numericColumns,
  classColumn,
  responseColumns,
  predictorColumns,
  pca: { center: true, scale: true, scaling: [0, 1, 2] },
  lda: { scale: false, scaling: [0, 1, 2] },
  rda: { scale: true, scaling: [0, 1, 2] },
};

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tangent-r-'));
const inputPath = path.join(tmpDir, 'input.json');
const outputPath = path.join(tmpDir, 'output.json');
fs.writeFileSync(inputPath, JSON.stringify(payload), 'utf8');

const rScriptPath = path.join(__dirname, 'penguins_reference.R');
const rRun = spawnSync('Rscript', [rScriptPath, inputPath, outputPath], {
  stdio: 'inherit',
});
if (rRun.status !== 0) {
  throw new Error(`Rscript exited with status ${rRun.status}`);
}

const rData = JSON.parse(fs.readFileSync(outputPath, 'utf8'));

function toMatrixFromR(entry) {
  const rows = entry.rows ?? [];
  const columns = entry.columns ?? [];
  const data = entry.data?.map((row) => row.map(Number)) ?? [];
  return { data, rows, columns };
}

function buildScoreMatrix(scoreObjects, columns, { includeClass = false } = {}) {
  const cols = columns.slice();
  return {
    data: scoreObjects.map((row) =>
      cols.map((col) => {
        if (!(col in row)) {
          if (includeClass && col === 'class') {
            return row[col];
          }
          throw new Error(`Missing column ${col} in score object`);
        }
        return Number(row[col]);
      })
    ),
    columns: cols,
  };
}

function buildLoadingMatrix(loadingObjects, columns, rowOrder) {
  const map = new Map(loadingObjects.map((row) => [row.variable, row]));
  const rows = rowOrder ?? Array.from(map.keys());
  const cols = columns.slice();
  const data = rows.map((rowName) => {
    const obj = map.get(rowName);
    if (!obj) {
      throw new Error(`Missing loading for variable ${rowName}`);
    }
    return cols.map((col) => Number(obj[col]));
  });
  return { data, rows, columns: cols };
}

function alignColumns(reference, target) {
  const refCols = reference.columns.length;
  const refRows = reference.data.length;
  const aligned = target.data.map((row) => row.slice());
  for (let j = 0; j < refCols; j++) {
    const refVector = reference.data.map((row) => row[j]);
    const tgtVector = target.data.map((row) => row[j]);
    const dot = refVector.reduce((sum, val, idx) => sum + val * tgtVector[idx], 0);
    const sign = dot < 0 ? -1 : 1;
    for (let i = 0; i < refRows; i++) {
      aligned[i][j] = tgtVector[i] * sign;
    }
  }
  return aligned;
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) max = diff;
  }
  return max;
}

function flattenMatrix(matrix) {
  return matrix.reduce((acc, row) => acc.concat(row), []);
}

function assertClose(label, reference, target, tolerance = 1e-6) {
  if (reference.length !== target.length) {
    throw new Error(`${label}: length mismatch (${reference.length} vs ${target.length})`);
  }
  const maxDiff = maxAbsDiff(reference, target);
  if (maxDiff > tolerance) {
    throw new Error(`${label}: max absolute difference ${maxDiff} exceeded tolerance ${tolerance}`);
  }
}

function reorderMatrixRows(matrix, currentRows, desiredRows) {
  if (!desiredRows || desiredRows.length === 0) {
    return matrix;
  }
  const indexMap = new Map(currentRows.map((name, idx) => [name, idx]));
  const data = desiredRows.map((name) => {
    if (!indexMap.has(name)) {
      throw new Error(`Row ${name} not found when reordering`);
    }
    return matrix.data[indexMap.get(name)];
  });
  return { data, rows: desiredRows.slice(), columns: matrix.columns.slice() };
}

function reorderMatrixColumns(matrix, desiredColumns) {
  const colIndex = new Map(matrix.columns.map((name, idx) => [name, idx]));
  const data = matrix.data.map((row) =>
    desiredColumns.map((name) => {
      if (!colIndex.has(name)) {
        throw new Error(`Column ${name} not found when reordering`);
      }
      return row[colIndex.get(name)];
    })
  );
  return { data, rows: matrix.rows ? matrix.rows.slice() : undefined, columns: desiredColumns.slice() };
}

function comparePCA() {
  const tolerance = 1e-6;
  for (const sc of payload.pca.scaling) {
    const model = pca.fit({
      data: dataset,
      columns: numericColumns,
      scale: payload.pca.scale,
      center: payload.pca.center,
      omit_missing: false,
      scaling: sc,
    });

    const rEntry = rData.pca.scalings[`scaling_${sc}`];
    if (!rEntry) {
      throw new Error(`Missing R PCA results for scaling ${sc}`);
    }

    const rScores = toMatrixFromR(rEntry.scores);
    const rLoadings = toMatrixFromR(rEntry.loadings);
    const nodeScoreMatrix = buildScoreMatrix(model.scores, rScores.columns);
    const nodeLoadings = buildLoadingMatrix(model.loadings, rLoadings.columns, rLoadings.rows);

    const nodeScoresAligned = alignColumns(rScores, nodeScoreMatrix);
    const nodeLoadingsAligned = alignColumns(rLoadings, nodeLoadings);

    assertClose(
      `PCA scores (scaling ${sc})`,
      flattenMatrix(rScores.data),
      flattenMatrix(nodeScoresAligned),
      tolerance
    );

    assertClose(
      `PCA loadings (scaling ${sc})`,
      flattenMatrix(rLoadings.data),
      flattenMatrix(nodeLoadingsAligned),
      tolerance
    );

    assertClose(
      'PCA eigenvalues',
      rData.pca.eigenvalues,
      model.eigenvalues.map(Number),
      tolerance
    );
  }
}

function compareLDA() {
  const tolerance = 1e-6;
  for (const sc of payload.lda.scaling) {
    const model = lda.fit({
      X: numericColumns,
      y: classColumn,
      data: dataset,
      omit_missing: false,
      scale: payload.lda.scale,
      scaling: sc,
    });

    const rEntry = rData.lda.scalings[`scaling_${sc}`];
    if (!rEntry) {
      throw new Error(`Missing R LDA results for scaling ${sc}`);
    }

    const rScores = toMatrixFromR(rEntry.scores);
    const rLoadings = toMatrixFromR(rEntry.loadings);

    const scoreColumns = rScores.columns;
    const nodeScoreMatrixRaw = buildScoreMatrix(
      model.scores.map(({ class: _class, ...rest }) => rest),
      scoreColumns
    );

    const nodeScoresAligned = alignColumns(rScores, nodeScoreMatrixRaw);
    assertClose(
      `LDA scores (scaling ${sc})`,
      flattenMatrix(rScores.data),
      flattenMatrix(nodeScoresAligned),
      tolerance
    );

    const nodeLoadings = buildLoadingMatrix(model.loadings, rLoadings.columns, rLoadings.rows);
    const nodeLoadingsAligned = alignColumns(rLoadings, nodeLoadings);
    assertClose(
      `LDA loadings (scaling ${sc})`,
      flattenMatrix(rLoadings.data),
      flattenMatrix(nodeLoadingsAligned),
      tolerance
    );

    assertClose(
      'LDA eigenvalues',
      rData.lda.eigenvalues,
      model.eigenvalues.map(Number),
      tolerance
    );
  }
}

function compareRDA() {
  const tolerance = 1e-6;
  const responsePrep = prepareX({
    columns: responseColumns,
    data: dataset,
    omit_missing: false,
  });
  const predictorPrep = prepareX({
    columns: predictorColumns,
    data: dataset,
    omit_missing: false,
    encode: {
      species: 'onehot',
      island: 'onehot',
    },
  });

  for (const sc of payload.rda.scaling) {
    const model = rda.fit(responsePrep.X, predictorPrep.X, {
      scale: payload.rda.scale,
      scaling: sc,
      responseNames: responsePrep.columns,
      predictorNames: predictorPrep.columns,
    });

    const rEntry = rData.rda.scalings[`scaling_${sc}`];
    if (!rEntry) {
      throw new Error(`Missing R RDA results for scaling ${sc}`);
    }

    const rSiteScores = toMatrixFromR(rEntry.sites);
    const rSpeciesScores = toMatrixFromR(rEntry.species);

    const nodeScores = {
      data: model.canonicalScores.map((row) =>
        rSiteScores.columns.map((col) => Number(row[col]))
      ),
      columns: rSiteScores.columns,
    };
    const alignedScores = alignColumns(rSiteScores, nodeScores);
    assertClose(
      `RDA site scores (scaling ${sc})`,
      flattenMatrix(rSiteScores.data),
      flattenMatrix(alignedScores),
      tolerance
    );

    const nodeLoadings = buildLoadingMatrix(
      model.canonicalLoadings,
      rSpeciesScores.columns,
      rSpeciesScores.rows
    );
    const alignedLoadings = alignColumns(rSpeciesScores, nodeLoadings);
    assertClose(
      `RDA species scores (scaling ${sc})`,
      flattenMatrix(rSpeciesScores.data),
      flattenMatrix(alignedLoadings),
      tolerance
    );

    assertClose(
      'RDA eigenvalues',
      rData.rda.eigenvalues,
      model.eigenvalues.map(Number),
      tolerance
    );
  }
}

comparePCA();
compareLDA();
compareRDA();

fs.rmSync(tmpDir, { recursive: true, force: true });

console.log('All PCA, LDA, and RDA comparisons against R passed within tolerance.');
