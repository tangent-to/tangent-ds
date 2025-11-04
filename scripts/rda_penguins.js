import { OneHotEncoder } from '../tangent-ds/src/core/table.js';
import { RDA } from '../tangent-ds/src/mva/index.js';
import fetch from 'node-fetch';

const res = await fetch('https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json');
const penguins = await res.json();

const predictors = [
  'Beak Length (mm)',
  'Beak Depth (mm)',
  'Body Mass (g)',
  'Flipper Length (mm)'
];
const responseLevels = [
  'species_Adelie',
  'species_Chinstrap',
  'species_Gentoo'
];

const clean = penguins.filter((row) =>
  row.species && row.sex && row.island &&
  predictors.every((col) => row[col] != null && Number.isFinite(row[col]))
);

const encoder = new OneHotEncoder();
const encoded = encoder.fitTransform({
  data: clean,
  columns: ['species', 'sex', 'island']
});

const merged = clean.map((row, i) => ({
  ...row,
  ...encoded[i]
}));

console.log('rows', merged.length);

const rda = new RDA({ scale: true, scaling: 2, omit_missing: true });

rda.fit({
  data: merged,
  response: responseLevels,
  predictors: [
    'Beak Length (mm)',
    'Beak Depth (mm)',
    'Body Mass (g)',
    'Flipper Length (mm)',
    'sex_female',
    'sex_male',
    'island_Biscoe',
    'island_Dream',
    'island_Torgersen'
  ]
});

console.log(rda.model.eigenvalues.slice(0,3));
