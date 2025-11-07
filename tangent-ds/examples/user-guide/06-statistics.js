// ---
// title: Statistics with Palmer Penguins
// id: stats-penguins
// ---

import * as ds from '@tangent.to/ds';

const fetchJson = async (url) => (await fetch(url)).json();

globalThis.penguinsRaw = await fetchJson('https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json');

const numericCols = [
  'Beak Length (mm)',
  'Beak Depth (mm)',
  'Flipper Length (mm)',
  'Body Mass (g)'
];

globalThis.penguinsClean = penguinsRaw.filter((row) =>
  row.species && row.sex && row.island &&
  numericCols.every((col) => row[col] != null)
);

console.log(`Samples after cleaning: ${globalThis.penguinsClean.length}`);

// -- Descriptive stats --
const beakLength = penguinsClean.map((p) => p['Beak Length (mm)']);
const meanBeak = ds.core.math.mean(beakLength, { naOmit: true });
const sdBeak = ds.core.math.stddev(beakLength, true, { naOmit: true });
console.log({ meanBeak, sdBeak });

// -- Hypothesis tests --
const bodyMassBySpecies = (species) => penguinsClean
  .filter((p) => p.species === species)
  .map((p) => p['Body Mass (g)']);

const gentooBodyMass = bodyMassBySpecies('Gentoo');
const adelieBodyMass = bodyMassBySpecies('Adelie');

const ttest = ds.stats.hypothesis.twoSampleTTest(gentooBodyMass, adelieBodyMass);
console.log('Two-sample t-test Gentoo vs Adelie:', ttest);

const anovaMass = ds.stats.hypothesis.oneWayAnova(
  penguinsClean.map((p) => p['Body Mass (g)']),
  penguinsClean.map((p) => p.species)
);
console.log('One-way ANOVA mass ~ species:', anovaMass);

// -- Linear model --
const lm = new ds.stats.GLM({ family: 'gaussian' }{
  formula: 'BodyMass ~ Species + Sex + FlipperLength'
});

await lm.fit(penguinsClean.map((p) => ({
  BodyMass: p['Body Mass (g)'],
  Species: p.species,
  Sex: p.sex,
  FlipperLength: p['Flipper Length (mm)']
})));

console.log('Linear model coefficients:', lm.coefficients());

// -- Logistic regression --
const logit = new ds.stats.GLM({ family: 'binomial', link: 'logit' }{
  formula: 'Sex ~ BodyMass + BeakLength + Species'
});

await logit.fit(penguinsClean.map((p) => ({
  Sex: p.sex === 'female' ? 1 : 0,
  BodyMass: p['Body Mass (g)'],
  BeakLength: p['Beak Length (mm)'],
  Species: p.species
})));

console.log('Logistic regression summary:', logit.summary());

// -- Mixed-effects model demonstration (simple random intercept) --
const lmm = new ds.stats.GLM({
  family: 'gaussian',
  formula: 'BodyMass ~ Species + (1 | Island)'
});

await lmm.fit(penguinsClean.map((p) => ({
  BodyMass: p['Body Mass (g)'],
  Species: p.species,
  Island: p.island
})));

console.log('Mixed model summary (random intercept for island):', lmm.summary());

console.log('Tip: For richer random-effects structures, use Vega datasets such as gapminder.json (country x année) ou seattle-weather.csv (mesures journalières).');
