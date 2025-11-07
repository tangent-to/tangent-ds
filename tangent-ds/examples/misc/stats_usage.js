// ---
// title: Statistics with tangent-ds
// id: statistics-usage
// ---

// %% [markdown]
/*
# Statistical analysis examples for @tangent.to/ds

This notebook uses real datasets from Vega Datasets: https://cdn.jsdelivr.net/npm/vega-datasets@2/data/
*/

// %% [javascript]
import * as aq from "arquero";
import * as ds from "@tangent.to/ds";

// %% [markdown]
/*
## Example 1: Ordinray least square linear regression

The cars dataset contains various attributes of different car models, including weight and miles per gallon (MPG). We will assess the correlation effect of weight on miles per gallon using linear regression.

We fist fetch Fetch cars dataset.
*/

// %% [javascript]
globalThis.cars = aq.fromJSON(
  await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json")
    .then((r) => r.json()),
);

// %% [markdown]
/*
There might be missing values in the dataset, so we filter out rows with missing weight or MPG values. We then set up the predictor variable (weight) and response variable (MPG) for linear regression.
*/

// %% [javascript]
const lr = new ds.stats.GLM({ family: 'gaussian', intercept: true });
lr.fit({
  X: ["Miles_per_Gallon"],
  y: "Weight_in_lbs",
  data: globalThis.cars,
  omit_missing: true,
});
const model = lr.model;

console.log("Predicting MPG from car weight");
console.log("Model coefficients:", model.coefficients.map((c) => c.toFixed(4)));
console.log("R-squared:", model.rSquared.toFixed(4));
console.log(
  "First 5 fitted values:",
  model.fitted.slice(0, 5).map((v) => v.toFixed(2)),
);
console.log(
  "Interpretation: Each 1000 lbs increases MPG by",
  model.coefficients[1].toFixed(2),
);

// %% [javascript]
// Example 2: Two-Sample T-Test - Penguins Dataset
const penguinsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json",
);
const penguinsData = await penguinsResponse.json();
globalThis.penguinsData = penguinsData;
console.log(
  aq.from(penguinsData)
    .columnNames(),
);

// %%
// Compare body mass between Adelie and Gentoo penguins
const adelie = penguinsData
  .filter((p) => p.Species === "Adelie" && p["Body Mass (g)"])
  .map((p) => p["Body Mass (g)"]);

const chinstrap = penguinsData
  .filter((p) => p.Species === "Chinstrap" && p["Body Mass (g)"])
  .map((p) => p["Body Mass (g)"]);

const gentoo = penguinsData
  .filter((p) => p.Species === "Gentoo" && p["Body Mass (g)"])
  .map((p) => p["Body Mass (g)"]);

const ttestEstimator = new ds.stats.twoSampleTTest();
ttestEstimator.fit(adelie, gentoo);
const ttest = ttestEstimator.summary();
console.log("Comparing body mass: Adelie vs Gentoo");
console.log("Adelie mean:", ttest.mean1.toFixed(2), "g");
console.log("Gentoo mean:", ttest.mean2.toFixed(2), "g");
console.log("t-statistic:", ttest.statistic.toFixed(4));
console.log("p-value:", ttest.pValue.toFixed(6));
console.log(
  "Conclusion:",
  ttest.pValue < 0.05
    ? "Significantly different"
    : "Not significantly different",
);

// %% [javascript]
// Compare petal length across three species
const anovaEstimator = new ds.stats.oneWayAnova();
anovaEstimator.fit([adelie, chinstrap, gentoo]);
const anova = anovaEstimator.summary();
console.log("Comparing body mass across species");
console.log("F-statistic:", anova.statistic.toFixed(4));
console.log("p-value:", anova.pValue.toFixed(6));
console.log(
  "Conclusion:",
  anova.pValue < 0.001
    ? "Highly significant differences"
    : "No significant differences",
);

// %% [javascript]
// Example 4: Normal Distribution
console.log("\n📐 Example 4: Normal Distribution");
console.log("-".repeat(50));
const z95 = ds.stats.normal.quantile(0.975);
const z99 = ds.stats.normal.quantile(0.995);
console.log("95% quantile (1.96):", z95.toFixed(4));
console.log("99% quantile (2.58):", z99.toFixed(4));
console.log("P(Z < 1.96):", ds.stats.normal.cdf(1.96).toFixed(4));

// %% [javascript]
// Example 5: Working with Table Data - Movies Dataset
console.log("\n📋 Example 5: Table Data Operations - Movies Dataset");
console.log("-".repeat(50));
const moviesResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/movies.json",
);
const moviesData = await moviesResponse.json();

// Filter valid data
const validMovies = moviesData.filter((m) =>
  m["IMDB Rating"] && m["Rotten Tomatoes Rating"]
);

const imdbVector = ds.core.table.toVector(
  validMovies.slice(0, 100),
  "IMDB Rating",
);
const rtVector = ds.core.table.toVector(
  validMovies.slice(0, 100),
  "Rotten Tomatoes Rating",
);
console.log(
  "IMDB ratings (first 5):",
  imdbVector.slice(0, 5).map((v) => v.toFixed(1)),
);
console.log("RT ratings (first 5):", rtVector.slice(0, 5));
console.log("Mean IMDB rating:", ds.core.math.mean(imdbVector).toFixed(2));
console.log("Mean RT rating:", ds.core.math.mean(rtVector).toFixed(2));

// %% [javascript]
// Example 6: Logistic Regression - Weather Dataset
const weatherResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/seattle-weather.csv",
);
const weatherText = await weatherResponse.text();
const weatherData = weatherText.split("\n").slice(1).map((line) => {
  const parts = line.split(",");
  return {
    precipitation: parseFloat(parts[1]),
    temp_max: parseFloat(parts[2]),
    weather: parts[4],
  };
}).filter((d) => d.precipitation >= 0 && d.temp_max);

weatherData;

// %%

// Use declarative table-style API for logistic regression
//
const weatherHasRain = weatherData.map((d) => ({
  ...d,
  hasRain: d.precipitation > 0 ? 1 : 0,
}));
const logitEstimator = new ds.stats.GLM({
  family: 'binomial',
  link: 'logit',
  maxIter: 100,
  intercept: true
});
logitEstimator.fit({
  X: ["temp_max"],
  y: "hasRain",
  data: weatherHasRain,
  omit_missing: true,
});
const logitModel = logitEstimator.model;
console.log("Predicting rain from max temperature");
console.log("Coefficients:", logitModel.coefficients.map((c) => c.toFixed(4)));
console.log(
  "Fitted probabilities (first 5):",
  logitModel.fitted.slice(0, 5).map((p) => p.toFixed(4)),
);
console.log("Converged:", logitModel.converged);

// %% [javascript]
// Example 7: Linear Mixed Model - Barley Dataset
console.log("\n🔀 Example 7: Linear Mixed Model - Barley Dataset");
console.log("-".repeat(50));
const barleyResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/barley.json",
);
const barleyData = await barleyResponse.json();

// Use year as predictor, site as random effect
const barleyFiltered = barleyData.slice(0, 30);
const Xmixed = barleyFiltered.map((d) => [d.year === 1931 ? 0 : 1]);
const ymixed = barleyFiltered.map((d) => d.yield);
const groups = barleyFiltered.map((d) => d.site);

const lmmEstimator = new ds.stats.GLM({
  family: 'gaussian',
  randomEffects: { intercept: groups }
});
lmmEstimator.fit(Xmixed, ymixed);
const lmmModel = lmmEstimator.model;
console.log("Modeling barley yield by year with site random effects");
console.log("Fixed effects:", lmmModel.fixedEffects.map((e) => e.toFixed(4)));
console.log("Random effects:", lmmModel.randomEffects.map((e) => e.toFixed(4)));
console.log("Residual variance:", lmmModel.varResidual.toFixed(4));
console.log("Random variance:", lmmModel.varRandom.toFixed(4));
