/**
 * Visualization Example
 * Demonstrates plot configuration generators for multivariate analysis
 * Using Tangent Notebook format
 * Uses real datasets from Vega Datasets: https://cdn.jsdelivr.net/npm/vega-datasets@2/data/
 */

import { mva, ml, plot } from "@tangent.to/ds";

// ## Generate PCA Dataset - Penguins Dataset
console.log("=== PCA Visualization - Penguins Dataset ===\n");

// Fetch penguins dataset
const penguinsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json",
);
const penguinsData = await penguinsResponse.json();

// Filter complete cases for PCA (use bill + flipper + mass)
const validPCA = penguinsData.filter((d) =>
  d.bill_length_mm != null &&
  d.bill_depth_mm != null &&
  d.flipper_length_mm != null &&
  d.body_mass_g != null &&
  !isNaN(d.bill_length_mm) &&
  !isNaN(d.bill_depth_mm) &&
  !isNaN(d.flipper_length_mm) &&
  !isNaN(d.body_mass_g)
).slice(0, 150);

// Use four measurements for PCA
const pcaData = validPCA.map((d) => [
  d.bill_length_mm,
  d.bill_depth_mm,
  d.flipper_length_mm,
  d.body_mass_g,
]);

const pcaEstimator = new mva.PCA({ center: true, scale: true });
pcaEstimator.fit(pcaData);
const pcaResult = pcaEstimator.model;
console.log("PCA completed on Penguins dataset");
console.log(
  "Variance explained:",
  pcaResult.varianceExplained.map((v) => (v * 100).toFixed(1) + "%").join(", "),
);

// Map species to numeric values for color coding
// Species in the dataset: "Adelie", "Chinstrap", "Gentoo"
const speciesMap = { "Adelie": 0, "Chinstrap": 1, "Gentoo": 2 };
const colorBy = validPCA.map((d) => speciesMap[d.species] ?? -1);

// Generate PCA plot config
const pcaPlot = plot.plotPCA(pcaResult, {
  colorBy: colorBy,
  colorLegend: {
    labels: ["Adelie", "Chinstrap", "Gentoo"],
  },
  showLoadings: true,
  width: 640,
  height: 400,
});

console.log("\n[Plot Config] PCA Biplot - Penguin Species");
console.log("Type:", pcaPlot.type);
console.log("Dimensions:", `${pcaPlot.width}x${pcaPlot.height}`);
console.log("Data keys:", Object.keys(pcaPlot.data));
console.log("Number of marks:", pcaPlot.marks.length);

// Generate Scree plot
const screePlot = plot.plotScree(pcaResult, { width: 640, height: 300 });
console.log("\n[Plot Config] Scree Plot");
console.log("Type:", screePlot.type);
console.log("Components:", screePlot.data.components.length);

// ## LDA Visualization - Penguins Dataset
console.log("\n\n=== LDA Visualization - Penguins Dataset ===\n");

// Prepare features and labels for LDA using two measurements
const validLDA = penguinsData.filter((d) =>
  d.flipper_length_mm != null &&
  d.bill_depth_mm != null &&
  d.species
).slice(0, 150);

const ldaX = validLDA.map((d) => [d.flipper_length_mm, d.bill_depth_mm]);
const ldaY = validLDA.map((d) => d.species);

const ldaEstimator = new mva.LDA();
ldaEstimator.fit(ldaX, ldaY);
const ldaResult = ldaEstimator.model;
console.log("LDA completed on Penguins dataset");
console.log(
  "Number of discriminants:",
  ldaResult.scores[0] && ldaResult.scores[0].ld2 !== undefined ? 2 : 1,
);
console.log("Classes:", ldaResult.classes);

const ldaPlot = plot.plotLDA(ldaResult, {
  colorBy: ldaResult.scores.map((s) => speciesMap[s.class] ?? -1),
  width: 640,
  height: 400,
});

console.log("\n[Plot Config] LDA - Species Discrimination");
console.log("Type:", ldaPlot.type);
console.log("Dimensions:", `${ldaPlot.width}x${ldaPlot.height}`);
console.log("Axes:", ldaPlot.axes);

// ## HCA Visualization with Dendrogram - Penguins Dataset
console.log("\n\n=== HCA Dendrogram - Penguins Dataset ===\n");

// Use bill measurements for clustering (subset for clarity)
const validHCA = penguinsData.filter((p) =>
  p.bill_length_mm != null && p.bill_depth_mm != null &&
  !isNaN(p.bill_length_mm) && !isNaN(p.bill_depth_mm)
).slice(0, 20);

const hcaData = validHCA.map((p) => [
  p.bill_length_mm,
  p.bill_depth_mm,
]);

const hcaEstimator = new ml.HCA({ linkage: "complete" });
hcaEstimator.fit(hcaData);
const hcaResult = hcaEstimator.model;
const hcaSummary = hcaEstimator.summary();
console.log("HCA completed on Penguin bill measurements");
console.log("Linkage method:", hcaSummary.linkage);
console.log("Number of merges:", hcaSummary.merges);

const hcaPlot = plot.plotHCA(hcaResult);
console.log("\n[Plot Config] Dendrogram - Penguin Bill Clustering");
console.log("Type:", hcaPlot.type);
console.log("Linkage:", hcaPlot.config.linkage);
console.log("Orientation:", hcaPlot.config.orientation);

// Generate dendrogram layout
const dendroLayout = plot.dendrogramLayout(hcaPlot, {
  width: 640,
  height: 400,
  orientation: "vertical",
});

console.log("\n[Layout] Dendrogram Coordinates");
console.log("Type:", dendroLayout.type);
console.log("Nodes:", dendroLayout.data.nodes.length);
console.log("Links:", dendroLayout.data.links.length);
console.log("Sample node:", dendroLayout.data.nodes[0]);
console.log("Sample link:", dendroLayout.data.links[0]);

// ## RDA Visualization - Cars Dataset (kept as an example)
console.log("\n\n=== RDA Triplot - Cars Dataset ===\n");

// Fetch cars dataset
const carsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json",
);
const carsData = await carsResponse.json();

// Filter complete cases
const validCars = carsData.filter((c) =>
  c.Miles_per_Gallon && c.Acceleration && c.Horsepower &&
  c.Weight_in_lbs && c.Cylinders
).slice(0, 30);

// Y = performance metrics (MPG, Acceleration, combined metric)
const rdaY = validCars.map((c) => [
  c.Miles_per_Gallon,
  c.Acceleration,
  c.Miles_per_Gallon / (c.Weight_in_lbs / 1000), // Efficiency ratio
]);

// X = engine characteristics (Cylinders, Horsepower scaled)
const rdaX = validCars.map((c) => [
  c.Cylinders,
  c.Horsepower / 100, // Scale down
]);

const rdaEstimator = new mva.RDA();
rdaEstimator.fit(rdaY, rdaX);
const rdaResult = rdaEstimator.model;
console.log("RDA completed on Cars dataset");
console.log(
  "Variance explained:",
  rdaResult.varianceExplained.map((v) => (v * 100).toFixed(1) + "%").join(", "),
);

const rdaPlot = plot.plotRDA(rdaResult, {
  width: 640,
  height: 400,
});

console.log("\n[Plot Config] RDA Triplot - Car Performance vs Engine");
console.log("Type:", rdaPlot.type);
console.log("Data keys:", Object.keys(rdaPlot.data));
console.log("Number of sites:", rdaPlot.data.sites.length);
console.log("Number of species:", rdaPlot.data.species.length);

console.log("\n✓ All visualization configs generated successfully");
console.log("\nNote: These configs are designed for Observable Plot rendering");
console.log("In a browser/notebook environment, use:");
console.log('  import * as Plot from "@observablehq/plot"');
console.log("  Plot.plot(config)");
