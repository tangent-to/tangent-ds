// %% [markdown]

/**
 * Multivariate Analysis (MVA) examples for @tangent.to/ds
 * Run with: node examples/mva_usage.js
 * Uses real datasets from Vega Datasets: https://cdn.jsdelivr.net/npm/vega-datasets@2/data/
 */

// %% [javascript]
import { core, mva, ml } from "../src/index.js";

console.log("=".repeat(70));
console.log("@tangent.to/ds - Multivariate Analysis Examples (Penguins)");
console.log("=".repeat(70));

// %% [javascript]
// Example 1: Principal Component Analysis (PCA) - Penguins Dataset
console.log(
  "\n📊 Example 1: Principal Component Analysis (PCA) - Penguins Dataset",
);
console.log("-".repeat(50));

// %% [javascript]
// Fetch penguins dataset
const penguinsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json",
);
const penguinsData = await penguinsResponse.json();

// Filter complete cases for PCA using common numeric fields
const validPCA = penguinsData.filter(
  (p) =>
    p.bill_length_mm != null &&
    p.bill_depth_mm != null &&
    p.flipper_length_mm != null &&
    p.body_mass_g != null &&
    !isNaN(p.bill_length_mm) &&
    !isNaN(p.bill_depth_mm) &&
    !isNaN(p.flipper_length_mm) &&
    !isNaN(p.body_mass_g),
).slice(0, 150);

// Use four measurements for PCA
const pcaData = validPCA.map((d) => [
  d.bill_length_mm,
  d.bill_depth_mm,
  d.flipper_length_mm,
  d.body_mass_g,
]);

const pcaEstimator = new mva.PCA({ scale: true });
try {
  pcaEstimator.fit(pcaData);
} catch (err) {
  console.error("PCA fit failed:", err.message);
}

if (pcaEstimator.fitted) {
  const pcaModel = pcaEstimator.model;
  console.log("PCA Results on Penguins:");
  console.log("  Eigenvalues:", pcaModel.eigenvalues.map((v) => v.toFixed(4)));
  console.log(
    "  Variance explained:",
    pcaModel.varianceExplained.map((v) => (v * 100).toFixed(2) + "%"),
  );

  const cumVar = pcaEstimator.cumulativeVariance();
  console.log(
    "  Cumulative variance:",
    cumVar.map((v) => (v * 100).toFixed(2) + "%"),
  );

  // Show first few PC scores
  console.log(
    "  First 3 PC1 scores:",
    pcaModel.scores.slice(0, 3).map((s) => s.pc1.toFixed(3)),
  );

  // Transform new data (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g)
  const newPenguinData = [[45.0, 17.5, 200, 4200]]; // example penguin measurements
  const transformed = pcaEstimator.transform(newPenguinData);
  console.log(
    "  New penguin projection:",
    `PC1=${transformed[0].pc1.toFixed(3)}, PC2=${transformed[0].pc2.toFixed(3)}`,
  );
} else {
  console.warn("Skipping PCA transform because the estimator was not fitted.");
}

// %% [javascript]
// Example 2: Hierarchical Clustering (HCA) - Penguins Dataset
console.log("\n🌳 Example 2: Hierarchical Clustering (HCA) - Penguins Dataset");
console.log("-".repeat(50));

// Use beak measurements for clustering (smaller subset for clarity)
const validPenguinsForHCA = penguinsData.filter((p) =>
  p.bill_length_mm != null && p.bill_depth_mm != null &&
  !isNaN(p.bill_length_mm) && !isNaN(p.bill_depth_mm)
).slice(0, 30);

const clusterData = validPenguinsForHCA.map((p) => [
  p.bill_length_mm,
  p.bill_depth_mm,
]);

const hcaEstimator = new ml.HCA({ linkage: "average" });
hcaEstimator.fit(clusterData);
const hcaModel = hcaEstimator.model;
const hcaSummary = hcaEstimator.summary();
console.log("HCA Results on Penguin Bill Measurements:");
console.log("  Number of merges:", hcaSummary.merges);
console.log("  Linkage method:", hcaSummary.linkage);

// Show merge sequence
console.log("  First 3 merges:");
hcaModel.dendrogram.slice(0, 3).forEach((merge, i) => {
  console.log(
    `    Merge ${i + 1}: distance=${
      merge.distance.toFixed(3)
    }, size=${merge.size}`,
  );
});

// Cut into 3 clusters (expecting species groups)
const labels3 = hcaEstimator.cut(3);
console.log("  3-cluster assignment:", labels3);

// Cut at specific height
const labelsHeight = hcaEstimator.cutHeight(5);
console.log("  Clusters at height 5:", labelsHeight);

// %% [javascript]
// Example 3: Linear Discriminant Analysis (LDA) - Penguins Dataset
console.log(
  "\n🎯 Example 3: Linear Discriminant Analysis (LDA) - Penguins Dataset",
);
console.log("-".repeat(50));

// Prepare features and labels for LDA using two measurements
const validLDA = penguinsData.filter((p) =>
  p.flipper_length_mm != null &&
  p.bill_depth_mm != null &&
  p.species
).slice(0, 150);

const ldaX = validLDA.map((d) => [d.flipper_length_mm, d.bill_depth_mm]);
const ldaY = validLDA.map((d) => d.species);

const ldaEstimator = new mva.LDA();
ldaEstimator.fit(ldaX, ldaY);
const ldaModel = ldaEstimator.model;
console.log("LDA Results on Penguins:");
console.log("  Classes found:", ldaModel.classes);
console.log("  Discriminant axes:", ldaModel.discriminantAxes.length);
console.log("  Eigenvalues:", ldaModel.eigenvalues.map((v) => v.toFixed(4)));

// Show sample discriminant scores
console.log("  Sample scores (first 2 per class):");
const uniqueClasses = [...new Set(ldaY)];
uniqueClasses.forEach((cls) => {
  const scores = ldaModel.scores.filter((s) => s.class === cls).slice(0, 2);
  scores.forEach((s) => {
    console.log(
      `    ${cls}: LD1=${s.ld1.toFixed(3)}, LD2=${s.ld2?.toFixed(3) || "N/A"}`,
    );
  });
});

// Predict new samples (flipper_length_mm, bill_depth_mm)
const newSamples = [[200, 15.5], [185, 17.0], [210, 14.2]];
const ldaPredictions = ldaEstimator.predict(newSamples);
console.log("  LDA Predictions for new samples:", ldaPredictions);

// %% [javascript]
// Example 4: Redundancy Analysis (RDA) - Ecological/Carried-over Example
console.log("\n🔗 Example 4: Redundancy Analysis (RDA) - Cars Dataset");
console.log("-".repeat(50));

// Fetch cars dataset
const carsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json",
);
const carsData = await carsResponse.json();

// Filter complete cases
const validCars = carsData.filter((c) =>
  c.Miles_per_Gallon && c.Acceleration && c.Horsepower &&
  c.Weight_in_lbs && c.Cylinders && c.Displacement
).slice(0, 50);

// Y = performance metrics (MPG, Acceleration)
// X = engine characteristics (Cylinders, Displacement, Horsepower, Weight)
const speciesMatrix = validCars.map((c) => [
  c.Miles_per_Gallon,
  c.Acceleration,
]);

const environment = validCars.map((c) => [
  c.Cylinders,
  c.Horsepower / 100, // Scale down
  c.Weight_in_lbs / 1000, // Scale to thousands
]);

const rdaEstimator = new mva.RDA();
rdaEstimator.fit(speciesMatrix, environment);
const rdaModel = rdaEstimator.model;
console.log("RDA Results (Car Performance vs Engine Specs):");
console.log(
  "  Constrained variance explained:",
  (rdaModel.constrainedVariance * 100).toFixed(2) + "%",
);
console.log(
  "  Canonical eigenvalues:",
  rdaModel.eigenvalues.slice(0, 2).map((v) => v.toFixed(4)),
);
console.log(
  "  Variance explained by axes:",
  rdaModel.varianceExplained.slice(0, 2).map((v) => (v * 100).toFixed(2) + "%"),
);

// Show canonical scores
console.log("  Canonical scores (first 3 cars):");
rdaModel.canonicalScores.slice(0, 3).forEach((score, i) => {
  console.log(
    `    Car ${i + 1}: RDA1=${score.rda1.toFixed(3)}, RDA2=${
      score.rda2?.toFixed(3) || "N/A"
    }`,
  );
});

// %% [javascript]
// Example 5: PCA for Dimensionality Reduction - Movies Dataset
console.log(
  "\n💡 Example 5: Dimensionality Reduction with PCA - Movies Dataset",
);
console.log("-".repeat(50));

// Fetch movies dataset
const moviesResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/movies.json",
);
const moviesData = await moviesResponse.json();

// Filter movies with complete data
const validMovies = moviesData.filter((m) =>
  m["IMDB Rating"] && m["Rotten Tomatoes Rating"] &&
  m["US Gross"] && m["Worldwide Gross"]
).slice(0, 100);

// Create feature matrix (ratings and financial performance)
const highDimData = validMovies.map((m) => [
  m["IMDB Rating"],
  m["Rotten Tomatoes Rating"] / 10, // Scale to 0-10
  Math.log10(m["US Gross"] || 1), // Log scale
  Math.log10(m["Worldwide Gross"] || 1), // Log scale
  m["US DVD Sales"] ? Math.log10(m["US DVD Sales"]) : 0,
]);

const dimReductionEstimator = new mva.PCA({ scale: true });
dimReductionEstimator.fit(highDimData);
const dimReductionModel = dimReductionEstimator.model;

console.log("Original dimensions:", highDimData[0].length);
console.log(
  "First 2 PCs explain:",
  ((dimReductionModel.varianceExplained[0] +
    dimReductionModel.varianceExplained[1]) * 100).toFixed(2),
  "% of variance",
);

// Extract first 2 principal components
const reduced = dimReductionModel.scores.map((s) => [s.pc1, s.pc2]);
console.log("Reduced data (first movie):", reduced[0].map((v) => v.toFixed(3)));

// %% [javascript]
// Example 6: Comparing Linkage Methods in HCA - Stock Prices
console.log("\n🔀 Example 6: Comparing HCA Linkage Methods - Stock Prices");
console.log("-".repeat(50));

// Fetch S&P 500 data
const sp500Response = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/sp500.csv",
);
const sp500Text = await sp500Response.text();
const sp500Data = sp500Text.split("\n").slice(1, 21).map((line) => {
  const parts = line.split(",");
  return parseFloat(parts[1]);
}).filter((p) => !isNaN(p));

// Use price values as simple 1D clustering example
const compareData = sp500Data.map((p) => [p]);

["single", "complete", "average"].forEach((linkage) => {
  const estimator = new ml.HCA({ linkage });
  estimator.fit(compareData);
  const labels = estimator.cut(3);
  console.log(`  ${linkage} linkage clusters:`, labels);
});

console.log("\n" + "=".repeat(70));
console.log("✅ All MVA examples completed successfully!");
console.log("=".repeat(70) + "\n");
