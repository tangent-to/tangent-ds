#!/usr/bin/env node
/**
 * Generate API documentation from source code
 * Creates markdown files in the docs/ directory
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');
const docsDir = path.join(rootDir, 'docs');

// Create docs directory
if (!fs.existsSync(docsDir)) {
  fs.mkdirSync(docsDir, { recursive: true });
}

/**
 * Extract JSDoc comment from file content
 */
function extractJSDocComments(content) {
  const comments = [];
  const jsdocRegex = /\/\*\*\s*([\s\S]*?)\*\//g;
  let match;

  while ((match = jsdocRegex.exec(content)) !== null) {
    const comment = match[1]
      .split('\n')
      .map(line => line.trim().replace(/^\*\s?/, ''))
      .join('\n')
      .trim();
    comments.push(comment);
  }

  return comments;
}

/**
 * Extract exports from a file
 */
function extractExports(content) {
  const exports = [];

  // Match export class/function declarations
  const exportRegex = /export\s+(class|function|const|let|var)\s+(\w+)/g;
  let match;

  while ((match = exportRegex.exec(content)) !== null) {
    exports.push({ type: match[1], name: match[2] });
  }

  // Match export { ... } statements
  const namedExportRegex = /export\s+\{([^}]+)\}/g;
  while ((match = namedExportRegex.exec(content)) !== null) {
    const names = match[1].split(',').map(n => n.trim().split(' as ')[0]);
    exports.push(...names.map(name => ({ type: 'named', name })));
  }

  return exports;
}

/**
 * Generate documentation for a module
 */
function generateModuleDocs(moduleName, indexPath) {
  const content = fs.readFileSync(indexPath, 'utf8');
  const exports = extractExports(content);
  const comments = extractJSDocComments(content);

  let md = `# ${moduleName.charAt(0).toUpperCase() + moduleName.slice(1)} Module\n\n`;

  // Add module description from first comment
  if (comments.length > 0) {
    md += `${comments[0]}\n\n`;
  }

  md += `## Exports\n\n`;

  // List all exports
  for (const exp of exports) {
    if (exp.name) {
      md += `- \`${exp.name}\` (${exp.type})\n`;
    }
  }

  return md;
}

/**
 * Main documentation generator
 */
function generateDocs() {
  console.log('Generating API documentation...\n');

  // Get all module directories
  const modules = fs.readdirSync(srcDir)
    .filter(name => {
      const fullPath = path.join(srcDir, name);
      return fs.statSync(fullPath).isDirectory();
    });

  // Generate main API.md
  let apiDoc = `# API Reference\n\n`;
  apiDoc += `Generated documentation for @tangent.to/ds\n\n`;
  apiDoc += `## Modules\n\n`;

  for (const module of modules) {
    const indexPath = path.join(srcDir, module, 'index.js');

    if (fs.existsSync(indexPath)) {
      console.log(`Processing ${module} module...`);

      // Generate module-specific doc
      const moduleDocs = generateModuleDocs(module, indexPath);
      const moduleDocPath = path.join(docsDir, `${module}.md`);
      fs.writeFileSync(moduleDocPath, moduleDocs);

      // Add to main API doc
      apiDoc += `- [${module}](./${module}.md) - ${module} module\n`;
    }
  }

  // Write main API doc
  fs.writeFileSync(path.join(docsDir, 'API.md'), apiDoc);

  // Generate updated README reference
  generateReadme();

  console.log(`\n✓ Documentation generated in ${docsDir}/`);
}

/**
 * Generate main README with updated API examples
 */
function generateReadme() {
  const readmePath = path.join(rootDir, 'README.md');

  const readme = `# @tangent.to/ds

A browser-friendly data science library in modern JavaScript (ESM).

## Installation

\`\`\`bash
npm install @tangent.to/ds
\`\`\`

## Quick Start

\`\`\`javascript
import { core, stats, ml, mva, plot } from '@tangent.to/ds';

// Linear algebra
const transposed = core.linalg.transpose([[1, 2], [3, 4]]);

// Generalized Linear Models
const model = new stats.GLM({ family: 'gaussian' });
model.fit(X, y);
const predictions = model.predict(X_new);

// K-Means clustering
const kmeans = new ml.KMeans({ k: 3 });
kmeans.fit(data);

// PCA
const pca = new mva.PCA({ center: true, scale: false });
pca.fit(data);
\`\`\`

## Modules

- **core** - Linear algebra, tables, math, optimization, formulas
- **stats** - Distributions, GLM/GLMM, hypothesis tests, model comparison
- **ml** - Machine learning algorithms (clustering, classification, regression)
- **mva** - Multivariate analysis (PCA, LDA, RDA, CCA, HCA)
- **plot** - Observable Plot configuration generators

## Statistics (stats)

### Generalized Linear Models (GLM)

The \`GLM\` class unifies all regression models with a consistent interface:

\`\`\`javascript
// Linear regression (Gaussian family)
const lm = new stats.GLM({ family: 'gaussian' });
lm.fit(X, y);

// Logistic regression (Binomial family)
const logit = new stats.GLM({ family: 'binomial', link: 'logit' });
logit.fit(X, y);

// Poisson regression
const poisson = new stats.GLM({ family: 'poisson' });
poisson.fit(X, y);

// Mixed-effects model (GLMM) with random intercepts
const lmm = new stats.GLM({
  family: 'gaussian',
  randomEffects: { intercept: groups }
});
lmm.fit(X, y);

// Access model results
console.log(model.summary());      // Detailed summary
console.log(model.coefficients);   // Coefficient estimates
const predictions = model.predict(X_new);
\`\`\`

### Formula Syntax

\`\`\`javascript
const model = new stats.GLM({ family: 'gaussian' });
model.fit({ formula: 'y ~ x1 + x2', data });
\`\`\`

### Distributions

\`\`\`javascript
// Normal distribution
stats.normal.pdf(x, { mean: 0, sd: 1 });
stats.normal.cdf(x, { mean: 0, sd: 1 });
stats.normal.quantile(p, { mean: 0, sd: 1 });

// Other distributions: uniform, gamma, beta
stats.gamma.pdf(x, { shape: 1, scale: 1 });
stats.beta.pdf(x, { alpha: 1, beta: 1 });
\`\`\`

### Hypothesis Tests

\`\`\`javascript
// One-sample t-test
const result = stats.hypothesis.oneSampleTTest(data, { mu0: 0 });

// Two-sample t-test
const result = stats.hypothesis.twoSampleTTest(group1, group2);

// Chi-square test
const result = stats.hypothesis.chiSquareTest(observed, expected);

// One-way ANOVA
const result = stats.hypothesis.oneWayAnova(groups);
\`\`\`

### Model Comparison

\`\`\`javascript
// Compare models with AIC/BIC
const comparison = stats.compareModels([model1, model2, model3]);

// Likelihood ratio test for nested models
const lrt = stats.likelihoodRatioTest(model1, model2);
\`\`\`

## Machine Learning (ml)

### K-Means Clustering

\`\`\`javascript
const model = new ml.KMeans({ k: 3, maxIter: 100 });
model.fit(data);
console.log(model.labels);      // Cluster assignments
console.log(model.centroids);   // Cluster centers
\`\`\`

### K-Nearest Neighbors

\`\`\`javascript
// Classification
const knn = new ml.KNNClassifier({ k: 5 });
knn.fit(X_train, y_train);
const predictions = knn.predict(X_test);

// Regression
const knn = new ml.KNNRegressor({ k: 5 });
\`\`\`

### Decision Trees & Random Forests

\`\`\`javascript
// Decision tree
const dt = new ml.DecisionTreeClassifier({ maxDepth: 5 });
dt.fit(X_train, y_train);

// Random forest
const rf = new ml.RandomForestClassifier({ nEstimators: 100 });
rf.fit(X_train, y_train);
\`\`\`

### Polynomial Regression

\`\`\`javascript
const poly = new ml.PolynomialRegressor({ degree: 3 });
poly.fit(X, y);
\`\`\`

### Neural Networks

\`\`\`javascript
const mlp = new ml.MLPRegressor({
  layerSizes: [10, 8, 1],
  activation: 'relu',
  epochs: 100,
  learningRate: 0.01
});
mlp.fit(X_train, y_train);
\`\`\`

### Model Selection

\`\`\`javascript
// Cross-validation
const scores = ml.validation.crossValidate(model, X, y, { cv: 5 });

// Grid search
import { GridSearchCV } from '@tangent.to/ds/ml';
const result = GridSearchCV(fitFn, scoreFn, X, y, paramGrid, { k: 5 });
\`\`\`

## Multivariate Analysis (mva)

### PCA

\`\`\`javascript
const pca = new mva.PCA({ center: true, scale: false });
pca.fit(X);
console.log(pca.model.explainedVarianceRatio);
const X_transformed = pca.transform(X);
\`\`\`

### LDA

\`\`\`javascript
const lda = new mva.LDA();
lda.fit(X, y);
const X_transformed = lda.transform(X);
\`\`\`

### RDA (Redundancy Analysis)

\`\`\`javascript
const rda = new mva.RDA();
rda.fit(response, explanatory);
\`\`\`

### Hierarchical Clustering

\`\`\`javascript
const hca = new ml.HCA({ linkage: 'ward' });
hca.fit(data);
console.log(hca.model.dendrogram);
\`\`\`

## Visualization (plot)

Returns Observable Plot configurations:

\`\`\`javascript
// ROC curve
const config = plot.plotROC(yTrue, yPred);

// Confusion matrix
const config = plot.plotConfusionMatrix(yTrue, yPred);

// PCA biplot
const config = plot.plotPCA(pca.model);

// GLM diagnostics
const config = plot.diagnosticDashboard(model);
\`\`\`

## Metrics

\`\`\`javascript
// Classification
ml.metrics.accuracy(yTrue, yPred);
ml.metrics.f1Score(yTrue, yPred);

// Regression
ml.metrics.mse(yTrue, yPred);
ml.metrics.r2(yTrue, yPred);
\`\`\`

## Testing

\`\`\`bash
npm test                # Watch mode
npm run test:run        # Run once
npm run test:coverage   # With coverage
\`\`\`

## Documentation

See [docs/API.md](docs/API.md) for complete API reference.

## Examples

See \`examples/\` directory for working examples.

## License

GPL-3.0
`;

  fs.writeFileSync(readmePath, readme);
  console.log('✓ Updated README.md');
}

// Run the generator
generateDocs();
