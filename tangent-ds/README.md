# @tangent.to/ds

A browser-friendly data science library in modern JavaScript (ESM).

## Installation

```bash
npm install @tangent.to/ds
```

## Quick Start

```javascript
import { core, stats, ml, mva, plot } from '@tangent.to/ds';

// Linear algebra
const transposed = core.linalg.transpose([[1, 2], [3, 4]]);

// K-Means clustering
const kmeans = new ml.KMeans({ k: 3 });
kmeans.fit(data);

// PCA
const pca = new mva.PCA({ center: true, scale: false });
pca.fit(data);
```

## Modules

### Core (`core`)

#### Linear Algebra (`core.linalg`)

Matrix and vector operations:

```javascript
// Matrix transpose
core.linalg.transpose(matrix)

// Dot product
core.linalg.dot(vec1, vec2)

// Matrix multiplication
core.linalg.matmul(A, B)
```

#### Tables (`core.table`)

Powered by Arquero for data frame operations.

### Statistics (`stats`)

#### Distributions

Probability density, cumulative distribution, and quantile functions:

```javascript
// Normal distribution N(μ, σ²)
// PDF: f(x) = (1/(σ√(2π))) exp(-(x-μ)²/(2σ²))
stats.normal.pdf(x, { mean: 0, sd: 1 })
stats.normal.cdf(x, { mean: 0, sd: 1 })
stats.normal.quantile(p, { mean: 0, sd: 1 })

// Uniform distribution U(a, b)
stats.uniform.pdf(x, { min: 0, max: 1 })

// Gamma distribution Γ(k, θ)
stats.gamma.pdf(x, { shape: 1, scale: 1 })

// Beta distribution Beta(α, β)
stats.beta.pdf(x, { alpha: 1, beta: 1 })
```

**Note:** Use `stats.normal.quantile(Math.random())` for random sampling.

#### Regression Models

**Linear Model (OLS)**

Fits: $y = X\beta + \epsilon$

```javascript
const model = new stats.lm();
model.fit(X, y);
const predictions = model.predict(X_new);
const summary = model.summary(); // coefficients, R², p-values
```

**Logistic Regression**

Models: $P(y=1|X) = \sigma(X\beta)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$

```javascript
const model = new stats.logit();
model.fit(X, y);
```

**Linear Mixed Models**

Random intercept model: $y_{ij} = X_{ij}\beta + u_j + \epsilon_{ij}$

```javascript
const model = new stats.lmm();
model.fit(X, y, groups);
```

#### Hypothesis Tests

```javascript
// One-sample t-test
const result = stats.hypothesis.oneSampleTTest(data, { mu0: 0 });
// Returns: { statistic, pValue, df, mean, se }

// Two-sample t-test
const result = stats.hypothesis.twoSampleTTest(group1, group2);

// Chi-square test
const result = stats.hypothesis.chiSquareTest(observed, expected);

// One-way ANOVA
const result = stats.hypothesis.oneWayAnova(groups);
```

### Machine Learning (`ml`)

#### K-Means Clustering

Minimizes: $\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$

```javascript
const model = new ml.KMeans({ k: 3, maxIter: 100 });
model.fit(data);
console.log(model.labels);      // Cluster assignments
console.log(model.centroids);   // Cluster centers
```

#### K-Nearest Neighbors

```javascript
// Classification
const knn = new ml.KNNClassifier({ k: 5 });
knn.fit(X_train, y_train);
const predictions = knn.predict(X_test);

// Regression
const knn = new ml.KNNRegressor({ k: 5 });
knn.fit(X_train, y_train);
```

#### Decision Trees

```javascript
// Classification
const dt = new ml.DecisionTreeClassifier({
  maxDepth: 5,
  minSamplesSplit: 2
});
dt.fit(X_train, y_train);

// Regression
const dt = new ml.DecisionTreeRegressor({ maxDepth: 5 });
dt.fit(X_train, y_train);
```

#### Random Forest

```javascript
const rf = new ml.RandomForestClassifier({
  nEstimators: 100,
  maxDepth: 10
});
rf.fit(X_train, y_train);
```

#### Generalized Additive Models (GAM)

```javascript
const gam = new ml.GAMRegressor({ splineOrder: 3 });
gam.fit(X, y);
```

#### Polynomial Regression

```javascript
const poly = new ml.PolynomialRegressor({ degree: 3 });
poly.fit(X, y);
```

#### Neural Networks

```javascript
const mlp = new ml.MLPRegressor({
  layerSizes: [10, 8, 1],
  activation: 'relu',
  epochs: 100,
  learningRate: 0.01
});
mlp.fit(X_train, y_train);
```

### Multivariate Analysis (`mva`)

#### Principal Component Analysis (PCA)

Finds orthogonal directions of maximum variance:

$X' = XW$ where $W$ contains eigenvectors of $\text{Cov}(X)$

```javascript
const pca = new mva.PCA({
  center: true,  // Center data
  scale: false   // Scale to unit variance
});
pca.fit(X);

console.log(pca.model.explainedVariance);      // Eigenvalues
console.log(pca.model.explainedVarianceRatio); // Proportion of variance
console.log(pca.model.components);             // Principal components (eigenvectors)

const X_transformed = pca.transform(X);
```

#### Linear Discriminant Analysis (LDA)

Finds linear combinations that maximize between-class variance relative to within-class variance.

```javascript
const lda = new mva.LDA();
lda.fit(X, y);
const X_transformed = lda.transform(X);
```

#### Redundancy Analysis (RDA)

Constrained ordination combining regression and PCA.

```javascript
const rda = new mva.RDA();
rda.fit(response, explanatory);
```

#### Hierarchical Clustering

```javascript
const hca = new ml.HCA({ linkage: 'ward' });
hca.fit(data);
console.log(hca.model.dendrogram);
```

### Visualization (`plot`)

Returns Observable Plot configurations (not rendered plots):

```javascript
// ROC curve
const config = plot.plotROC(yTrue, yPred);

// Confusion matrix
const config = plot.plotConfusionMatrix(yTrue, yPred);

// PCA biplot
const config = plot.plotPCA(pca.model);

// Residual plots
const config = plot.plotResiduals(yTrue, yPred);

// Feature importance
const config = plot.plotFeatureImportance(model.featureImportances);

// Silhouette plot
const config = plot.plotSilhouette({ data: X, labels });
```

Use these configs with Observable Plot or tangent-notebook to render actual visualizations.

## Model Selection & Evaluation

### Cross-Validation

```javascript
const scores = ml.validation.crossValidate(model, X, y, { cv: 5 });
```

### Metrics

```javascript
// Classification
ml.metrics.accuracy(yTrue, yPred);
ml.metrics.precision(yTrue, yPred);
ml.metrics.recall(yTrue, yPred);
ml.metrics.f1Score(yTrue, yPred);

// Regression
ml.metrics.mse(yTrue, yPred);
ml.metrics.rmse(yTrue, yPred);
ml.metrics.r2(yTrue, yPred);
ml.metrics.mae(yTrue, yPred);
```

### Grid Search

```javascript
const grid = new ml.GridSearchCV(model, {
  paramGrid: { k: [3, 5, 7], metric: ['euclidean', 'manhattan'] },
  cv: 5
});
grid.fit(X, y);
console.log(grid.bestParams);
```

## Preprocessing

```javascript
// Train-test split
const { XTrain, XTest, yTrain, yTest } = ml.preprocessing.trainTestSplit(X, y, {
  testSize: 0.2,
  shuffle: true
});

// Standard scaling: z = (x - μ) / σ
const scaler = ml.preprocessing.standardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

// Min-Max scaling: x' = (x - min) / (max - min)
const scaler = ml.preprocessing.minMaxScaler();

// Label encoding
const encoder = ml.preprocessing.labelEncoder();
const encoded = encoder.fit_transform(labels);
```

## Testing

```bash
npm test                # Watch mode
npm run test:run        # Run once (259 tests)
npm run test:coverage   # With coverage
```

## Examples

See `examples/` directory for full working examples:

- `examples/quick-test.js` - Quick smoke test
- `examples/misc/api_overview.js` - Complete API demo
- `examples/user-guide/` - Sequential data science pipeline

## License

GPL-3.0
