---
layout: default
title: Home
---

# @tangent.to/ds

A minimalist, browser-friendly data science library in modern JavaScript (ESM).

## Quick Links

- [API Reference](API.md) - Complete API documentation
- [GitHub Repository](https://github.com/tangent-to/tangent-ds)
- [npm Package](https://www.npmjs.com/package/@tangent.to/ds)

## Installation

```bash
npm install @tangent.to/ds
```

## Quick Start

```javascript
import { core, stats, ml, mva, plot } from '@tangent.to/ds';

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
```

## Modules

### [Core Module](core.md)
Linear algebra, tables, math, optimization, and formulas.

### [Stats Module](stats.md)
Distributions, GLM/GLMM, hypothesis tests, and model comparison.

**Key Features:**
- Unified `GLM` class for all regression models (gaussian, binomial, poisson, gamma, etc.)
- Mixed-effects models (GLMM) with random intercepts and slopes
- Hypothesis tests (t-tests, chi-square, ANOVA)
- Model comparison tools (AIC, BIC, likelihood ratio tests)

### [ML Module](ml.md)
Machine learning algorithms: clustering, classification, and regression.

**Algorithms:**
- K-Means clustering
- K-Nearest Neighbors (KNN)
- Decision Trees & Random Forests
- Generalized Additive Models (GAM)
- Polynomial Regression
- Neural Networks (MLP)

### [MVA Module](mva.md)
Multivariate analysis: PCA, LDA, RDA, CCA, and hierarchical clustering.

**Methods:**
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Redundancy Analysis (RDA)
- Canonical Correlation Analysis (CCA)
- Hierarchical Clustering

### [Plot Module](plot.md)
Observable Plot configuration generators for data visualization.

**Plot Types:**
- ROC curves and confusion matrices
- PCA biplots and scree plots
- Feature importance plots
- Residual and QQ plots
- GLM diagnostic plots

## API Examples

### Generalized Linear Models (GLM)

```javascript
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

// Get results
console.log(model.summary());
console.log(model.coefficients);
```

### Formula Syntax

```javascript
const model = new stats.GLM({ family: 'gaussian' });
model.fit({
  formula: 'y ~ x1 + x2 + x3',
  data: myData
});

// Mixed models with random effects
model.fit({
  formula: 'y ~ x1 + x2 + (1 | group)',
  data: myData
});
```

### Machine Learning

```javascript
// K-Means clustering
const kmeans = new ml.KMeans({ k: 3 });
kmeans.fit(data);
console.log(kmeans.labels);

// Random Forest classifier
const rf = new ml.RandomForestClassifier({ nEstimators: 100 });
rf.fit(X_train, y_train);
const predictions = rf.predict(X_test);

// Cross-validation
const scores = ml.validation.crossValidate(model, X, y, { cv: 5 });
```

### Multivariate Analysis

```javascript
// PCA
const pca = new mva.PCA({ center: true, scale: false });
pca.fit(X);
console.log(pca.model.explainedVarianceRatio);
const X_transformed = pca.transform(X);

// LDA
const lda = new mva.LDA();
lda.fit(X, y);
const X_lda = lda.transform(X);
```

## Resources

- [Full API Reference](API.md)
- [Examples Directory](https://github.com/tangent-to/tangent-ds/tree/main/tangent-ds/examples)
- [GitHub Issues](https://github.com/tangent-to/tangent-ds/issues)

## License

GPL-3.0
