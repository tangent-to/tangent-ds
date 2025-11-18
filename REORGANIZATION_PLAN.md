# Package Reorganization & Branching Strategy

## Table of Contents
1. [Current Structure](#current-structure)
2. [Proposed Reorganization](#proposed-reorganization)
3. [Migration Plan](#migration-plan)
4. [Git Branching Strategy](#git-branching-strategy)
5. [Pipeline Branching (ML Concept)](#pipeline-branching-ml-concept)

---

## Current Structure

```
src/
├── core/                    # Core utilities
│   ├── estimators/
│   │   └── estimator.js    # Base estimator class
│   ├── math.js             # Mathematical utilities
│   ├── table.js            # Table/data handling
│   ├── linalg.js           # Linear algebra
│   ├── formula.js          # Formula parsing
│   ├── optimize.js         # Optimization routines
│   └── persistence.js      # Serialization
│
├── ml/                      # Machine learning (MIXED)
│   ├── estimators/         # Class-based estimators
│   │   ├── KMeans.js
│   │   ├── DBSCAN.js
│   │   ├── HDBSCAN.js
│   │   ├── HCA.js
│   │   ├── KNN.js
│   │   ├── DecisionTree.js
│   │   ├── RandomForest.js
│   │   ├── GAM.js
│   │   ├── PolynomialRegressor.js
│   │   └── MLPRegressor.js
│   ├── kmeans.js           # Functional clustering
│   ├── dbscan.js
│   ├── hdbscan.js
│   ├── hca.js
│   ├── silhouette.js       # Clustering metrics
│   ├── polynomial.js       # Regression
│   ├── mlp.js
│   ├── gam_utils.js
│   ├── splines.js
│   ├── pipeline.js         # Pipeline & GridSearch
│   ├── recipe.js           # Preprocessing pipeline
│   ├── preprocessing.js    # Preprocessing functions
│   ├── impute.js           # Missing data imputation
│   ├── outliers.js         # Outlier detection
│   ├── metrics.js          # Evaluation metrics
│   ├── distances.js        # Distance metrics
│   ├── criteria.js         # Impurity criteria
│   ├── loss.js             # Loss functions
│   ├── train.js            # Training utilities
│   ├── tuning.js           # Hyperparameter tuning
│   ├── validation.js       # Cross-validation
│   ├── interpret.js        # Model interpretation
│   └── utils.js
│
├── mva/                     # Multivariate analysis
│   ├── estimators/
│   │   ├── PCA.js
│   │   ├── LDA.js
│   │   ├── RDA.js
│   │   └── CCA.js
│   └── [functional implementations]
│
├── stats/                   # Statistical methods
│   ├── estimators/
│   │   ├── GLM.js
│   │   └── tests.js
│   ├── glm.js
│   ├── families.js
│   ├── distribution.js
│   ├── tests.js
│   ├── multinomial.js
│   └── model_comparison.js
│
└── plot/                    # Visualizations
    ├── renderers/
    │   └── d3Dendrogram.js
    ├── ordiplot.js
    ├── plotScree.js
    ├── plotHCA.js
    ├── plotHDBSCAN.js
    ├── plotSilhouette.js
    ├── classification.js
    ├── diagnostics.js
    ├── utils.js
    └── show.js
```

**Issues with Current Structure:**
- `/ml` directory is too large and mixed
- Clustering, classification, regression all in one folder
- Hard to find specific algorithm types
- Preprocessing mixed with algorithms
- No clear separation of concerns

---

## Proposed Reorganization

### Option A: Detailed Functional Organization

```
src/
├── core/                           # Core utilities (UNCHANGED)
│   ├── estimators/
│   │   ├── estimator.js           # Base estimator
│   │   ├── regressor.js           # Base regressor
│   │   ├── classifier.js          # Base classifier
│   │   └── transformer.js         # Base transformer
│   ├── math.js
│   ├── linalg.js
│   ├── table.js
│   ├── formula.js
│   ├── optimize.js
│   └── persistence.js
│
├── preprocessing/                  # NEW - Data preprocessing
│   ├── encoders/
│   │   ├── OneHotEncoder.js
│   │   ├── LabelEncoder.js
│   │   └── TargetEncoder.js
│   ├── scalers/
│   │   ├── StandardScaler.js
│   │   ├── MinMaxScaler.js
│   │   └── RobustScaler.js
│   ├── imputation/
│   │   ├── SimpleImputer.js
│   │   ├── KNNImputer.js
│   │   └── IterativeImputer.js
│   ├── feature_engineering/
│   │   ├── PolynomialFeatures.js
│   │   ├── Interactions.js
│   │   └── Binning.js
│   ├── recipe.js                  # Recipe API (MOVED)
│   └── index.js
│
├── clustering/                     # NEW - Clustering algorithms
│   ├── functional/                # Functional implementations
│   │   ├── kmeans.js
│   │   ├── dbscan.js
│   │   ├── hdbscan.js
│   │   └── hca.js
│   ├── estimators/                # Class-based implementations
│   │   ├── KMeans.js
│   │   ├── DBSCAN.js
│   │   ├── HDBSCAN.js
│   │   └── HCA.js
│   ├── metrics/
│   │   ├── silhouette.js
│   │   ├── davies_bouldin.js
│   │   └── calinski_harabasz.js
│   └── index.js
│
├── classification/                 # NEW - Classification algorithms
│   ├── functional/
│   │   ├── knn.js
│   │   ├── decision_tree.js
│   │   └── random_forest.js
│   ├── estimators/
│   │   ├── KNNClassifier.js
│   │   ├── DecisionTreeClassifier.js
│   │   ├── RandomForestClassifier.js
│   │   └── GAMClassifier.js
│   └── index.js
│
├── regression/                     # NEW - Regression algorithms
│   ├── functional/
│   │   ├── polynomial.js
│   │   ├── gam.js
│   │   └── mlp.js
│   ├── estimators/
│   │   ├── KNNRegressor.js
│   │   ├── DecisionTreeRegressor.js
│   │   ├── RandomForestRegressor.js
│   │   ├── PolynomialRegressor.js
│   │   ├── GAMRegressor.js
│   │   └── MLPRegressor.js
│   ├── gam_utils.js
│   └── splines.js
│
├── ensemble/                       # NEW - Ensemble methods
│   ├── bagging.js
│   ├── boosting.js
│   ├── stacking.js
│   └── voting.js
│
├── model_selection/                # NEW - Model selection
│   ├── cross_validation.js
│   ├── grid_search.js
│   ├── random_search.js
│   ├── train_test_split.js
│   └── index.js
│
├── metrics/                        # NEW - All metrics
│   ├── clustering/
│   │   ├── silhouette.js
│   │   ├── davies_bouldin.js
│   │   └── calinski_harabasz.js
│   ├── classification/
│   │   ├── accuracy.js
│   │   ├── precision_recall.js
│   │   ├── roc_auc.js
│   │   └── confusion_matrix.js
│   ├── regression/
│   │   ├── mse.js
│   │   ├── mae.js
│   │   ├── r2.js
│   │   └── rmse.js
│   ├── distances.js
│   └── index.js
│
├── outliers/                       # NEW - Outlier detection
│   ├── IsolationForest.js
│   ├── LocalOutlierFactor.js
│   ├── MahalanobisDistance.js
│   └── index.js
│
├── interpret/                      # NEW - Model interpretation
│   ├── feature_importance.js
│   ├── partial_dependence.js
│   ├── shap.js
│   └── lime.js
│
├── pipeline/                       # NEW - Pipeline utilities
│   ├── Pipeline.js
│   ├── FeatureUnion.js
│   └── index.js
│
├── utils/                          # NEW - Shared utilities
│   ├── loss.js
│   ├── train.js
│   ├── criteria.js
│   └── validation.js
│
├── mva/                            # Multivariate analysis (MOSTLY UNCHANGED)
│   ├── estimators/
│   │   ├── PCA.js
│   │   ├── LDA.js
│   │   ├── RDA.js
│   │   └── CCA.js
│   └── index.js
│
├── stats/                          # Statistics (UNCHANGED)
│   ├── estimators/
│   │   ├── GLM.js
│   │   └── tests.js
│   ├── glm.js
│   ├── families.js
│   ├── distribution.js
│   └── tests.js
│
└── plot/                           # Visualizations (ORGANIZED)
    ├── renderers/
    │   └── d3Dendrogram.js
    ├── clustering/
    │   ├── plotHCA.js
    │   ├── plotHDBSCAN.js
    │   └── plotSilhouette.js
    ├── classification/
    │   ├── plotROC.js
    │   ├── plotConfusionMatrix.js
    │   ├── plotPrecisionRecall.js
    │   └── plotCalibration.js
    ├── regression/
    │   ├── plotResiduals.js
    │   ├── plotQQ.js
    │   └── diagnostics.js
    ├── multivariate/
    │   ├── ordiplot.js
    │   └── plotScree.js
    ├── interpretation/
    │   ├── plotFeatureImportance.js
    │   ├── plotPartialDependence.js
    │   └── plotLearningCurve.js
    ├── utils.js
    └── show.js
```

### Option B: Simpler Organization (Recommended)

```
src/
├── core/                    # Core utilities
├── preprocessing/           # All preprocessing
├── clustering/              # All clustering
├── classification/          # All classification
├── regression/              # All regression
├── metrics/                 # All metrics
├── model_selection/         # CV, GridSearch, etc.
├── outliers/                # Outlier detection
├── pipeline/                # Pipeline utilities
├── mva/                     # Multivariate analysis
├── stats/                   # Statistics
└── plot/                    # Visualizations
    ├── clustering/
    ├── classification/
    ├── regression/
    └── multivariate/
```

**Benefits:**
- ✅ Clear separation by algorithm type
- ✅ Easy to find specific algorithms
- ✅ Better scalability
- ✅ Follows scikit-learn organization
- ✅ Improved code navigation

---

## Migration Plan

### Phase 1: Create New Structure (Week 1)

1. **Create new directories**
   ```bash
   mkdir -p src/{preprocessing,clustering,classification,regression,metrics,model_selection,outliers,pipeline,interpret}
   mkdir -p src/clustering/{functional,estimators}
   mkdir -p src/classification/{functional,estimators}
   mkdir -p src/regression/{functional,estimators}
   ```

2. **Move clustering files**
   ```bash
   # Functional implementations
   mv src/ml/kmeans.js src/clustering/functional/
   mv src/ml/dbscan.js src/clustering/functional/
   mv src/ml/hdbscan.js src/clustering/functional/
   mv src/ml/hca.js src/clustering/functional/
   mv src/ml/silhouette.js src/clustering/metrics/

   # Estimators
   mv src/ml/estimators/KMeans.js src/clustering/estimators/
   mv src/ml/estimators/DBSCAN.js src/clustering/estimators/
   mv src/ml/estimators/HDBSCAN.js src/clustering/estimators/
   mv src/ml/estimators/HCA.js src/clustering/estimators/
   ```

3. **Create index files with backward compatibility**
   ```javascript
   // src/clustering/index.js
   export * from './functional/kmeans.js';
   export * from './functional/dbscan.js';
   export * from './functional/hdbscan.js';
   export * from './functional/hca.js';
   export * from './estimators/KMeans.js';
   export * from './estimators/DBSCAN.js';
   export * from './estimators/HDBSCAN.js';
   export * from './estimators/HCA.js';
   export * from './metrics/silhouette.js';
   ```

4. **Update imports in moved files**
   - Update relative paths
   - Update core imports
   - Update cross-module dependencies

### Phase 2: Maintain Backward Compatibility (Week 1-2)

```javascript
// src/ml/index.js - Keep old exports working
export {
  KMeans,
  DBSCAN,
  HDBSCAN,
  HCA,
  kmeans,
  dbscan,
  hdbscan,
  hca,
  silhouette
} from '../clustering/index.js';

export {
  KNNClassifier,
  DecisionTreeClassifier,
  RandomForestClassifier
} from '../classification/index.js';

// ... etc
```

### Phase 3: Update Tests (Week 2)

```javascript
// Update test imports
// OLD:
import { KMeans } from '../src/ml/index.js';

// NEW:
import { KMeans } from '../src/clustering/index.js';
```

### Phase 4: Update Documentation (Week 2-3)

- Update all examples
- Update README
- Add migration guide
- Update API documentation

### Phase 5: Deprecation Period (Release N)

```javascript
// src/ml/index.js
/**
 * @deprecated Use '@tangent.to/ds/clustering' instead
 * This export will be removed in v1.0.0
 */
export { KMeans } from '../clustering/index.js';
```

### Phase 6: Remove Old Structure (Release N+1)

- Remove deprecated exports
- Remove old `/ml` structure
- Update version to 1.0.0

---

## Git Branching Strategy

### Overview

We use **GitHub Flow** with feature branches for development.

```
main
  ├── feature/reorganize-structure
  ├── feature/add-spectral-clustering
  ├── bugfix/hdbscan-stability
  └── docs/update-examples
```

### Branch Naming Convention

```
<type>/<description>-<session-id>

Types:
- feature/   - New features
- bugfix/    - Bug fixes
- hotfix/    - Critical fixes for production
- docs/      - Documentation only
- refactor/  - Code refactoring
- test/      - Test additions
- chore/     - Maintenance tasks
- claude/    - AI-assisted development

Examples:
- feature/add-spectral-clustering-01XYZ
- bugfix/kmeans-convergence-01ABC
- claude/implement-hdbscan-plotting-01RffDKyTEWTq1c72zRrDSVQ
```

### Workflow

#### 1. **Create Feature Branch**

```bash
# From main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/reorganize-structure-01ABC
```

#### 2. **Development**

```bash
# Make changes
git add src/clustering/
git commit -m "refactor: move clustering to dedicated module"

# Push frequently
git push -u origin feature/reorganize-structure-01ABC
```

#### 3. **Keep Branch Updated**

```bash
# Rebase on main (preferred)
git fetch origin
git rebase origin/main

# Or merge (if conflicts are complex)
git merge origin/main
```

#### 4. **Create Pull Request**

```bash
# Push final changes
git push origin feature/reorganize-structure-01ABC

# Create PR via GitHub CLI
gh pr create \
  --title "Reorganize package structure" \
  --body "$(cat <<'EOF'
## Summary
- Move clustering algorithms to dedicated `/clustering` module
- Maintain backward compatibility via re-exports
- Update all tests and documentation

## Breaking Changes
None - backward compatibility maintained

## Test Plan
- [ ] All existing tests pass
- [ ] Import paths work from both old and new locations
- [ ] Documentation examples still work

## Migration Guide
See REORGANIZATION_PLAN.md for details
EOF
)"
```

#### 5. **Review & Merge**

```bash
# After approval, merge to main
gh pr merge --squash  # or --merge, or --rebase

# Delete feature branch
git branch -d feature/reorganize-structure-01ABC
git push origin --delete feature/reorganize-structure-01ABC
```

### Branch Protection Rules

```yaml
main:
  require_pull_request: true
  require_reviews: 1
  require_status_checks:
    - tests
    - lint
    - build
  allow_force_push: false
  allow_deletions: false
```

### Long-Running Branches

For major reorganizations, use long-running branches:

```
main
  └── refactor/v1.0-reorganization
        ├── feature/move-clustering
        ├── feature/move-classification
        └── feature/move-regression
```

```bash
# Create long-running branch
git checkout -b refactor/v1.0-reorganization

# Create sub-features
git checkout -b feature/move-clustering
# ... work ...
git rebase refactor/v1.0-reorganization
git checkout refactor/v1.0-reorganization
git merge feature/move-clustering

# When all done, merge to main
git checkout main
git merge refactor/v1.0-reorganization
```

---

## Pipeline Branching (ML Concept)

### What is Pipeline Branching?

Pipeline branching allows you to split your data preprocessing or modeling pipeline into multiple paths, apply different transformations or models to each branch, and then combine the results.

### Use Cases

1. **Ensemble Learning**: Train multiple models in parallel
2. **Feature Engineering**: Apply different transformations to different feature sets
3. **A/B Testing**: Compare different preprocessing strategies
4. **Hybrid Models**: Combine rule-based and ML approaches

### Implementation

#### Basic Branch Pipeline

```javascript
import { BranchPipeline } from '@tangent.to/ds/pipeline';

const pipeline = new BranchPipeline({
  branches: {
    // Branch 1: K-Means
    kmeans: new Pipeline()
      .add('scaler', new StandardScaler())
      .add('pca', new PCA({ nComponents: 10 }))
      .add('cluster', new KMeans({ k: 3 })),

    // Branch 2: HDBSCAN
    hdbscan: new Pipeline()
      .add('scaler', new StandardScaler())
      .add('cluster', new HDBSCAN({ minClusterSize: 5 })),

    // Branch 3: Hierarchical
    hca: new Pipeline()
      .add('scaler', new RobustScaler())
      .add('cluster', new HCA({ linkage: 'ward' }))
  },

  // How to combine results
  combiner: 'vote'  // or 'average', 'max', 'stack', custom function
});

// Fit all branches
pipeline.fit(X);

// Predict using consensus
const labels = pipeline.predict(X);
```

#### Advanced: Conditional Branching

```javascript
import { ConditionalPipeline } from '@tangent.to/ds/pipeline';

const pipeline = new ConditionalPipeline({
  router: (X) => {
    // Route based on data characteristics
    const nSamples = X.length;
    const nFeatures = X[0].length;

    if (nSamples < 100) return 'small';
    if (nFeatures > 50) return 'highdim';
    return 'default';
  },

  branches: {
    small: new DBSCAN({ eps: 0.5, minSamples: 3 }),
    highdim: new Pipeline()
      .add('pca', new PCA({ nComponents: 10 }))
      .add('cluster', new HDBSCAN({ minClusterSize: 5 })),
    default: new KMeans({ k: 3 })
  }
});
```

#### Feature Union (Parallel Features)

```javascript
import { FeatureUnion } from '@tangent.to/ds/pipeline';

const featureUnion = new FeatureUnion({
  branches: {
    // Numerical features
    numeric: new Pipeline()
      .add('select', new ColumnSelector(['age', 'income']))
      .add('scale', new StandardScaler()),

    // Categorical features
    categorical: new Pipeline()
      .add('select', new ColumnSelector(['category', 'region']))
      .add('encode', new OneHotEncoder()),

    // Text features
    text: new Pipeline()
      .add('select', new ColumnSelector(['description']))
      .add('vectorize', new TfidfVectorizer({ maxFeatures: 100 }))
  },

  // Concatenate all transformed features
  join: 'concat'  // or 'stack', 'merge'
});

// Apply all transformations and concatenate
const Xtransformed = featureUnion.fitTransform(data);

// Use in a larger pipeline
const fullPipeline = new Pipeline()
  .add('features', featureUnion)
  .add('model', new RandomForestClassifier());
```

#### Ensemble Voting Pipeline

```javascript
import { VotingPipeline } from '@tangent.to/ds/pipeline';

const ensemble = new VotingPipeline({
  estimators: {
    rf: new RandomForestClassifier({ nEstimators: 100 }),
    knn: new KNNClassifier({ k: 5 }),
    tree: new DecisionTreeClassifier({ maxDepth: 10 })
  },

  voting: 'soft',  // 'soft' (probabilities) or 'hard' (majority vote)
  weights: [2, 1, 1]  // Optional: weight the votes
});

ensemble.fit(X, y);
const predictions = ensemble.predict(Xtest);
```

#### Stacking Pipeline

```javascript
import { StackingPipeline } from '@tangent.to/ds/pipeline';

const stacker = new StackingPipeline({
  baseEstimators: {
    rf: new RandomForestClassifier(),
    knn: new KNNClassifier(),
    svm: new SVMClassifier()
  },

  // Meta-estimator learns from base estimator predictions
  metaEstimator: new LogisticRegression(),

  // Use cross-validation predictions for training meta-estimator
  usePredictions: 'cv',  // or 'holdout', 'all'
  cv: 5
});

stacker.fit(X, y);
const predictions = stacker.predict(Xtest);
```

### Implementation Example

```javascript
// src/pipeline/BranchPipeline.js
export class BranchPipeline {
  constructor({ branches, combiner = 'vote' }) {
    this.branches = branches;
    this.combiner = combiner;
    this.fitted = false;
  }

  fit(X, y = null) {
    // Fit all branches in parallel
    const fitPromises = Object.entries(this.branches).map(
      ([name, pipeline]) => {
        return new Promise((resolve) => {
          pipeline.fit(X, y);
          resolve({ name, pipeline });
        });
      }
    );

    // Wait for all branches to complete
    Promise.all(fitPromises).then(() => {
      this.fitted = true;
    });

    return this;
  }

  predict(X) {
    if (!this.fitted) {
      throw new Error('Pipeline not fitted');
    }

    // Get predictions from all branches
    const predictions = Object.entries(this.branches).map(
      ([name, pipeline]) => ({
        name,
        labels: pipeline.predict(X)
      })
    );

    // Combine predictions based on combiner strategy
    return this._combine(predictions);
  }

  _combine(predictions) {
    const n = predictions[0].labels.length;
    const combined = new Array(n);

    switch (this.combiner) {
      case 'vote':
        // Majority voting
        for (let i = 0; i < n; i++) {
          const votes = {};
          predictions.forEach(({ labels }) => {
            votes[labels[i]] = (votes[labels[i]] || 0) + 1;
          });
          combined[i] = Object.entries(votes)
            .sort((a, b) => b[1] - a[1])[0][0];
        }
        break;

      case 'average':
        // Average predictions (for regression)
        for (let i = 0; i < n; i++) {
          const sum = predictions.reduce(
            (acc, { labels }) => acc + labels[i],
            0
          );
          combined[i] = sum / predictions.length;
        }
        break;

      case 'max':
        // Maximum confidence
        for (let i = 0; i < n; i++) {
          combined[i] = Math.max(
            ...predictions.map(({ labels }) => labels[i])
          );
        }
        break;

      default:
        if (typeof this.combiner === 'function') {
          return this.combiner(predictions);
        }
        throw new Error(`Unknown combiner: ${this.combiner}`);
    }

    return combined;
  }
}
```

### Usage Example

```javascript
import { BranchPipeline } from '@tangent.to/ds/pipeline';
import { KMeans, DBSCAN, HDBSCAN } from '@tangent.to/ds/clustering';

// Create consensus clustering
const consensusClustering = new BranchPipeline({
  branches: {
    kmeans: new KMeans({ k: 3 }),
    dbscan: new DBSCAN({ eps: 0.5, minSamples: 5 }),
    hdbscan: new HDBSCAN({ minClusterSize: 5 })
  },
  combiner: 'vote'
});

// Fit all algorithms
consensusClustering.fit(data);

// Get consensus labels
const labels = consensusClustering.predict(data);

// Compare individual branch results
const kmeansLabels = consensusClustering.branches.kmeans.labels;
const dbscanLabels = consensusClustering.branches.dbscan.labels;
const hdbscanLabels = consensusClustering.branches.hdbscan.labels;

console.log('Consensus:', labels);
console.log('K-Means:', kmeansLabels);
console.log('DBSCAN:', dbscanLabels);
console.log('HDBSCAN:', hdbscanLabels);
```

---

## Next Steps

### Immediate (This Week)
1. **Review** this reorganization plan
2. **Decide** on Option A or Option B structure
3. **Create** feature branch for reorganization
4. **Start** Phase 1: Create new directories

### Short Term (Next 2 Weeks)
1. **Migrate** clustering module
2. **Update** tests
3. **Add** backward compatibility
4. **Document** changes

### Long Term (Next Release)
1. **Complete** full reorganization
2. **Implement** pipeline branching
3. **Add** ensemble methods
4. **Release** v1.0.0

---

## Questions to Address

1. **Which organization option?** A (detailed) or B (simpler)?
2. **Breaking changes?** Or maintain full backward compatibility?
3. **Timeline?** Gradual over multiple releases or big-bang?
4. **Version bump?** v0.4.0 or v1.0.0?
5. **Pipeline branching?** Implement now or later?

---

## Summary

**Git Branching**: We use feature branches with naming convention `<type>/<description>-<id>`

**Package Structure**: Reorganize from flat `/ml` to organized modules by algorithm type

**Pipeline Branching**: ML concept for parallel processing and ensemble methods

**Migration**: Phased approach with backward compatibility maintained
