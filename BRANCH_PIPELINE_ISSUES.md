# BranchPipeline: Critical Issues & Solutions

## 🚨 Problem: Label Alignment in Clustering

You've identified a **critical flaw** - cluster labels are arbitrary and don't match across models!

### What Goes Wrong

```javascript
// Three models assign different meanings to the same label:
Data:     [(0,0), (0,1), (10,10), (10,11), (5,5)]

KMeans:   [0, 0, 1, 1, 2]  // 0=top-left, 1=bottom-right, 2=middle
DBSCAN:   [0, 0, 1, 1, -1] // 0=top-left, 1=bottom-right, -1=noise
HDBSCAN:  [1, 1, 0, 0, 2]  // 1=top-left, 0=bottom-right, 2=middle

// Naive voting at sample 0:
// KMeans=0, DBSCAN=0, HDBSCAN=1 → Vote=0
// But KMeans:0 ≠ DBSCAN:0 ≠ HDBSCAN:1 (they all mean different clusters!)
```

### The Fundamental Issue

**Clustering is unsupervised** - labels are internal identifiers with no shared meaning across models. Voting on raw labels is **meaningless**.

---

## ✅ Solutions

### Solution 1: Co-Association Matrix (Recommended)

Instead of voting on labels, measure **pairwise agreement**: "Do two samples belong to the same cluster?"

```javascript
/**
 * ConsensusCluster - Proper consensus clustering via co-association
 */
export class ConsensusCluster {
  constructor({ estimators = [], threshold = 0.5 } = {}) {
    this.estimators = estimators;
    this.threshold = threshold;  // Minimum agreement to be in same cluster
    this.fitted = false;
  }

  fit(X) {
    const n = X.length;

    // Initialize co-association matrix (n x n)
    const coAssoc = Array(n).fill(0).map(() => Array(n).fill(0));

    // Fit each estimator and update co-association matrix
    for (const est of this.estimators) {
      est.fit(X);
      const labels = est.labels || est.predict(X);

      // Count co-occurrences
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          if (labels[i] === labels[j] && labels[i] !== -1) {
            coAssoc[i][j]++;
            coAssoc[j][i]++;
          }
        }
      }
    }

    // Normalize by number of estimators
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        coAssoc[i][j] /= this.estimators.length;
      }
    }

    // Extract consensus clusters using threshold
    this.labels = this._extractClusters(coAssoc, this.threshold);
    this.coAssocMatrix = coAssoc;
    this.fitted = true;

    return this;
  }

  _extractClusters(coAssoc, threshold) {
    const n = coAssoc.length;
    const labels = new Array(n).fill(-1);
    let clusterId = 0;

    // Simple connected components
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1) continue;

      // BFS to find connected component
      const queue = [i];
      labels[i] = clusterId;

      while (queue.length > 0) {
        const curr = queue.shift();

        for (let j = 0; j < n; j++) {
          if (labels[j] === -1 && coAssoc[curr][j] >= threshold) {
            labels[j] = clusterId;
            queue.push(j);
          }
        }
      }

      clusterId++;
    }

    return labels;
  }

  // Get consensus strength for each sample
  getConsensusStrength() {
    if (!this.fitted) throw new Error('Not fitted');

    const n = this.labels.length;
    const strength = new Array(n);

    for (let i = 0; i < n; i++) {
      const myCluster = this.labels[i];
      if (myCluster === -1) {
        strength[i] = 0;
        continue;
      }

      // Average co-association with cluster members
      let sum = 0;
      let count = 0;

      for (let j = 0; j < n; j++) {
        if (this.labels[j] === myCluster && i !== j) {
          sum += this.coAssocMatrix[i][j];
          count++;
        }
      }

      strength[i] = count > 0 ? sum / count : 1;
    }

    return strength;
  }
}
```

**Usage:**
```javascript
import { ConsensusCluster } from '@tangent.to/ds/clustering';
import { KMeans, DBSCAN, HDBSCAN } from '@tangent.to/ds/ml';

const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 3 }),
    new DBSCAN({ eps: 0.5, minSamples: 5 }),
    new HDBSCAN({ minClusterSize: 5 })
  ],
  threshold: 0.5  // 50% of models must agree
});

consensus.fit(data);
console.log('Consensus labels:', consensus.labels);

// Get confidence (how strong is consensus?)
const strength = consensus.getConsensusStrength();
console.log('Consensus strength:', strength);
```

### Solution 2: Label Alignment via Hungarian Algorithm

Match cluster labels by maximizing overlap:

```javascript
/**
 * Align cluster labels from multiple models
 */
function alignClusterLabels(referenceLabels, targetLabels) {
  // Build confusion matrix
  const refClusters = [...new Set(referenceLabels)].filter(l => l !== -1);
  const tgtClusters = [...new Set(targetLabels)].filter(l => l !== -1);

  const confusion = Array(refClusters.length)
    .fill(0)
    .map(() => Array(tgtClusters.length).fill(0));

  for (let i = 0; i < referenceLabels.length; i++) {
    const refIdx = refClusters.indexOf(referenceLabels[i]);
    const tgtIdx = tgtClusters.indexOf(targetLabels[i]);

    if (refIdx !== -1 && tgtIdx !== -1) {
      confusion[refIdx][tgtIdx]++;
    }
  }

  // Use Hungarian algorithm to find best matching
  // (simplified: use greedy matching for demo)
  const mapping = new Map();
  const used = new Set();

  for (let i = 0; i < refClusters.length; i++) {
    let maxOverlap = -1;
    let bestMatch = -1;

    for (let j = 0; j < tgtClusters.length; j++) {
      if (!used.has(j) && confusion[i][j] > maxOverlap) {
        maxOverlap = confusion[i][j];
        bestMatch = j;
      }
    }

    if (bestMatch !== -1) {
      mapping.set(tgtClusters[bestMatch], refClusters[i]);
      used.add(bestMatch);
    }
  }

  // Remap target labels
  return targetLabels.map(l => mapping.get(l) ?? -1);
}

// Usage:
const kmeans = new KMeans({ k: 3 });
kmeans.fit(data);

const dbscan = new DBSCAN({ eps: 0.5 });
dbscan.fit(data);

// Align DBSCAN labels to match KMeans
const alignedLabels = alignClusterLabels(kmeans.labels, dbscan.labels);
console.log('Original DBSCAN:', dbscan.labels);
console.log('Aligned DBSCAN:', alignedLabels);
```

### Solution 3: Use BranchPipeline for Regression/Classification ONLY

For supervised tasks, labels have shared meaning:

```javascript
// ✅ GOOD: Classification (labels have meaning)
const classifier = new BranchPipeline({
  branches: {
    rf: new RandomForestClassifier(),
    knn: new KNNClassifier(),
    tree: new DecisionTreeClassifier()
  },
  combiner: 'vote'
});

classifier.fit(Xtrain, ytrain);  // ytrain provides meaning
const predictions = classifier.predict(Xtest);

// ✅ GOOD: Regression (continuous values)
const regressor = new BranchPipeline({
  branches: {
    rf: new RandomForestRegressor(),
    poly: new PolynomialRegressor()
  },
  combiner: 'average'
});

// ❌ BAD: Unsupervised clustering
const bad = new BranchPipeline({
  branches: {
    kmeans: new KMeans({ k: 3 }),
    dbscan: new DBSCAN({ eps: 0.5 })
  },
  combiner: 'vote'  // MEANINGLESS without alignment!
});
```

---

## Integration with Preprocessing

### Q: Are branches pluggable in preprocessing pipelines?

**Yes!** Branches can be full pipelines:

```javascript
import { Pipeline } from '@tangent.to/ds/ml';
import { StandardScaler, PCA } from '@tangent.to/ds/preprocessing';

const ensemble = new BranchPipeline({
  branches: {
    // Branch 1: Scale then cluster
    scaled_kmeans: new Pipeline()
      .add('scaler', new StandardScaler())
      .add('cluster', new KMeans({ k: 3 })),

    // Branch 2: PCA then cluster
    pca_hdbscan: new Pipeline()
      .add('pca', new PCA({ nComponents: 2 }))
      .add('cluster', new HDBSCAN({ minClusterSize: 5 })),

    // Branch 3: Raw data
    raw_dbscan: new DBSCAN({ eps: 0.5 })
  },
  combiner: (predictions) => {
    // Custom consensus logic
    return consensusCluster(predictions);
  }
});

ensemble.fit(data);
```

### Q: Are solo models pluggable to preprocessing pipelines?

**Yes!** Use the existing `Pipeline` class:

```javascript
import { Pipeline } from '@tangent.to/ds/ml';
import { StandardScaler, PCA } from '@tangent.to/ds/preprocessing';
import { HDBSCAN } from '@tangent.to/ds/ml';

// Single model with preprocessing
const pipeline = new Pipeline()
  .add('scaler', new StandardScaler())
  .add('pca', new PCA({ nComponents: 10 }))
  .add('cluster', new HDBSCAN({ minClusterSize: 5 }));

pipeline.fit(data);
const labels = pipeline.predict(data);

// Access intermediate steps
const scaledData = pipeline.steps.scaler.transform(data);
const pcaData = pipeline.steps.pca.transform(scaledData);
```

---

## Model Persistence

### Q: How does persistence work?

Each estimator implements `toJSON()` / `fromJSON()`:

```javascript
// Single model
const hdbscan = new HDBSCAN({ minClusterSize: 5 });
hdbscan.fit(data);

// Serialize
const json = hdbscan.toJSON();
localStorage.setItem('model', JSON.stringify(json));

// Deserialize
const loaded = HDBSCAN.fromJSON(
  JSON.parse(localStorage.getItem('model'))
);
const predictions = loaded.predict(newData);
```

### BranchPipeline Persistence

```javascript
// Serialize BranchPipeline
class BranchPipeline {
  toJSON() {
    return {
      __class__: 'BranchPipeline',
      combiner: this.combiner,
      weights: this.weights,
      fitted: this.fitted,
      branches: Object.fromEntries(
        Object.entries(this.branches).map(([name, est]) => [
          name,
          est.toJSON()  // Serialize each branch
        ])
      ),
      // Store training data for clustering alignment
      X_train: this.X_train
    };
  }

  static fromJSON(obj, estimatorRegistry) {
    // Reconstruct each branch
    const branches = {};
    for (const [name, branchJson] of Object.entries(obj.branches)) {
      const EstimatorClass = estimatorRegistry[branchJson.__class__];
      branches[name] = EstimatorClass.fromJSON(branchJson);
    }

    const pipeline = new BranchPipeline({
      branches,
      combiner: obj.combiner,
      weights: obj.weights
    });

    pipeline.fitted = obj.fitted;
    pipeline.X_train = obj.X_train;

    return pipeline;
  }
}

// Usage:
const ensemble = new BranchPipeline({
  branches: {
    rf: new RandomForestClassifier(),
    knn: new KNNClassifier()
  }
});

ensemble.fit(Xtrain, ytrain);

// Save
const json = ensemble.toJSON();
fs.writeFileSync('ensemble.json', JSON.stringify(json));

// Load
const loaded = BranchPipeline.fromJSON(
  JSON.parse(fs.readFileSync('ensemble.json')),
  { RandomForestClassifier, KNNClassifier }  // Registry
);

const predictions = loaded.predict(Xtest);
```

---

## When to Use What

### ✅ Use BranchPipeline (with voting) for:

1. **Classification** - Labels have meaning
   ```javascript
   // Predict species from features
   branches: {
     rf: new RandomForestClassifier(),
     knn: new KNNClassifier()
   }
   ```

2. **Regression** - Continuous values (use average)
   ```javascript
   branches: {
     rf: new RandomForestRegressor(),
     poly: new PolynomialRegressor()
   },
   combiner: 'average'
   ```

### ❌ DON'T use BranchPipeline for:

1. **Unsupervised Clustering** - Use `ConsensusCluster` instead

2. **Different preprocessing paths** - Use `FeatureUnion`

### ✅ Use ConsensusCluster for:

1. **Ensemble clustering** - Robust to algorithm choice
2. **Uncertainty quantification** - Get consensus strength
3. **Exploring data structure** - Compare different algorithms

---

## Correct Examples

### Example 1: Consensus Clustering

```javascript
import { ConsensusCluster } from '@tangent.to/ds/clustering';

const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 3 }),
    new DBSCAN({ eps: 0.5 }),
    new HDBSCAN({ minClusterSize: 5 })
  ],
  threshold: 0.6  // 60% agreement required
});

consensus.fit(data);
const labels = consensus.labels;
const strength = consensus.getConsensusStrength();

// Filter low-confidence points
const reliable = labels.map((l, i) =>
  strength[i] > 0.7 ? l : -1
);
```

### Example 2: Classification Ensemble

```javascript
const classifier = new BranchPipeline({
  branches: {
    rf: new RandomForestClassifier({ nEstimators: 100 }),
    knn: new KNNClassifier({ k: 5 }),
    tree: new DecisionTreeClassifier({ maxDepth: 10 })
  },
  combiner: 'vote',
  weights: [2, 1, 1]  // Trust RF more
});

classifier.fit(Xtrain, ytrain);
const yPred = classifier.predict(Xtest);

// Save model
fs.writeFileSync('model.json', JSON.stringify(classifier.toJSON()));
```

### Example 3: Preprocessing Integration

```javascript
import { Recipe } from '@tangent.to/ds/ml';

// Preprocess then cluster
const recipe = Recipe({ data, X: ['x', 'y'], y: null })
  .scale(['x', 'y'])
  .pca({ nComponents: 2 })
  .prep();

const trainData = recipe.train.X;

// Apply consensus clustering
const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 3 }),
    new HDBSCAN({ minClusterSize: 5 })
  ]
});

consensus.fit(trainData);

// Apply to new data
const newPrepped = recipe.bake(newData);
// Note: Clustering doesn't have predict() - need different approach
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Label alignment problem?** | ✅ Yes - critical issue! Use ConsensusCluster or alignment |
| **Branches in pipelines?** | ✅ Yes - branches can be full Pipeline objects |
| **Solo models in pipelines?** | ✅ Yes - use existing Pipeline class |
| **Prediction?** | ⚠️ Works for classification/regression, not clustering |
| **Persistence?** | ✅ Yes - via toJSON()/fromJSON() |

**Key Takeaway:** BranchPipeline with voting is only meaningful for **supervised learning** (classification/regression). For clustering, use **ConsensusCluster** with co-association matrix instead!
