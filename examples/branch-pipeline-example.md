# Branch Pipeline Example

Demonstrates how to use `BranchPipeline` to run multiple algorithms in parallel and combine their results for consensus clustering.

## Basic Consensus Clustering

```javascript
import { BranchPipeline } from '@tangent.to/ds/pipeline';
import { KMeans, DBSCAN, HDBSCAN } from '@tangent.to/ds/ml';

// Sample data: two well-separated clusters
const data = [
  // Cluster 1
  [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
  [0.1, 0.1], [0.4, 0.4], [0.3, 0.2],
  // Cluster 2
  [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25],
  [10.1, 10.1], [10.4, 10.4], [10.3, 10.2],
  // Noise/outliers
  [5, 5], [-2, -2]
];

// Create branch pipeline with three clustering algorithms
const consensusPipeline = new BranchPipeline({
  branches: {
    kmeans: new KMeans({ k: 2 }),
    dbscan: new DBSCAN({ eps: 1.0, minSamples: 3 }),
    hdbscan: new HDBSCAN({ minClusterSize: 3 })
  },
  combiner: 'vote'  // Majority voting
});

// Fit all algorithms
consensusPipeline.fit(data);

// Get consensus predictions
const consensusLabels = consensusPipeline.predict(data);
console.log('Consensus labels:', consensusLabels);

// Get individual branch predictions
const allPredictions = consensusPipeline.predictAll(data);
console.log('K-Means labels:', allPredictions.kmeans);
console.log('DBSCAN labels:', allPredictions.dbscan);
console.log('HDBSCAN labels:', allPredictions.hdbscan);

// Check agreement between algorithms
const agreement = consensusPipeline.agreementScore(data);
console.log('Agreement score:', agreement);
// Output: 0.85 (85% of points have same label across all algorithms)

// Get per-point confidence
const confidence = consensusPipeline.confidence(data);
console.log('Confidence scores:', confidence);
// Higher confidence = more algorithms agree on the label
```

## Weighted Voting

Give more weight to certain algorithms:

```javascript
const weightedPipeline = new BranchPipeline({
  branches: {
    hdbscan: new HDBSCAN({ minClusterSize: 5 }),
    dbscan: new DBSCAN({ eps: 0.5, minSamples: 5 }),
    kmeans: new KMeans({ k: 3 })
  },
  combiner: 'weighted_vote',
  weights: [3, 2, 1]  // HDBSCAN gets 3x weight, DBSCAN 2x, KMeans 1x
});

weightedPipeline.fit(data);
const labels = weightedPipeline.predict(data);
```

## Ensemble with Different Preprocessing

```javascript
import { Pipeline } from '@tangent.to/ds/ml';
import { StandardScaler, PCA } from '@tangent.to/ds/preprocessing';

const ensemble = new BranchPipeline({
  branches: {
    // Branch 1: Raw data + K-Means
    raw_kmeans: new KMeans({ k: 3 }),

    // Branch 2: Scaled + HDBSCAN
    scaled_hdbscan: new Pipeline()
      .add('scaler', new StandardScaler())
      .add('cluster', new HDBSCAN({ minClusterSize: 5 })),

    // Branch 3: PCA + K-Means
    pca_kmeans: new Pipeline()
      .add('pca', new PCA({ nComponents: 2 }))
      .add('cluster', new KMeans({ k: 3 }))
  },
  combiner: 'vote'
});

ensemble.fit(highDimData);
const robustLabels = ensemble.predict(highDimData);
```

## Custom Combiner Function

```javascript
// Custom combiner that uses median instead of voting
const customCombiner = (branchPredictions) => {
  const n = branchPredictions[0].predictions.length;
  const combined = new Array(n);

  for (let i = 0; i < n; i++) {
    const values = branchPredictions
      .map(({ predictions }) => predictions[i])
      .sort((a, b) => a - b);

    // Take median value
    const mid = Math.floor(values.length / 2);
    combined[i] = values.length % 2 === 0
      ? (values[mid - 1] + values[mid]) / 2
      : values[mid];
  }

  return combined;
};

const pipeline = new BranchPipeline({
  branches: { /* ... */ },
  combiner: customCombiner
});
```

## Regression Ensemble

```javascript
import {
  RandomForestRegressor,
  KNNRegressor,
  PolynomialRegressor
} from '@tangent.to/ds/ml';

const regressionEnsemble = new BranchPipeline({
  branches: {
    rf: new RandomForestRegressor({ nEstimators: 100 }),
    knn: new KNNRegressor({ k: 5 }),
    poly: new PolynomialRegressor({ degree: 3 })
  },
  combiner: 'average'  // Average predictions for regression
});

regressionEnsemble.fit(Xtrain, ytrain);
const predictions = regressionEnsemble.predict(Xtest);

// Check model agreement
const confidence = regressionEnsemble.confidence(Xtest);
// Low confidence = models disagree (might indicate difficult samples)
```

## Classification Ensemble

```javascript
import {
  RandomForestClassifier,
  KNNClassifier,
  DecisionTreeClassifier
} from '@tangent.to/ds/ml';

const classifier = new BranchPipeline({
  branches: {
    rf: new RandomForestClassifier({ nEstimators: 100 }),
    knn: new KNNClassifier({ k: 5 }),
    tree: new DecisionTreeClassifier({ maxDepth: 10 })
  },
  combiner: 'vote',
  weights: [2, 1, 1]  // Give more weight to Random Forest
});

classifier.fit(Xtrain, ytrain);
const yPred = classifier.predict(Xtest);

// Analyze predictions
console.log('Summary:', classifier.summary());
```

## Anomaly Detection Ensemble

```javascript
import {
  IsolationForest,
  LocalOutlierFactor,
  DBSCAN
} from '@tangent.to/ds/ml';

const anomalyDetector = new BranchPipeline({
  branches: {
    iforest: new IsolationForest({ contamination: 0.1 }),
    lof: new LocalOutlierFactor({ nNeighbors: 20 }),
    dbscan: new DBSCAN({ eps: 0.5, minSamples: 5 })
  },
  combiner: (branchPredictions) => {
    // Custom: Mark as anomaly if at least 2 out of 3 agree
    const n = branchPredictions[0].predictions.length;
    const combined = new Array(n);

    for (let i = 0; i < n; i++) {
      const anomalyCount = branchPredictions
        .filter(({ predictions }) => predictions[i] === -1)
        .length;

      combined[i] = anomalyCount >= 2 ? -1 : 0;
    }

    return combined;
  }
});

anomalyDetector.fit(data);
const anomalies = anomalyDetector.predict(data);
// -1 = anomaly, 0 = normal
```

## Confidence-Based Selection

```javascript
// Only use predictions where models agree
const data = [/* ... */];

const pipeline = new BranchPipeline({
  branches: {
    model1: new HDBSCAN({ minClusterSize: 5 }),
    model2: new DBSCAN({ eps: 0.5 }),
    model3: new KMeans({ k: 3 })
  },
  combiner: 'vote'
});

pipeline.fit(data);

const labels = pipeline.predict(data);
const confidence = pipeline.confidence(data);

// Filter low-confidence predictions
const reliableLabels = labels.map((label, i) =>
  confidence[i] >= 0.66 ? label : -1  // -1 for uncertain
);

console.log('High confidence predictions:',
  reliableLabels.filter(l => l !== -1).length
);
```

## Visualizing Branch Agreement

```javascript
import { plotClusterMembership } from '@tangent.to/ds/plot';
import * as Plot from '@observablehq/plot';

const pipeline = new BranchPipeline({
  branches: {
    hdbscan: new HDBSCAN({ minClusterSize: 5 }),
    dbscan: new DBSCAN({ eps: 0.5 }),
    kmeans: new KMeans({ k: 3 })
  },
  combiner: 'vote'
});

pipeline.fit(data);

const labels = pipeline.predict(data);
const confidence = pipeline.confidence(data);
const allPredictions = pipeline.predictAll(data);

// Create visualization data
const vizData = data.map((point, i) => ({
  x: point[0],
  y: point[1],
  cluster: labels[i],
  confidence: confidence[i],
  kmeans: allPredictions.kmeans[i],
  dbscan: allPredictions.dbscan[i],
  hdbscan: allPredictions.hdbscan[i]
}));

// Plot with confidence as opacity
const spec = {
  marks: [
    Plot.dot(vizData, {
      x: 'x',
      y: 'y',
      fill: 'cluster',
      opacity: 'confidence',
      r: 5,
      tip: true
    })
  ]
};

const svg = Plot.plot(spec);
```

## Comparison Dashboard

```javascript
// Compare all branch results side-by-side
function createComparisonDashboard(pipeline, data) {
  const allPredictions = pipeline.predictAll(data);
  const consensus = pipeline.predict(data);
  const confidence = pipeline.confidence(data);

  const dashboard = {
    data: data.map((point, i) => ({
      x: point[0],
      y: point[1],
      consensus: consensus[i],
      confidence: confidence[i],
      ...Object.fromEntries(
        Object.entries(allPredictions).map(([name, labels]) =>
          [name, labels[i]]
        )
      )
    })),
    metrics: {
      agreement: pipeline.agreementScore(data),
      avgConfidence: confidence.reduce((a, b) => a + b, 0) / confidence.length,
      nBranches: Object.keys(pipeline.branches).length,
      branches: Object.keys(pipeline.branches)
    }
  };

  return dashboard;
}

// Usage
const dashboard = createComparisonDashboard(consensusPipeline, data);
console.log('Dashboard:', dashboard);
```

## A/B Testing Different Strategies

```javascript
// Test multiple ensemble strategies
const strategies = ['vote', 'weighted_vote', 'average'];

const results = strategies.map(strategy => {
  const pipeline = new BranchPipeline({
    branches: {
      hdbscan: new HDBSCAN({ minClusterSize: 5 }),
      dbscan: new DBSCAN({ eps: 0.5 }),
      kmeans: new KMeans({ k: 3 })
    },
    combiner: strategy,
    weights: strategy === 'weighted_vote' ? [3, 2, 1] : null
  });

  pipeline.fit(data);
  const labels = pipeline.predict(data);
  const confidence = pipeline.confidence(data);

  return {
    strategy,
    labels,
    avgConfidence: confidence.reduce((a, b) => a + b) / confidence.length
  };
});

// Compare strategies
results.forEach(result => {
  console.log(`${result.strategy}: avg confidence = ${result.avgConfidence}`);
});
```

## Best Practices

1. **Use different algorithm types** - Combine algorithms with different strengths
   - K-Means: Fast, good for spherical clusters
   - DBSCAN: Good for arbitrary shapes
   - HDBSCAN: Good for varying density

2. **Check agreement scores** - Low agreement may indicate:
   - Ambiguous data
   - Need for more preprocessing
   - Inappropriate algorithm choices

3. **Use confidence for filtering** - High-confidence predictions are more reliable

4. **Weight algorithms appropriately** - Give more weight to algorithms that perform better on your data type

5. **Validate with ground truth** - Compare ensemble vs individual algorithms on labeled data

## When to Use Branch Pipelines

✅ **Good for:**
- Consensus clustering on ambiguous data
- Robust predictions when no single algorithm is clearly best
- Exploring different preprocessing strategies
- A/B testing algorithm configurations

❌ **Avoid when:**
- Single algorithm clearly outperforms others
- Real-time performance is critical (branching adds overhead)
- Memory is constrained (stores multiple models)
- Interpretability is paramount (ensemble is harder to explain)
