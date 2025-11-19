# Consensus Clustering - The Right Way

## Problem: Why BranchPipeline Doesn't Work for Clustering

```javascript
// ❌ WRONG: Naive voting on cluster labels
const bad = new BranchPipeline({
  branches: {
    kmeans: new KMeans({ k: 3 }),
    dbscan: new DBSCAN({ eps: 0.5 })
  },
  combiner: 'vote'  // MEANINGLESS!
});

// Labels have no shared meaning:
// KMeans: [0, 0, 1, 1, 2]  // 0=left, 1=right, 2=middle
// DBSCAN: [1, 1, 0, 0, -1] // 1=left, 0=right, -1=noise
// Voting on these is nonsense!
```

## Solution: ConsensusCluster

Instead of voting on arbitrary labels, measure **pairwise co-occurrence**:
- "Do samples i and j belong to the same cluster?"
- Aggregate across all models
- Extract final clusters from co-association matrix

```javascript
import { ConsensusCluster } from '@tangent.to/ds/clustering';
import { KMeans, DBSCAN, HDBSCAN } from '@tangent.to/ds/ml';

// ✅ CORRECT: Consensus via co-association
const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 3 }),
    new DBSCAN({ eps: 0.5, minSamples: 5 }),
    new HDBSCAN({ minClusterSize: 5 })
  ],
  threshold: 0.5  // 50% of models must agree points are together
});

consensus.fit(data);
console.log('Consensus labels:', consensus.labels);
```

---

## Basic Usage

```javascript
import { ConsensusCluster } from '@tangent.to/ds/clustering';
import { KMeans, DBSCAN, HDBSCAN } from '@tangent.to/ds/ml';

// Sample data: Two clusters + noise
const data = [
  // Cluster 1
  [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [0.25, 0.25],
  [0.1, 0.1], [0.4, 0.4],
  // Cluster 2
  [10, 10], [10.5, 10], [10, 10.5], [10.5, 10.5], [10.25, 10.25],
  [10.1, 10.1], [10.4, 10.4],
  // Noise
  [5, 5], [5.1, 5.1]
];

// Create consensus with multiple algorithms
const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 2 }),
    new DBSCAN({ eps: 1.0, minSamples: 3 }),
    new HDBSCAN({ minClusterSize: 3 })
  ],
  threshold: 0.5  // At least 50% agreement
});

// Fit
consensus.fit(data);

// Get results
console.log('Labels:', consensus.labels);
console.log('Number of clusters:', consensus.nClusters);
console.log('Number of noise points:', consensus.nNoise);

// Get confidence
const strength = consensus.getConsensusStrength();
console.log('Consensus strength:', strength);
// [0.95, 0.92, 0.98, ...] - high values = strong agreement

// Overall agreement
console.log('Overall agreement:', consensus.agreementScore);
// 0.75 = 75% of models agree on pairwise relationships
```

---

## Understanding Co-Association Matrix

```javascript
// After fitting:
const coAssoc = consensus.coAssocMatrix;

// coAssoc[i][j] = fraction of models that put i and j together
console.log(coAssoc[0][1]);  // e.g., 1.0 = all models agree
console.log(coAssoc[0][10]); // e.g., 0.0 = no models agree
console.log(coAssoc[0][15]); // e.g., 0.33 = 1/3 models agree (noise)

// Visualize as heatmap
import * as Plot from '@observablehq/plot';

const heatmapData = [];
for (let i = 0; i < coAssoc.length; i++) {
  for (let j = 0; j < coAssoc.length; j++) {
    heatmapData.push({ x: i, y: j, value: coAssoc[i][j] });
  }
}

const plot = Plot.plot({
  marks: [
    Plot.cell(heatmapData, {
      x: 'x',
      y: 'y',
      fill: 'value',
      inset: 0.5
    })
  ],
  color: { scheme: 'YlGnBu', legend: true }
});
```

---

## Adjusting Consensus Threshold

```javascript
// Strict consensus (70% agreement required)
const strict = new ConsensusCluster({
  estimators: [kmeans, dbscan, hdbscan],
  threshold: 0.7
});
strict.fit(data);
// Result: Fewer, more confident clusters

// Loose consensus (30% agreement required)
const loose = new ConsensusCluster({
  estimators: [kmeans, dbscan, hdbscan],
  threshold: 0.3
});
loose.fit(data);
// Result: More, less confident clusters

// Compare
console.log('Strict clusters:', strict.nClusters);
console.log('Loose clusters:', loose.nClusters);
```

---

## Filtering Low-Confidence Points

```javascript
const consensus = new ConsensusCluster({
  estimators: [kmeans, dbscan, hdbscan],
  threshold: 0.5
});

consensus.fit(data);

const strength = consensus.getConsensusStrength();
const labels = consensus.labels;

// Only keep high-confidence assignments
const minStrength = 0.7;
const reliableLabels = labels.map((label, i) =>
  strength[i] >= minStrength ? label : -1  // Mark as uncertain
);

console.log('Original labels:', labels);
console.log('High-confidence only:', reliableLabels);
console.log('Uncertain points:', reliableLabels.filter(l => l === -1).length);
```

---

## Comparing Estimator Contributions

```javascript
const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 3 }),
    new DBSCAN({ eps: 0.5 }),
    new HDBSCAN({ minClusterSize: 5 })
  ]
});

consensus.fit(data);

// See how each estimator performed
const agreement = consensus.getEstimatorAgreement();
console.log(agreement);
/*
[
  { estimator: 'KMeans', ari: 0.85, nClusters: 3, nNoise: 0 },
  { estimator: 'DBSCAN', ari: 0.78, nClusters: 2, nNoise: 5 },
  { estimator: 'HDBSCAN', ari: 0.92, nClusters: 2, nNoise: 3 }
]
*/

// HDBSCAN agrees most with consensus (ARI=0.92)
```

---

## Integration with Preprocessing

```javascript
import { Recipe } from '@tangent.to/ds/ml';
import { StandardScaler, PCA } from '@tangent.to/ds/preprocessing';

// Preprocess data
const recipe = Recipe({ data, X: ['x', 'y', 'z'], y: null })
  .scale(['x', 'y', 'z'])
  .pca({ nComponents: 2 })
  .prep();

const trainData = recipe.train.X;

// Apply consensus clustering
const consensus = new ConsensusCluster({
  estimators: [
    new KMeans({ k: 3 }),
    new DBSCAN({ eps: 0.5 })
  ]
});

consensus.fit(trainData);

// Apply to new data (note: no predict() for clustering)
const testPrepped = recipe.bake(testData);
// Would need to refit or use nearest neighbor assignment
```

---

## Ensemble with Different Configurations

```javascript
// Test multiple parameter settings
const consensus = new ConsensusCluster({
  estimators: [
    // Multiple K-Means with different k
    new KMeans({ k: 2 }),
    new KMeans({ k: 3 }),
    new KMeans({ k: 4 }),

    // Multiple DBSCAN with different eps
    new DBSCAN({ eps: 0.3, minSamples: 5 }),
    new DBSCAN({ eps: 0.5, minSamples: 5 }),
    new DBSCAN({ eps: 0.7, minSamples: 5 }),

    // HDBSCAN
    new HDBSCAN({ minClusterSize: 5 })
  ],
  threshold: 0.4  // Lower threshold since more variance
});

consensus.fit(data);

// More robust to parameter choices
console.log('Consensus from 7 runs:', consensus.labels);
console.log('Agreement:', consensus.agreementScore);
```

---

## Visualization

```javascript
import { plotClusterMembership } from '@tangent.to/ds/plot';
import * as Plot from '@observablehq/plot';

const consensus = new ConsensusCluster({
  estimators: [kmeans, dbscan, hdbscan]
});

consensus.fit(data);

const strength = consensus.getConsensusStrength();

// Scatter plot with consensus strength as opacity
const vizData = data.map((point, i) => ({
  x: point[0],
  y: point[1],
  cluster: String(consensus.labels[i]),
  strength: strength[i]
}));

const plot = Plot.plot({
  marks: [
    Plot.dot(vizData, {
      x: 'x',
      y: 'y',
      fill: 'cluster',
      opacity: 'strength',  // Fade out low-confidence points
      r: 6,
      tip: true
    })
  ],
  color: { legend: true }
});
```

---

## Comparison Dashboard

```javascript
function createConsensusDashboard(consensus, data) {
  const labels = consensus.labels;
  const strength = consensus.getConsensusStrength();
  const clusterStrength = consensus.getClusterStrength();
  const agreement = consensus.getEstimatorAgreement();

  return {
    summary: consensus.summary(),
    perSample: data.map((point, i) => ({
      x: point[0],
      y: point[1],
      cluster: labels[i],
      strength: strength[i],
      reliable: strength[i] > 0.7
    })),
    perCluster: Object.entries(clusterStrength).map(([cluster, strength]) => ({
      cluster: Number(cluster),
      strength,
      size: labels.filter(l => l === Number(cluster)).length
    })),
    estimators: agreement
  };
}

// Usage
const dashboard = createConsensusDashboard(consensus, data);
console.log('Dashboard:', dashboard);

// Find problematic samples
const uncertain = dashboard.perSample
  .filter(s => s.strength < 0.5)
  .map(s => ({ x: s.x, y: s.y }));

console.log('Uncertain samples:', uncertain);
```

---

## When to Use ConsensusCluster

### ✅ Good for:

1. **Exploratory analysis** - Not sure which algorithm to use
2. **Robust clustering** - Reduce algorithm bias
3. **Uncertainty quantification** - Get confidence scores
4. **Ambiguous data** - Natural cluster boundaries unclear
5. **No ground truth** - Can't validate algorithm choice

### ❌ Not ideal for:

1. **Real-time applications** - Slower than single model
2. **Very large datasets** - O(n²) co-association matrix
3. **Clear structure** - Single algorithm sufficient
4. **Memory constraints** - Stores multiple models

---

## Advanced: Custom Ensemble Strategy

```javascript
// Create custom consensus based on specific needs
class WeightedConsensus extends ConsensusCluster {
  constructor({ estimators, weights, threshold }) {
    super({ estimators, threshold });
    this.weights = weights;  // Weight each estimator
  }

  _updateCoAssociation(coAssoc, labels, estimatorIdx) {
    const n = labels.length;
    const weight = this.weights[estimatorIdx];

    for (let i = 0; i < n; i++) {
      if (labels[i] === -1) continue;

      for (let j = i; j < n; j++) {
        if (labels[j] === -1) continue;

        if (labels[i] === labels[j]) {
          coAssoc[i][j] += weight;
          if (i !== j) coAssoc[j][i] += weight;
        }
      }
    }
  }
}

// Usage: Trust HDBSCAN more than others
const weighted = new WeightedConsensus({
  estimators: [kmeans, dbscan, hdbscan],
  weights: [1, 1, 2],  // HDBSCAN gets 2x weight
  threshold: 0.5
});
```

---

## Summary

| Approach | Clustering | Classification | Regression |
|----------|------------|----------------|------------|
| **BranchPipeline** | ❌ Wrong | ✅ Works | ✅ Works |
| **ConsensusCluster** | ✅ Correct | N/A | N/A |

**Key Insight:** Clustering labels are arbitrary identifiers with no shared meaning. Consensus clustering works by measuring pairwise co-occurrence, not voting on labels!
