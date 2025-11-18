# Complete Branching Guide

This document explains both **Git Branching** (version control) and **Pipeline Branching** (ML concept).

---

## Part 1: Git Branching Strategy

### Visual Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         MAIN BRANCH                          │
│  ═══════════════════════════════════════════════════════    │
└─────────────────────────────────────────────────────────────┘
       │          │          │          │
       │          │          │          └──> hotfix/critical-bug-01XYZ
       │          │          │                └─> merge back quickly
       │          │          │
       │          │          └──> feature/add-spectral-clustering-01ABC
       │          │                └─> develop, test, PR, merge
       │          │
       │          └──> bugfix/hdbscan-performance-01DEF
       │                └─> fix, test, PR, merge
       │
       └──> refactor/v1.0-reorganization
              └─> long-running reorganization branch
                    │
                    ├──> feature/move-clustering-01GHI
                    ├──> feature/move-classification-01JKL
                    └──> feature/move-regression-01MNO
                          └─> merge to refactor/v1.0
                                └─> eventually merge to main
```

### Branch Types

| Type | Purpose | Lifetime | Example |
|------|---------|----------|---------|
| `main` | Production code | Permanent | - |
| `feature/*` | New features | Days to weeks | `feature/add-umap-01ABC` |
| `bugfix/*` | Bug fixes | Days | `bugfix/kmeans-init-01DEF` |
| `hotfix/*` | Critical fixes | Hours | `hotfix/security-patch-01XYZ` |
| `refactor/*` | Code reorganization | Weeks to months | `refactor/v1.0-reorg` |
| `docs/*` | Documentation only | Days | `docs/add-tutorials` |
| `test/*` | Test additions | Days | `test/add-benchmarks` |
| `claude/*` | AI-assisted dev | Variable | `claude/implement-hdbscan-...` |

### Workflow Steps

#### 1. Create Branch

```bash
# Always start from updated main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/add-optics-clustering-01ABC
```

#### 2. Develop

```bash
# Make changes
git add src/clustering/optics.js
git commit -m "feat: implement OPTICS clustering algorithm"

# Push regularly (creates remote branch)
git push -u origin feature/add-optics-clustering-01ABC
```

#### 3. Keep Updated

```bash
# Option A: Rebase (cleaner history - preferred)
git fetch origin
git rebase origin/main

# Resolve conflicts if any
# git add <resolved-files>
# git rebase --continue

# Force push after rebase (safe on feature branch)
git push --force-with-lease

# Option B: Merge (preserves history)
git merge origin/main
git push
```

#### 4. Create Pull Request

```bash
# Using GitHub CLI
gh pr create \
  --title "Add OPTICS clustering algorithm" \
  --body "Implements OPTICS clustering with consistent API"

# Or via web interface
# Visit: https://github.com/tangent-to/ds/pull/new/feature/add-optics-clustering-01ABC
```

#### 5. Review & Merge

```bash
# After approval, merge via GitHub UI or:
gh pr merge --squash  # Squash all commits into one
# or
gh pr merge --merge   # Keep all commits
# or
gh pr merge --rebase  # Rebase and merge
```

#### 6. Cleanup

```bash
# Delete local branch
git checkout main
git branch -d feature/add-optics-clustering-01ABC

# Delete remote branch (usually automatic after merge)
git push origin --delete feature/add-optics-clustering-01ABC
```

---

## Part 2: Pipeline Branching (ML Concept)

### Visual Overview

```
┌──────────────┐
│  INPUT DATA  │
└──────┬───────┘
       │
       ├────────────────┬────────────────┬────────────────┐
       │                │                │                │
       ▼                ▼                ▼                ▼
  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
  │ Branch 1│      │ Branch 2│      │ Branch 3│      │ Branch 4│
  │ K-Means │      │ DBSCAN  │      │ HDBSCAN │      │   HCA   │
  └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘
       │                │                │                │
       │   Labels       │   Labels       │   Labels       │   Labels
       │   [0,0,1,1]    │   [0,0,1,1]    │   [1,1,0,0]    │   [0,0,1,1]
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │   COMBINER    │
                        │  (Majority    │
                        │   Voting)     │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ FINAL LABELS  │
                        │   [0,0,1,1]   │
                        │ Confidence:   │
                        │ [1.0,1.0,0.75,│
                        │     1.0]      │
                        └───────────────┘
```

### Types of Pipeline Branching

#### 1. Parallel Branching (Ensemble)

Run multiple models in parallel, combine results:

```
Data → [Model 1, Model 2, Model 3] → Vote → Result
```

**Use case:** Consensus clustering, robust predictions

#### 2. Conditional Branching

Route data to different models based on conditions:

```
         ┌─> Small dataset → Simple Model
         │
Data ──Route
         │
         └─> Large dataset → Complex Model
```

**Use case:** Adaptive modeling, performance optimization

#### 3. Feature Branching (Feature Union)

Process different features separately, combine:

```
         ┌─> Numeric features → Scale →
         │
Data ──Split                           ├─> Concat → Model
         │
         └─> Categorical → OneHot →
```

**Use case:** Heterogeneous data, different preprocessing per feature type

#### 4. Sequential Branching (Stacking)

Use predictions from multiple models as features:

```
         ┌─> Model 1 ─┐
         │            │
Data ──>─┤─> Model 2 ─┤─> [Pred1, Pred2, Pred3] ─> Meta Model → Final
         │            │
         └─> Model 3 ─┘
```

**Use case:** Advanced ensembles, stacking

### Comparison Table

| Concept | Domain | Purpose | Merging |
|---------|--------|---------|---------|
| **Git Branching** | Version control | Parallel development | Code merge via PR |
| **Pipeline Branching** | Machine learning | Parallel modeling | Prediction combination |

### When to Use Each

#### Git Branching
- ✅ Working on new features
- ✅ Multiple developers
- ✅ Testing experimental changes
- ✅ Maintaining stable releases

#### Pipeline Branching
- ✅ Uncertain which algorithm is best
- ✅ Want robust predictions
- ✅ Ensemble learning
- ✅ Different preprocessing strategies

---

## Part 3: Combined Example

Here's how both branching concepts work together:

### Scenario: Implementing Ensemble Clustering

**Git Branching (Development):**
```bash
# Create feature branch for ensemble implementation
git checkout -b feature/add-ensemble-clustering-01XYZ

# Develop BranchPipeline class
git add src/pipeline/BranchPipeline.js
git commit -m "feat: add BranchPipeline for ensemble learning"

# Develop tests
git add tests/branch-pipeline.test.js
git commit -m "test: add BranchPipeline tests"

# Push and create PR
git push -u origin feature/add-ensemble-clustering-01XYZ
gh pr create --title "Add ensemble clustering via BranchPipeline"
```

**Pipeline Branching (Usage):**
```javascript
// User code using the new BranchPipeline feature
import { BranchPipeline } from '@tangent.to/ds/pipeline';
import { KMeans, DBSCAN, HDBSCAN } from '@tangent.to/ds/clustering';

// Create ensemble with parallel branches
const ensemble = new BranchPipeline({
  branches: {
    kmeans: new KMeans({ k: 3 }),
    dbscan: new DBSCAN({ eps: 0.5 }),
    hdbscan: new HDBSCAN({ minClusterSize: 5 })
  },
  combiner: 'vote'
});

// Use the ensemble
ensemble.fit(data);
const labels = ensemble.predict(data);
```

---

## Part 4: Best Practices

### Git Branching Best Practices

1. **Name branches descriptively**
   ```bash
   # Good
   feature/add-spectral-clustering-01ABC
   bugfix/hdbscan-memory-leak-01DEF

   # Bad
   my-branch
   test
   new-stuff
   ```

2. **Keep branches short-lived**
   - Feature branches: < 1 week
   - Bug fix branches: < 2 days
   - Long-running only for major refactors

3. **Commit often, push regularly**
   ```bash
   # Make atomic commits
   git commit -m "feat: add core distance calculation"
   git commit -m "feat: add MST construction"
   git commit -m "test: add HDBSCAN tests"
   ```

4. **Rebase before merging**
   ```bash
   git fetch origin
   git rebase origin/main
   git push --force-with-lease
   ```

5. **Write good commit messages**
   ```
   feat: add HDBSCAN clustering algorithm

   - Implement core distance calculation
   - Build minimum spanning tree
   - Extract stable clusters
   - Add comprehensive tests

   Closes #123
   ```

### Pipeline Branching Best Practices

1. **Use different algorithm types**
   ```javascript
   // Good: Different strengths
   branches: {
     partitioning: new KMeans({ k: 3 }),
     density: new HDBSCAN({ minClusterSize: 5 }),
     hierarchical: new HCA({ linkage: 'ward' })
   }

   // Avoid: Too similar
   branches: {
     kmeans1: new KMeans({ k: 2 }),
     kmeans2: new KMeans({ k: 3 }),
     kmeans3: new KMeans({ k: 4 })
   }
   ```

2. **Check agreement scores**
   ```javascript
   const agreement = pipeline.agreementScore(data);
   if (agreement < 0.5) {
     console.warn('Low agreement - consider different algorithms');
   }
   ```

3. **Use confidence for filtering**
   ```javascript
   const confidence = pipeline.confidence(data);
   const reliable = labels.map((l, i) =>
     confidence[i] > 0.66 ? l : -1
   );
   ```

4. **Weight algorithms appropriately**
   ```javascript
   // Give more weight to historically better performers
   weights: [3, 2, 1]  // HDBSCAN, DBSCAN, K-Means
   ```

5. **Profile performance**
   ```javascript
   console.time('ensemble');
   pipeline.fit(data);
   console.timeEnd('ensemble');
   // Ensure acceptable runtime
   ```

---

## Part 5: Troubleshooting

### Git Branching Issues

**Problem:** Conflicts when rebasing
```bash
# Solution
git rebase origin/main
# Fix conflicts in files
git add <resolved-files>
git rebase --continue
```

**Problem:** Accidentally committed to main
```bash
# Solution: Move commits to feature branch
git branch feature/my-changes
git reset --hard origin/main
git checkout feature/my-changes
```

**Problem:** Need to update PR with new commits
```bash
# Solution: Amend or add commits
git commit --amend  # Modify last commit
# or
git commit -m "Additional changes"
git push --force-with-lease
```

### Pipeline Branching Issues

**Problem:** Low agreement between branches
```javascript
// Solution: Analyze individual predictions
const allPreds = pipeline.predictAll(data);
console.table(
  data.map((_, i) => ({
    index: i,
    model1: allPreds.model1[i],
    model2: allPreds.model2[i],
    model3: allPreds.model3[i]
  }))
);
```

**Problem:** One branch dominates voting
```javascript
// Solution: Use weighted voting
const pipeline = new BranchPipeline({
  branches: { /* ... */ },
  combiner: 'weighted_vote',
  weights: [1, 1, 1]  // Equal weight
});
```

**Problem:** Slow performance
```javascript
// Solution: Profile and optimize
const times = {};
for (const [name, model] of Object.entries(pipeline.branches)) {
  console.time(name);
  model.fit(data);
  console.timeEnd(name);
}
// Remove slowest branches if not improving accuracy
```

---

## Summary

| Aspect | Git Branching | Pipeline Branching |
|--------|---------------|-------------------|
| **Purpose** | Parallel code development | Parallel model execution |
| **Creates** | Code branches | Model branches |
| **Merges via** | Pull requests | Prediction combination |
| **Result** | Updated codebase | Consensus predictions |
| **Used by** | Developers | Data scientists |
| **Tools** | Git, GitHub | BranchPipeline |

Both concepts share the idea of **parallel paths that converge**, but operate in different domains!
