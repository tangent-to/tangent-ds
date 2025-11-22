# Model Safeguards & Optimization Recommendations

## Executive Summary

This document provides recommendations for improving model safeguards, memory optimization, and user warnings in the `@tangent.to/ds` library, particularly for Observable environments where code execution patterns and memory constraints differ from Node.js.

**Key Findings:**
- ✅ Most estimator classes have fitted checks in place
- ⚠️ Observable-specific challenges: reactive cells, memory constraints, code run without fitting
- 🔄 Opportunities for lazy evaluation and memory optimization
- 📊 Need for better user feedback on model state

---

## 1. Current State Analysis

### 1.1 Existing Safeguards (Good Coverage)

Most estimator classes already implement `if (!this.fitted)` checks:

**Well-Protected Classes:**
- `GLM` (src/stats/estimators/GLM.js) - Lines 78, 831, 908, 931, 956
- `KMeans` (src/ml/estimators/KMeans.js) - Lines 95, 143
- `PCA` (src/mva/estimators/PCA.js) - Lines 93, 121, 131, 156
- `HCA` (src/ml/estimators/HCA.js) - Lines 68, 75, 82
- `RandomForest` (src/ml/estimators/RandomForest.js) - Lines 442, 476, 509, 516, 567
- `PolynomialRegressor` (src/ml/estimators/PolynomialRegressor.js) - Lines 70, 101

### 1.2 Observable-Specific Challenges

**Reactive Cell Execution:**
```javascript
// Observable cell pattern - model instantiation and usage often separated
model = new GLM({ family: 'binomial' });

// In another cell (may run before fit() if dependencies change)
model.summary() // Error: "Model has not been fitted yet"
```

**Memory Constraints:**
- Observable has tighter memory limits than Node.js
- Large design matrices and coefficient storage can cause issues
- Random Forest with many trees stores entire tree structures

---

## 2. Recommended Safeguards & Improvements

### 2.1 Enhanced Fitted State Checks

**Current Implementation:**
```javascript
summary() {
  if (!this.fitted) {
    throw new Error('Model has not been fitted yet. Call fit() first.');
  }
  // ... summary logic
}
```

**Recommended Enhancement:**
```javascript
summary() {
  this._ensureFitted('summary');
  // ... summary logic
}

_ensureFitted(methodName) {
  if (!this.fitted) {
    const className = this.constructor.name;
    throw new Error(
      `${className}.${methodName}() requires a fitted model.\n` +
      `Please call ${className}.fit() first before using ${methodName}().\n` +
      `Tip: In Observable, ensure fit() is called in a cell that executes before ${methodName}().`
    );
  }
}
```

**Benefits:**
- More informative error messages
- Observable-specific guidance
- Centralized error handling
- Consistent messaging across all estimators

### 2.2 Model State Inspection Methods

Add to base `Estimator` class:

```javascript
/**
 * Check if model is fitted
 * @returns {boolean}
 */
isFitted() {
  return !!this.fitted;
}

/**
 * Get model state summary
 * @returns {Object}
 */
getState() {
  return {
    fitted: this.fitted,
    className: this.constructor.name,
    params: this.getParams(),
    memoryEstimate: this._estimateMemoryUsage()
  };
}

/**
 * Estimate memory usage (MB)
 * @returns {number}
 * @private
 */
_estimateMemoryUsage() {
  if (!this.fitted) return 0;

  // Rough estimation based on model internals
  const json = this.toJSON();
  const jsonStr = JSON.stringify(json);
  return (jsonStr.length * 2) / (1024 * 1024); // Rough estimate in MB
}
```

**Usage in Observable:**
```javascript
// Check state before using
if (model.isFitted()) {
  return model.summary();
} else {
  return html`<div class="warning">Model not fitted yet</div>`;
}

// Monitor memory
model.getState() // { fitted: true, memoryEstimate: 2.3 MB, ... }
```

### 2.3 Lazy Computation for Summary Statistics

**Current Issue:**
Summary methods compute everything immediately, which can be expensive for large models.

**Recommendation:**
Implement lazy evaluation for expensive summary statistics:

```javascript
class GLM extends Estimator {
  // ... existing code

  summary(options = {}) {
    this._ensureFitted('summary');

    const alpha = options.alpha !== undefined ? options.alpha : this.params.alpha;
    const verbose = options.verbose !== undefined ? options.verbose : true;

    // Return object with lazy getters
    return {
      // Basic info (cheap)
      family: this._model.family,
      link: this._model.link,
      converged: this._model.converged,
      iterations: this._model.iterations,

      // Lazy expensive computations
      get coefficients() {
        return this._getCoefficientsTable(alpha);
      },

      get goodnessOfFit() {
        return {
          aic: this._model.aic,
          bic: this._model.bic,
          deviance: this._model.deviance,
          pseudoR2: this._model.pseudoR2
        };
      },

      // For backward compatibility, toString returns full summary
      toString() {
        if (verbose) {
          return this._summaryGLM(alpha);
        }
        return `GLM(${this._model.family}, ${this._model.link}, converged=${this._model.converged})`;
      }
    };
  }
}
```

**Benefits:**
- Only compute what's needed
- Better Observable reactivity
- Reduced memory pressure
- Backward compatible via toString()

---

## 3. Memory Optimization Strategies

### 3.1 Model Compression Options

Add optional model compression for fitted models:

```javascript
class GLM extends Estimator {
  constructor(params = {}) {
    super(params);
    this.params = {
      // ... existing params
      compress: false, // New option for Observable
      keepFittedValues: true, // Option to discard fitted values after fit
      ...params
    };
  }

  fit(...args) {
    // ... existing fit logic

    if (this.params.compress) {
      this._compressModel();
    }

    if (!this.params.keepFittedValues) {
      delete this._model.fitted;
      delete this._model.residuals;
    }

    return this;
  }

  _compressModel() {
    // Round coefficients to reduce precision (saves memory in JSON)
    if (this._model.coefficients) {
      this._model.coefficients = this._model.coefficients.map(c =>
        Math.round(c * 1e10) / 1e10
      );
    }

    // Similar for other arrays
    if (this._model.standardErrors) {
      this._model.standardErrors = this._model.standardErrors.map(se =>
        Math.round(se * 1e10) / 1e10
      );
    }
  }
}
```

**Usage:**
```javascript
// For Observable with limited memory
model = new GLM({
  family: 'poisson',
  compress: true,
  keepFittedValues: false // Only keep coefficients
});
```

### 3.2 Incremental Fitting for Large Datasets

For models that support it (e.g., GLM with stochastic gradient descent):

```javascript
class GLM extends Estimator {
  /**
   * Partial fit for mini-batch or online learning
   * Useful for large datasets in Observable
   */
  partialFit(X, y, options = {}) {
    const {
      resetOnFirstCall = true,
      learningRate = 0.01
    } = options;

    if (!this.fitted && resetOnFirstCall) {
      this._initializePartialFit(X, y);
    }

    // Update coefficients incrementally
    this._updateCoefficients(X, y, learningRate);

    this.fitted = true;
    return this;
  }
}
```

### 3.3 RandomForest Memory Optimization

RandomForest with many trees can be very memory-intensive:

```javascript
class RandomForestBase {
  constructor(params) {
    // ... existing params
    this.maxMemoryMB = params.maxMemoryMB || null; // Memory limit
    this.pruneAfterFit = params.pruneAfterFit || false;
  }

  _fitPrepared(X, y, columns, sampleWeight) {
    // ... existing fit logic

    for (let i = startIdx; i < this.nEstimators; i++) {
      // Check memory before adding tree
      if (this.maxMemoryMB) {
        const currentMemory = this._estimateMemoryMB();
        if (currentMemory > this.maxMemoryMB) {
          console.warn(
            `RandomForest: Stopped at ${i} trees due to memory limit ` +
            `(${this.maxMemoryMB}MB). Consider reducing nEstimators or ` +
            `increasing maxMemoryMB.`
          );
          break;
        }
      }

      // ... fit tree
    }

    if (this.pruneAfterFit) {
      this._pruneTreeDetails();
    }
  }

  _pruneTreeDetails() {
    // Remove training data from trees to save memory
    for (const tree of this.trees) {
      this._pruneNode(tree.tree.root);
    }
  }

  _pruneNode(node) {
    if (!node || node.type === 'leaf') return;

    // Remove training indices if stored
    delete node.sampleIndices;

    this._pruneNode(node.left);
    this._pruneNode(node.right);
  }

  _estimateMemoryMB() {
    return (JSON.stringify(this.trees).length * 2) / (1024 * 1024);
  }
}
```

---

## 4. Warning Systems

### 4.1 Performance Warnings

Add warnings for operations that may be slow or memory-intensive:

```javascript
class GLM extends Estimator {
  fit(...args) {
    const { X, y } = this._parseArgs(...args);

    // Warn about large datasets
    const n = X.length;
    const p = X[0].length;

    if (n > 10000 && typeof window !== 'undefined') {
      console.warn(
        `⚠️ GLM: Fitting on ${n} samples may be slow in browser environments.\n` +
        `Consider:\n` +
        `  - Using a sample for interactive development\n` +
        `  - Switching to Node.js for production fitting\n` +
        `  - Using partialFit() for incremental learning`
      );
    }

    if (p > 100 && this.params.family === 'binomial') {
      console.warn(
        `⚠️ GLM: ${p} predictors in logistic regression may cause convergence issues.\n` +
        `Consider:\n` +
        `  - Feature selection or dimensionality reduction\n` +
        `  - Regularization (set regularization parameter)\n` +
        `  - Checking for multicollinearity`
      );
    }

    // ... existing fit logic
  }
}
```

### 4.2 Convergence Warnings

Enhance convergence reporting:

```javascript
class GLM extends Estimator {
  fit(...args) {
    // ... fitting logic

    if (!this._model.converged) {
      const warning =
        `⚠️ GLM did not converge after ${this._model.iterations} iterations.\n` +
        `Possible causes:\n` +
        `  - Ill-conditioned data (check for perfect separation or multicollinearity)\n` +
        `  - maxIter too low (current: ${this.params.maxIter})\n` +
        `  - Tolerance too strict (current: ${this.params.tol})\n` +
        `Recommendations:\n` +
        `  - Increase maxIter or adjust tol\n` +
        `  - Check model.summary() for coefficient estimates\n` +
        `  - Consider regularization`;

      if (this.params.warnOnNoConvergence !== false) {
        console.warn(warning);
      }

      // Store warning for later access
      this._warnings = this._warnings || [];
      this._warnings.push({
        type: 'convergence',
        message: warning,
        iteration: this._model.iterations
      });
    }

    return this;
  }

  getWarnings() {
    return this._warnings || [];
  }
}
```

### 4.3 Observable-Friendly Display

Add HTML display methods for Observable:

```javascript
class GLM extends Estimator {
  /**
   * Observable display integration
   */
  _repr_html_() {
    if (!this.fitted) {
      return `
        <div style="padding: 1em; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;">
          <strong>⚠️ Model Not Fitted</strong>
          <p>This GLM model has not been fitted yet. Call <code>model.fit(X, y)</code> first.</p>
          <p><em>In Observable: Ensure the fit() cell executes before cells that use the model.</em></p>
        </div>
      `;
    }

    const warnings = this.getWarnings();
    let warningHtml = '';

    if (warnings.length > 0) {
      warningHtml = `
        <div style="padding: 0.5em; background: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 1em;">
          <strong>⚠️ ${warnings.length} Warning(s)</strong>
          <ul style="margin: 0.5em 0; padding-left: 1.5em;">
            ${warnings.map(w => `<li>${w.type}: ${w.message.split('\n')[0]}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    return warningHtml + this._summaryGLMHTML();
  }
}
```

---

## 5. Implementation Priority

### High Priority (Immediate Impact)

1. **Enhanced error messages** - Low effort, high impact for Observable users
   - Add `_ensureFitted()` helper to base `Estimator` class
   - Update all model classes to use centralized error handling

2. **Model state inspection** - Essential for debugging in Observable
   - Add `isFitted()` and `getState()` methods
   - Add memory estimation helper

3. **Convergence warnings** - Prevent confusion
   - Add warning system to GLM, GLMM, and multinomial
   - Store warnings for user inspection

### Medium Priority (Significant Value)

4. **Memory optimization options** - Important for Observable
   - Add `compress` and `keepFittedValues` options
   - Implement for GLM, RandomForest, and PCA

5. **Performance warnings** - User education
   - Add dataset size warnings
   - Add complexity warnings

6. **HTML display methods** - Better Observable UX
   - Enhance `_repr_html_()` across all models
   - Add warning display

### Low Priority (Nice to Have)

7. **Lazy summary computation** - Performance optimization
   - Refactor summary methods to use lazy getters
   - Maintain backward compatibility

8. **Incremental fitting** - Advanced feature
   - Add `partialFit()` for GLM with SGD
   - Useful for very large datasets

9. **Advanced memory management** - Complex feature
   - RandomForest pruning
   - Memory limits with automatic tree count adjustment

---

## 6. Backward Compatibility

All recommendations maintain backward compatibility:

- New features are opt-in (e.g., `compress: false` by default)
- Existing error messages enhanced but not changed fundamentally
- New methods added without modifying existing signatures
- HTML methods only called by display systems, don't affect programmatic use

---

## 7. Testing Recommendations

Create Observable-specific test scenarios:

```javascript
describe('Observable compatibility', () => {
  it('should provide helpful error when summary called before fit', () => {
    const model = new GLM({ family: 'gaussian' });
    expect(() => model.summary()).toThrow(/requires a fitted model/);
    expect(() => model.summary()).toThrow(/Observable/); // Mentions Observable
  });

  it('should check fitted state without errors', () => {
    const model = new GLM({ family: 'gaussian' });
    expect(model.isFitted()).toBe(false);
    model.fit(X, y);
    expect(model.isFitted()).toBe(true);
  });

  it('should estimate memory usage', () => {
    const model = new GLM({ family: 'gaussian' });
    model.fit(X, y);
    const state = model.getState();
    expect(state.memoryEstimate).toBeGreaterThan(0);
  });

  it('should respect memory limits in RandomForest', () => {
    const model = new RandomForestClassifier({
      nEstimators: 1000,
      maxMemoryMB: 1 // Very low limit
    });
    model.fit(X, y);
    expect(model.trees.length).toBeLessThan(1000); // Stopped early
  });
});
```

---

## 8. Documentation Updates

Update documentation to include:

1. **Observable-specific guide**
   - Best practices for reactive cells
   - Memory management tips
   - Debugging fitted state issues

2. **Memory optimization guide**
   - When to use compression
   - How to estimate model size
   - Strategies for large datasets

3. **API documentation**
   - Document all new methods
   - Add examples for Observable use cases
   - Include memory estimates for each model type

---

## 9. Example Implementation

See the following files for complete implementation examples:
- `src/core/estimators/estimator.js` - Base class enhancements
- `src/stats/estimators/GLM.js` - GLM-specific optimizations
- `src/ml/estimators/RandomForest.js` - RandomForest memory management

---

## Conclusion

These recommendations address the core issues:
- ✅ Better safeguards for unfitted models
- ✅ Observable-specific error messaging
- ✅ Memory optimization options
- ✅ User-friendly warnings

Implementation can be done incrementally, starting with high-priority items that provide immediate value with minimal code changes.
