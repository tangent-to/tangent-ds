/**
 * Gaussian Process Regressor
 * 
 * A scikit-learn style Gaussian Process implementation.
 * Supports:
 * - Fitting to training data
 * - Prediction with uncertainty quantification
 * - Sampling from posterior distribution
 * 
 * @example
 * const gp = new GaussianProcessRegressor({ kernel: 'rbf', lengthScale: 1.0 });
 * gp.fit(X_train, y_train);
 * const { mean, std } = gp.predict(X_test, { returnStd: true });
 * const samples = gp.sample(X_test, 5);
 */

import { Regressor } from '../../core/estimators/estimator.js';
import { toMatrix } from '../../core/linalg.js';
import { prepareXY, prepareX } from '../../core/table.js';
import { Kernel, RBF, Periodic, RationalQuadratic } from '../kernels/index.js';

/**
 * Seeded random number generator (Mulberry32)
 * @param {number} seed - Seed value
 * @returns {Function} Random number generator function
 */
function mulberry32(seed) {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/**
 * Cholesky decomposition (lower triangular)
 * @param {Matrix} A - Symmetric positive definite matrix
 * @returns {Matrix} Lower triangular matrix L such that A = L * L^T
 */
function choleskyDecomposition(A) {
  const n = A.rows;
  const L = new Array(n);
  
  for (let i = 0; i < n; i++) {
    L[i] = new Array(n).fill(0);
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      
      if (j === i) {
        for (let k = 0; k < j; k++) {
          sum += L[j][k] * L[j][k];
        }
        const diag = A.get(j, j) - sum;
        if (diag <= 0) {
          throw new Error('Matrix is not positive definite');
        }
        L[j][j] = Math.sqrt(diag);
      } else {
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        L[i][j] = (A.get(i, j) - sum) / L[j][j];
      }
    }
  }

  return toMatrix(L);
}

/**
 * Sample from multivariate normal distribution
 * @param {Array<number>} mean - Mean vector
 * @param {Matrix} cov - Covariance matrix
 * @param {Function} rng - Random number generator
 * @returns {Array<number>} Sample
 */
function sampleMultivariateNormal(mean, cov, rng = Math.random) {
  const n = mean.length;
  let L;
  
  try {
    L = choleskyDecomposition(cov);
  } catch (e) {
    // Add jitter if not positive definite
    const jitter = 1e-6;
    for (let i = 0; i < n; i++) {
      cov.set(i, i, cov.get(i, i) + jitter);
    }
    L = choleskyDecomposition(cov);
  }

  // Generate standard normal samples using Box-Muller
  const z = new Array(n);
  for (let i = 0; i < n; i++) {
    const u1 = rng();
    const u2 = rng();
    z[i] = Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  // Transform: sample = mean + L * z
  const sample = new Array(n);
  for (let i = 0; i < n; i++) {
    sample[i] = mean[i];
    for (let j = 0; j <= i; j++) {
      sample[i] += L.get(i, j) * z[j];
    }
  }

  return sample;
}

export class GaussianProcessRegressor extends Regressor {
  /**
   * @param {Object} opts - Options
   * @param {Kernel|string} opts.kernel - Kernel instance or type ('rbf', 'periodic', 'rational_quadratic')
   * @param {number} opts.lengthScale - Length scale for kernel (default: 1.0)
   * @param {number} opts.variance - Signal variance (default: 1.0)
   * @param {number} opts.alpha - Noise level / regularization (default: 1e-10)
   * @param {number} opts.noiseLevel - Alias for alpha
   * @param {number} opts.period - Period for periodic kernel
   */
  constructor(opts = {}) {
    super(opts);
    
    // Create kernel
    if (opts.kernel instanceof Kernel) {
      this.kernel = opts.kernel;
    } else {
      const kernelType = (opts.kernel || 'rbf').toLowerCase();
      const lengthScale = opts.lengthScale || 1.0;
      const variance = opts.variance || 1.0;
      
      switch (kernelType) {
        case 'rbf':
          this.kernel = new RBF(lengthScale, variance);
          break;
        case 'periodic':
          this.kernel = new Periodic(lengthScale, opts.period || 1.0, variance);
          break;
        case 'rational_quadratic':
        case 'rationalquadratic':
          this.kernel = new RationalQuadratic(lengthScale, opts.alpha || 1.0, variance);
          break;
        default:
          throw new Error(`Unknown kernel type: ${kernelType}`);
      }
    }
    
    // Support both alpha and noiseLevel
    this.alpha = opts.alpha ?? opts.noiseLevel ?? 1e-10;
    
    // Internal state
    this._XTrain = null;
    this._yTrain = null;
    this._L = null;
    this._alphaVector = null;
  }

  /**
   * Fit the GP to training data
   * @param {Array|Object} X - Training inputs (n x d) or { X, y, data }
   * @param {Array} [y] - Training targets (n)
   * @returns {this}
   */
  fit(X, y = null) {
    // Handle declarative input
    let dataX, dataY;
    if (X && typeof X === 'object' && !Array.isArray(X) && (X.data || X.X)) {
      const prepared = prepareXY({
        X: X.X || X.columns,
        y: X.y,
        data: X.data,
        omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
      });
      dataX = prepared.X;
      dataY = prepared.y;
    } else {
      dataX = X;
      dataY = y;
    }

    // Convert to matrix
    this._XTrain = toMatrix(dataX);
    this._yTrain = Array.isArray(dataY) ? [...dataY] : Array.from(dataY);

    // Compute kernel matrix
    const K = this.kernel.call(this._XTrain);
    
    // Add noise to diagonal
    for (let i = 0; i < K.rows; i++) {
      K.set(i, i, K.get(i, i) + this.alpha);
    }

    // Cholesky decomposition
    try {
      this._L = choleskyDecomposition(K);
    } catch (error) {
      throw new Error(`Failed to fit GP: ${error.message}. Try increasing alpha.`);
    }

    // Solve for alpha: L * L^T * alpha = y
    this._alphaVector = this._solveCholesky(this._L, this._yTrain);

    this.fitted = true;
    return this;
  }

  /**
   * Predict at test points
   * @param {Array} X - Test inputs (m x d)
   * @param {Object} opts - Options
   * @param {boolean} opts.returnStd - Return standard deviations
   * @returns {Array|Object} Predictions, or { mean, std } if returnStd=true
   */
  predict(X, opts = {}) {
    this._ensureFitted('predict');
    const { returnStd = false } = opts;

    const XTest = toMatrix(X);
    const KStar = this.kernel.call(this._XTrain, XTest);

    // Compute mean: K* @ alpha
    const mean = new Array(XTest.rows);
    for (let i = 0; i < XTest.rows; i++) {
      mean[i] = 0;
      for (let j = 0; j < this._XTrain.rows; j++) {
        mean[i] += KStar.get(j, i) * this._alphaVector[j];
      }
    }

    if (!returnStd) {
      return mean;
    }

    // Compute std
    const std = this._computeStd(XTest, KStar);
    return { mean, std };
  }

  /**
   * Sample from the posterior distribution
   * @param {Array} X - Test inputs
   * @param {number} nSamples - Number of samples
   * @param {number} [seed] - Random seed for reproducibility
   * @returns {Array<Array>} Array of samples
   */
  sample(X, nSamples = 1, seed = null) {
    this._ensureFitted('sample');

    const rng = seed !== null ? mulberry32(seed) : Math.random;
    const XTest = toMatrix(X);
    const { mean, std } = this.predict(X, { returnStd: true });

    const samples = [];
    for (let s = 0; s < nSamples; s++) {
      const sample = new Array(XTest.rows);
      for (let i = 0; i < XTest.rows; i++) {
        // Sample from N(mean[i], std[i]²)
        const u1 = rng();
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
        sample[i] = mean[i] + std[i] * z;
      }
      samples.push(sample);
    }

    return samples;
  }

  /**
   * Sample from the prior (unfitted GP)
   * @param {Array} X - Input points
   * @param {number} nSamples - Number of samples
   * @param {number} [seed] - Random seed for reproducibility
   * @returns {Array<Array>} Array of samples
   */
  samplePrior(X, nSamples = 1, seed = null) {
    const rng = seed !== null ? mulberry32(seed) : Math.random;
    const XMatrix = toMatrix(X);
    const n = XMatrix.rows;
    
    // Compute kernel matrix
    const K = this.kernel.call(XMatrix);
    
    // Add small noise for numerical stability
    for (let i = 0; i < n; i++) {
      K.set(i, i, K.get(i, i) + 1e-10);
    }

    const mean = new Array(n).fill(0);
    
    const samples = [];
    for (let s = 0; s < nSamples; s++) {
      samples.push(sampleMultivariateNormal(mean, K, rng));
    }

    return samples;
  }

  _computeStd(XTest, KStar) {
    const std = new Array(XTest.rows);

    for (let i = 0; i < XTest.rows; i++) {
      // K** diagonal element
      const kStarStar = this.kernel.compute(XTest.getRow(i), XTest.getRow(i));

      // Solve L @ v = k*
      const kStarColumn = new Array(this._XTrain.rows);
      for (let j = 0; j < this._XTrain.rows; j++) {
        kStarColumn[j] = KStar.get(j, i);
      }
      const v = this._forwardSubstitution(this._L, kStarColumn);

      // Variance = k** - v^T @ v
      let vTv = 0;
      for (let j = 0; j < v.length; j++) {
        vTv += v[j] * v[j];
      }

      const variance = kStarStar - vTv;
      std[i] = Math.sqrt(Math.max(0, variance));
    }

    return std;
  }

  _solveCholesky(L, y) {
    const z = this._forwardSubstitution(L, y);
    return this._backSubstitution(L, z);
  }

  _forwardSubstitution(L, b) {
    const n = L.rows;
    const x = new Array(n);

    for (let i = 0; i < n; i++) {
      x[i] = b[i];
      for (let j = 0; j < i; j++) {
        x[i] -= L.get(i, j) * x[j];
      }
      x[i] /= L.get(i, i);
    }

    return x;
  }

  _backSubstitution(L, b) {
    const n = L.rows;
    const x = new Array(n);

    for (let i = n - 1; i >= 0; i--) {
      x[i] = b[i];
      for (let j = i + 1; j < n; j++) {
        x[i] -= L.get(j, i) * x[j];
      }
      x[i] /= L.get(i, i);
    }

    return x;
  }

  toJSON() {
    return {
      type: 'GaussianProcessRegressor',
      kernel: {
        type: this.kernel.constructor.name,
        params: this.kernel.getParams()
      },
      alpha: this.alpha,
      fitted: this.fitted,
      XTrain: this._XTrain ? this._XTrain.to2DArray() : null,
      yTrain: this._yTrain,
      L: this._L ? this._L.to2DArray() : null,
      alphaVector: this._alphaVector
    };
  }

  static fromJSON(json) {
    let kernel;
    switch (json.kernel.type) {
      case 'RBF':
        kernel = new RBF(json.kernel.params.lengthScale, json.kernel.params.variance);
        break;
      case 'Periodic':
        kernel = new Periodic(json.kernel.params.lengthScale, json.kernel.params.period, json.kernel.params.variance);
        break;
      case 'RationalQuadratic':
        kernel = new RationalQuadratic(json.kernel.params.lengthScale, json.kernel.params.alpha, json.kernel.params.variance);
        break;
      default:
        throw new Error(`Unknown kernel type: ${json.kernel.type}`);
    }

    const gp = new GaussianProcessRegressor({ kernel, alpha: json.alpha });
    
    if (json.fitted) {
      gp._XTrain = toMatrix(json.XTrain);
      gp._yTrain = json.yTrain;
      gp._L = toMatrix(json.L);
      gp._alphaVector = json.alphaVector;
      gp.fitted = true;
    }

    return gp;
  }
}
