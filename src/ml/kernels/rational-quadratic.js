/**
 * Rational Quadratic Kernel
 * 
 * A scale mixture of RBF kernels with different length scales.
 * k(x1, x2) = variance * (1 + ||x1 - x2||² / (2 * alpha * lengthScale²))^(-alpha)
 * 
 * Properties:
 * - alpha controls the mixture (small alpha = more variation in length scales)
 * - As alpha → ∞, approaches RBF kernel
 * - Good for modeling multiple scales of variation
 */

import { Kernel } from './base.js';

export class RationalQuadratic extends Kernel {
  /**
   * @param {number} lengthScale - Length scale (default: 1.0)
   * @param {number} alpha - Scale mixture parameter (default: 1.0)
   * @param {number} variance - Signal variance (default: 1.0)
   */
  constructor(lengthScale = 1.0, alpha = 1.0, variance = 1.0) {
    super();
    this.lengthScale = lengthScale;
    this.alpha = alpha;
    this.variance = variance;
  }

  compute(x1, x2) {
    let squaredDistance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = x1[i] - x2[i];
      squaredDistance += diff * diff;
    }
    const term = 1 + squaredDistance / (2 * this.alpha * this.lengthScale * this.lengthScale);
    return this.variance * Math.pow(term, -this.alpha);
  }

  getParams() {
    return {
      lengthScale: this.lengthScale,
      alpha: this.alpha,
      variance: this.variance
    };
  }

  setParams({ lengthScale, alpha, variance }) {
    if (lengthScale !== undefined) this.lengthScale = lengthScale;
    if (alpha !== undefined) this.alpha = alpha;
    if (variance !== undefined) this.variance = variance;
  }
}
