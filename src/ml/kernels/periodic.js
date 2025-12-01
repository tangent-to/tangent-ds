/**
 * Periodic Kernel
 * 
 * For modeling periodic/seasonal patterns.
 * k(x1, x2) = variance * exp(-2 * sin²(π * |x1 - x2| / period) / lengthScale²)
 * 
 * Properties:
 * - Captures repeating patterns
 * - period controls the distance between repetitions
 * - lengthScale controls smoothness within each period
 */

import { Kernel } from './base.js';

export class Periodic extends Kernel {
  /**
   * @param {number} lengthScale - Length scale (default: 1.0)
   * @param {number} period - Period length (default: 1.0)
   * @param {number} variance - Signal variance (default: 1.0)
   */
  constructor(lengthScale = 1.0, period = 1.0, variance = 1.0) {
    super();
    this.lengthScale = lengthScale;
    this.period = period;
    this.variance = variance;
  }

  compute(x1, x2) {
    // For 1D, compute distance; for multi-D, use Euclidean
    let distance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = x1[i] - x2[i];
      distance += diff * diff;
    }
    distance = Math.sqrt(distance);

    const sinTerm = Math.sin(Math.PI * distance / this.period);
    return this.variance * Math.exp(-2 * sinTerm * sinTerm / (this.lengthScale * this.lengthScale));
  }

  getParams() {
    return {
      lengthScale: this.lengthScale,
      period: this.period,
      variance: this.variance
    };
  }

  setParams({ lengthScale, period, variance }) {
    if (lengthScale !== undefined) this.lengthScale = lengthScale;
    if (period !== undefined) this.period = period;
    if (variance !== undefined) this.variance = variance;
  }
}
