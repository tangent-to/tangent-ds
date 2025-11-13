/**
 * Statistical hypothesis tests
 */

import { mean, variance } from '../core/math.js';
import { normal } from './distribution.js';

// ============= t-Distribution Helper =============

/**
 * t-distribution CDF (approximation using normal for large df)
 * @param {number} t - t-statistic
 * @param {number} df - degrees of freedom
 * @returns {number} Cumulative probability
 */
function tCDF(t, df) {
  if (df > 30) {
    // For large df, t-distribution approximates normal
    return normal.cdf(t, { mean: 0, sd: 1 });
  }
  
  // Simplified approximation for smaller df
  const x = df / (df + t * t);
  return 1 - 0.5 * incompleteBeta(x, df / 2, 0.5);
}

function incompleteBeta(x, a, b) {
  // Very simple approximation
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x ** a * (1 - x) ** b;
}

// ============= t-tests =============

/**
 * One-sample t-test
 * @param {Array<number>} sample - Sample data
 * @param {Object} options - {mu: hypothesized mean, alternative: 'two-sided'|'less'|'greater'}
 * @returns {Object} {statistic, pValue, df, mean, se}
 */
export function oneSampleTTest(sample, { mu = 0, alternative = 'two-sided' } = {}) {
  const n = sample.length;
  if (n < 2) {
    throw new Error('Sample must have at least 2 observations');
  }
  
  const sampleMean = mean(sample);
  const sampleVar = variance(sample, true);
  const se = Math.sqrt(sampleVar / n);
  const statistic = (sampleMean - mu) / se;
  const df = n - 1;
  
  let pValue;
  if (alternative === 'two-sided') {
    pValue = 2 * (1 - tCDF(Math.abs(statistic), df));
  } else if (alternative === 'less') {
    pValue = tCDF(statistic, df);
  } else {
    pValue = 1 - tCDF(statistic, df);
  }
  
  return {
    statistic,
    pValue,
    df,
    mean: sampleMean,
    se,
    alternative
  };
}

/**
 * Two-sample t-test (assuming equal variances)
 * @param {Array<number>} sample1 - First sample
 * @param {Array<number>} sample2 - Second sample
 * @param {Object} options - {alternative: 'two-sided'|'less'|'greater'}
 * @returns {Object} {statistic, pValue, df, mean1, mean2, pooledSE}
 */
export function twoSampleTTest(sample1, sample2, { alternative = 'two-sided' } = {}) {
  const n1 = sample1.length;
  const n2 = sample2.length;
  
  if (n1 < 2 || n2 < 2) {
    throw new Error('Both samples must have at least 2 observations');
  }
  
  const mean1 = mean(sample1);
  const mean2 = mean(sample2);
  const var1 = variance(sample1, true);
  const var2 = variance(sample2, true);
  
  // Pooled variance
  const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
  const se = Math.sqrt(pooledVar * (1 / n1 + 1 / n2));
  
  const statistic = (mean1 - mean2) / se;
  const df = n1 + n2 - 2;
  
  let pValue;
  if (alternative === 'two-sided') {
    pValue = 2 * (1 - tCDF(Math.abs(statistic), df));
  } else if (alternative === 'less') {
    pValue = tCDF(statistic, df);
  } else {
    pValue = 1 - tCDF(statistic, df);
  }
  
  return {
    statistic,
    pValue,
    df,
    mean1,
    mean2,
    pooledSE: se,
    alternative
  };
}

// ============= Chi-square test =============

/**
 * Chi-square goodness of fit test
 * @param {Array<number>} observed - Observed frequencies
 * @param {Array<number>} expected - Expected frequencies
 * @returns {Object} {statistic, pValue, df}
 */
export function chiSquareTest(observed, expected) {
  if (observed.length !== expected.length) {
    throw new Error('Observed and expected must have same length');
  }
  
  if (observed.length < 2) {
    throw new Error('Need at least 2 categories');
  }
  
  let statistic = 0;
  for (let i = 0; i < observed.length; i++) {
    if (expected[i] <= 0) {
      throw new Error('Expected frequencies must be positive');
    }
    statistic += ((observed[i] - expected[i]) ** 2) / expected[i];
  }
  
  const df = observed.length - 1;
  const pValue = 1 - chiSquareCDF(statistic, df);
  
  return {
    statistic,
    pValue,
    df
  };
}

/**
 * Chi-square CDF approximation
 */
function chiSquareCDF(x, k) {
  if (x <= 0) return 0;
  
  // For large k, use normal approximation
  if (k > 30) {
    const z = (Math.sqrt(2 * x) - Math.sqrt(2 * k - 1));
    return normal.cdf(z, { mean: 0, sd: 1 });
  }
  
  // Simple gamma approximation: chi-square(k) = gamma(k/2, 2)
  return incompleteBeta(x / (x + k), k / 2, k / 2);
}

// ============= ANOVA =============

/**
 * One-way ANOVA
 * @param {Array<Array<number>>} groups - Array of group samples
 * @returns {Object} {statistic, pValue, dfBetween, dfWithin, MSbetween, MSwithin}
 */
export function oneWayAnova(groups) {
  if (groups.length < 2) {
    throw new Error('Need at least 2 groups');
  }
  
  const k = groups.length;
  const groupMeans = groups.map(g => mean(g));
  const groupSizes = groups.map(g => g.length);
  const n = groupSizes.reduce((a, b) => a + b, 0);
  
  // Grand mean
  const allData = groups.flat();
  const grandMean = mean(allData);
  
  // Between-group sum of squares
  let SSbetween = 0;
  for (let i = 0; i < k; i++) {
    SSbetween += groupSizes[i] * (groupMeans[i] - grandMean) ** 2;
  }
  
  // Within-group sum of squares
  let SSwithin = 0;
  for (let i = 0; i < k; i++) {
    for (const val of groups[i]) {
      SSwithin += (val - groupMeans[i]) ** 2;
    }
  }
  
  const dfBetween = k - 1;
  const dfWithin = n - k;
  
  const MSbetween = SSbetween / dfBetween;
  const MSwithin = SSwithin / dfWithin;
  
  const statistic = MSbetween / MSwithin;
  
  // F-test p-value (approximation)
  const pValue = 1 - fCDF(statistic, dfBetween, dfWithin);
  
  return {
    statistic,
    pValue,
    dfBetween,
    dfWithin,
    MSbetween,
    MSwithin
  };
}

/**
 * F-distribution CDF (simplified approximation)
 */
function fCDF(f, d1, d2) {
  if (f <= 0) return 0;

  // Very rough approximation using beta distribution relationship
  const x = d2 / (d2 + d1 * f);
  return 1 - incompleteBeta(x, d2 / 2, d1 / 2);
}

// ============= Tukey HSD (Post-hoc test) =============

/**
 * Studentized range distribution CDF approximation
 * @param {number} q - Studentized range statistic
 * @param {number} k - Number of groups
 * @param {number} df - Degrees of freedom
 * @returns {number} Cumulative probability
 */
function qrangeCDF(q, k, df) {
  if (q <= 0) return 0;

  // Approximation using normal distribution for large df
  // For simplicity, we use a conservative approximation
  // More accurate implementations would use specialized algorithms

  // Transform q to approximate normal
  const z = (q - Math.sqrt(2 * Math.log(k))) / Math.sqrt(1 / df);
  const p = normal.cdf(z, { mean: 0, sd: 1 });

  // Adjust for number of comparisons
  return Math.pow(p, k - 1);
}

/**
 * Tukey's Honestly Significant Difference (HSD) test
 * Post-hoc test for pairwise comparisons after ANOVA
 *
 * @param {Array<Array<number>>} groups - Array of group samples
 * @param {Object} options - {alpha: significance level (default 0.05), anovaResult: optional precomputed ANOVA result}
 * @returns {Object} {
 *   comparisons: Array of {groups: [i,j], diff, lowerCI, upperCI, pValue, significant},
 *   groupMeans: Array of group means,
 *   groupNames: Array of group indices,
 *   MSwithin: Mean square within groups,
 *   dfWithin: Degrees of freedom,
 *   alpha: Significance level
 * }
 */
export function tukeyHSD(groups, { alpha = 0.05, anovaResult = null } = {}) {
  if (groups.length < 2) {
    throw new Error('Need at least 2 groups for Tukey HSD test');
  }

  const k = groups.length;
  const groupSizes = groups.map(g => g.length);
  const n = groupSizes.reduce((a, b) => a + b, 0);

  // Compute or reuse ANOVA results
  let MSwithin, dfWithin;
  if (anovaResult) {
    MSwithin = anovaResult.MSwithin;
    dfWithin = anovaResult.dfWithin;
  } else {
    const anovaRes = oneWayAnova(groups);
    MSwithin = anovaRes.MSwithin;
    dfWithin = anovaRes.dfWithin;
  }

  // Compute group means
  const groupMeans = groups.map(g => mean(g));

  // Perform all pairwise comparisons
  const comparisons = [];

  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const ni = groupSizes[i];
      const nj = groupSizes[j];
      const meanDiff = groupMeans[i] - groupMeans[j];

      // Standard error for the difference
      const se = Math.sqrt(MSwithin * (1 / ni + 1 / nj));

      // Studentized range statistic
      const q = Math.abs(meanDiff) / (Math.sqrt(MSwithin / 2 * (1 / ni + 1 / nj)));

      // p-value (approximation)
      const pValue = 1 - qrangeCDF(q, k, dfWithin);

      // Critical value approximation
      // For accurate results, this should use studentized range tables
      // Here we use a conservative approximation based on t-distribution
      const tCrit = approximateQCritical(alpha, k, dfWithin);
      const margin = tCrit * se;

      comparisons.push({
        groups: [i, j],
        groupLabels: [`Group ${i}`, `Group ${j}`],
        diff: meanDiff,
        lowerCI: meanDiff - margin,
        upperCI: meanDiff + margin,
        pValue: Math.max(0, Math.min(1, pValue)),
        significant: pValue < alpha,
        qStatistic: q
      });
    }
  }

  return {
    comparisons,
    groupMeans,
    groupNames: groups.map((_, i) => `Group ${i}`),
    MSwithin,
    dfWithin,
    alpha,
    numGroups: k
  };
}

/**
 * Approximate critical value for studentized range distribution
 * This is a simplified approximation; production code should use tables or more accurate methods
 */
function approximateQCritical(alpha, k, df) {
  // Conservative approximation using t-distribution with Bonferroni-like adjustment
  // Tukey's q is approximately sqrt(2) * t for large df
  const numComparisons = k * (k - 1) / 2;
  const adjustedAlpha = alpha / numComparisons;

  // Approximate t critical value
  // For two-tailed test at adjusted alpha
  const z = Math.abs(normal.quantile(adjustedAlpha / 2, { mean: 0, sd: 1 }));

  // Adjust for df using t-distribution approximation
  const tCrit = z * Math.sqrt((df + 1) / df);

  // Scale for studentized range (approximately sqrt(2) times t for pairwise)
  return tCrit * Math.sqrt(2);
}
