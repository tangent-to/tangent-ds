/**
 * Statistical hypothesis tests, effect sizes, and multiple testing corrections
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

  // Handle zero variance case
  if (sampleVar === 0) {
    // If variance is 0, all values are identical
    // If mean equals mu, cannot reject null (p = 1)
    // If mean differs from mu, strongly reject null (p ≈ 0)
    const diff = sampleMean - mu;
    if (Math.abs(diff) < 1e-10) {
      return {
        statistic: 0,
        pValue: 1.0,
        df: n - 1,
        mean: sampleMean,
        se: 0,
        alternative
      };
    } else {
      // All values are identical but different from mu
      // This is strong evidence against H0
      return {
        statistic: diff > 0 ? Infinity : -Infinity,
        pValue: 0.0,
        df: n - 1,
        mean: sampleMean,
        se: 0,
        alternative
      };
    }
  }

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

  // Special cases for small df (exact formulas)
  if (k === 1) {
    // df=1: CDF = erf(sqrt(x/2))
    const z = Math.sqrt(x);
    return 2 * normal.cdf(z, { mean: 0, sd: 1 }) - 1;
  }

  if (k === 2) {
    // df=2: CDF = 1 - exp(-x/2)
    return 1 - Math.exp(-x / 2);
  }

  // For large k, use Wilson-Hilferty normal approximation
  if (k > 30) {
    const z = (Math.sqrt(2 * x) - Math.sqrt(2 * k - 1));
    return normal.cdf(z, { mean: 0, sd: 1 });
  }

  // For intermediate df, use incomplete gamma approximation
  // This is still an approximation but better than incomplete beta
  const a = k / 2;
  const z = x / 2;

  // Series expansion for lower incomplete gamma
  let sum = 1;
  let term = 1;
  for (let i = 1; i < 100; i++) {
    term *= z / (a + i - 1);
    sum += term;
    if (Math.abs(term) < 1e-10) break;
  }

  return 1 - Math.exp(-z) * Math.pow(z, a) * sum / gamma_func(a);
}

/**
 * Gamma function approximation using Lanczos approximation
 */
function gamma_func(z) {
  if (z < 0.5) {
    // Reflection formula for z < 0.5
    return Math.PI / (Math.sin(Math.PI * z) * gamma_func(1 - z));
  }

  // Lanczos coefficients for g=7
  const g = 7;
  const coef = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  z -= 1;
  let x = coef[0];
  for (let i = 1; i < g + 2; i++) {
    x += coef[i] / (z + i);
  }

  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
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

// ============= Paired t-test =============

/**
 * Paired t-test for dependent samples
 * @param {Array<number>} sample1 - First sample (before)
 * @param {Array<number>} sample2 - Second sample (after)
 * @param {Object} options - {mu: hypothesized mean difference (default 0), alternative: 'two-sided'|'less'|'greater'}
 * @returns {Object} {statistic, pValue, df, meanDiff, se}
 */
export function pairedTTest(sample1, sample2, { mu = 0, alternative = 'two-sided' } = {}) {
  if (sample1.length !== sample2.length) {
    throw new Error('Paired samples must have equal length');
  }

  const n = sample1.length;
  if (n < 2) {
    throw new Error('Paired samples must have at least 2 observations');
  }

  // Compute differences
  const differences = sample1.map((val, i) => val - sample2[i]);

  // Perform one-sample t-test on differences
  return oneSampleTTest(differences, { mu, alternative });
}

// ============= Non-parametric tests =============

/**
 * Mann-Whitney U test (Wilcoxon rank-sum test)
 * Non-parametric alternative to two-sample t-test
 * @param {Array<number>} sample1 - First sample
 * @param {Array<number>} sample2 - Second sample
 * @param {Object} options - {alternative: 'two-sided'|'less'|'greater'}
 * @returns {Object} {statistic (U), pValue, alternative}
 */
export function mannWhitneyU(sample1, sample2, { alternative = 'two-sided' } = {}) {
  const n1 = sample1.length;
  const n2 = sample2.length;

  if (n1 < 1 || n2 < 1) {
    throw new Error('Both samples must have at least 1 observation');
  }

  // Combine samples with group labels
  const combined = [
    ...sample1.map(val => ({ value: val, group: 1 })),
    ...sample2.map(val => ({ value: val, group: 2 }))
  ];

  // Sort by value
  combined.sort((a, b) => a.value - b.value);

  // Assign ranks (handle ties by averaging)
  const ranks = [];
  let i = 0;
  while (i < combined.length) {
    let j = i;
    // Find all tied values
    while (j < combined.length && combined[j].value === combined[i].value) {
      j++;
    }
    // Average rank for ties
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks.push(avgRank);
    }
    i = j;
  }

  // Sum ranks for group 1
  let R1 = 0;
  for (let k = 0; k < combined.length; k++) {
    if (combined[k].group === 1) {
      R1 += ranks[k];
    }
  }

  // Calculate U statistic
  const U1 = R1 - (n1 * (n1 + 1)) / 2;
  const U2 = n1 * n2 - U1;
  const U = Math.min(U1, U2);

  // Normal approximation for p-value
  const meanU = (n1 * n2) / 2;
  const stdU = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);

  // Use U1 for directional tests
  const z = (U1 - meanU) / stdU;

  let pValue;
  if (alternative === 'two-sided') {
    pValue = 2 * (1 - normal.cdf(Math.abs(z), { mean: 0, sd: 1 }));
  } else if (alternative === 'less') {
    pValue = normal.cdf(z, { mean: 0, sd: 1 });
  } else {
    pValue = 1 - normal.cdf(z, { mean: 0, sd: 1 });
  }

  return {
    statistic: U,
    pValue,
    alternative,
    U1,
    U2
  };
}

/**
 * Kruskal-Wallis H test
 * Non-parametric alternative to one-way ANOVA
 * @param {Array<Array<number>>} groups - Array of group samples
 * @returns {Object} {statistic (H), pValue, df}
 */
export function kruskalWallis(groups) {
  if (groups.length < 2) {
    throw new Error('Need at least 2 groups');
  }

  const k = groups.length;
  const groupSizes = groups.map(g => g.length);
  const n = groupSizes.reduce((a, b) => a + b, 0);

  // Combine all samples with group labels
  const combined = [];
  for (let i = 0; i < k; i++) {
    for (const val of groups[i]) {
      combined.push({ value: val, group: i });
    }
  }

  // Sort by value
  combined.sort((a, b) => a.value - b.value);

  // Assign ranks (handle ties by averaging)
  const ranks = new Array(combined.length);
  let i = 0;
  while (i < combined.length) {
    let j = i;
    // Find all tied values
    while (j < combined.length && combined[j].value === combined[i].value) {
      j++;
    }
    // Average rank for ties
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks[k] = avgRank;
    }
    i = j;
  }

  // Calculate sum of ranks for each group
  const rankSums = new Array(k).fill(0);
  for (let i = 0; i < combined.length; i++) {
    rankSums[combined[i].group] += ranks[i];
  }

  // Calculate H statistic
  let H = 0;
  for (let i = 0; i < k; i++) {
    H += (rankSums[i] ** 2) / groupSizes[i];
  }
  H = (12 / (n * (n + 1))) * H - 3 * (n + 1);

  // Degrees of freedom
  const df = k - 1;

  // p-value using chi-square approximation
  const pValue = 1 - chiSquareCDF(H, df);

  return {
    statistic: H,
    pValue,
    df
  };
}

// ============= Effect Sizes =============

/**
 * Cohen's d effect size for two samples
 * Standardized mean difference
 * @param {Array<number>} sample1 - First sample
 * @param {Array<number>} sample2 - Second sample
 * @param {Object} options - {pooled: use pooled SD (default true)}
 * @returns {number} Cohen's d
 */
export function cohensD(sample1, sample2, { pooled = true } = {}) {
  const mean1 = mean(sample1);
  const mean2 = mean(sample2);
  const n1 = sample1.length;
  const n2 = sample2.length;

  let sd;
  if (pooled) {
    // Pooled standard deviation
    const var1 = variance(sample1, true);
    const var2 = variance(sample2, true);
    sd = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
  } else {
    // Use SD of first sample (or average)
    sd = Math.sqrt(variance(sample1, true));
  }

  return (mean1 - mean2) / sd;
}

/**
 * Eta-squared effect size for ANOVA
 * Proportion of total variance explained by group differences
 * @param {Object} anovaResult - Result from oneWayAnova
 * @returns {number} Eta-squared
 */
export function etaSquared(anovaResult) {
  const { dfBetween, dfWithin, MSbetween, MSwithin } = anovaResult;
  const SSbetween = MSbetween * dfBetween;
  const SSwithin = MSwithin * dfWithin;
  const SStotal = SSbetween + SSwithin;

  // Handle case where there's no variance
  if (SStotal === 0) return 0;

  return SSbetween / SStotal;
}

/**
 * Omega-squared effect size for ANOVA
 * Less biased estimate than eta-squared
 * @param {Object} anovaResult - Result from oneWayAnova
 * @returns {number} Omega-squared
 */
export function omegaSquared(anovaResult) {
  const { dfBetween, dfWithin, MSbetween, MSwithin } = anovaResult;
  const SSbetween = MSbetween * dfBetween;
  const SSwithin = MSwithin * dfWithin;
  const SStotal = SSbetween + SSwithin;
  const n = dfBetween + dfWithin + 1;

  return (SSbetween - dfBetween * MSwithin) / (SStotal + MSwithin);
}

// ============= Assumption Testing =============

/**
 * Levene's test for equality of variances
 * Tests homogeneity of variance assumption (homoscedasticity)
 * @param {Array<Array<number>>} groups - Array of group samples
 * @param {Object} options - {center: 'mean'|'median'|'trimmed', trim: trim proportion for trimmed mean (default 0.1)}
 * @returns {Object} {statistic, pValue, df1, df2}
 */
export function leveneTest(groups, { center = 'median', trim = 0.1 } = {}) {
  if (groups.length < 2) {
    throw new Error('Need at least 2 groups');
  }

  const k = groups.length;
  const groupSizes = groups.map(g => g.length);
  const n = groupSizes.reduce((a, b) => a + b, 0);

  // Compute center for each group
  const centers = groups.map(g => {
    if (center === 'mean') {
      return mean(g);
    } else if (center === 'median') {
      const sorted = [...g].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
    } else if (center === 'trimmed') {
      const sorted = [...g].sort((a, b) => a - b);
      const trimCount = Math.floor(sorted.length * trim);
      const trimmed = sorted.slice(trimCount, sorted.length - trimCount);
      return mean(trimmed);
    }
    throw new Error('Invalid center type');
  });

  // Compute absolute deviations from center
  const deviations = groups.map((g, i) =>
    g.map(val => Math.abs(val - centers[i]))
  );

  // Perform one-way ANOVA on absolute deviations
  const anovaResult = oneWayAnova(deviations);

  return {
    statistic: anovaResult.statistic,
    pValue: anovaResult.pValue,
    df1: anovaResult.dfBetween,
    df2: anovaResult.dfWithin
  };
}


// ============= Correlation Tests =============

/**
 * Pearson correlation coefficient with significance test
 * @param {Array<number>} x - First variable
 * @param {Array<number>} y - Second variable
 * @returns {Object} {r, pValue, df, ci95}
 */
export function pearsonCorrelation(x, y) {
  if (x.length !== y.length) {
    throw new Error('x and y must have equal length');
  }

  const n = x.length;
  if (n < 3) {
    throw new Error('Need at least 3 observations');
  }

  const meanX = mean(x);
  const meanY = mean(y);

  let sumXY = 0;
  let sumX2 = 0;
  let sumY2 = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    sumXY += dx * dy;
    sumX2 += dx * dx;
    sumY2 += dy * dy;
  }

  // Handle edge case: zero variance in either variable
  if (sumX2 === 0 || sumY2 === 0) {
    // When one variable is constant, correlation is undefined but conventionally 0
    return {
      r: 0,
      pValue: 1.0,  // No evidence of correlation
      df: n - 2,
      tStatistic: 0,
      ci95: [0, 0]
    };
  }

  const r = sumXY / Math.sqrt(sumX2 * sumY2);

  // Test statistic
  const df = n - 2;
  let t, pValue;

  // Handle perfect correlation (r = ±1)
  if (Math.abs(r) >= 1 - 1e-10) {
    t = r > 0 ? Infinity : -Infinity;
    pValue = 0.0;
  } else {
    t = r * Math.sqrt(df / (1 - r * r));
    pValue = 2 * (1 - tCDF(Math.abs(t), df));
  }

  // Fisher's z transformation for confidence interval
  let rLower, rUpper;

  if (Math.abs(r) >= 1 - 1e-10) {
    // Perfect correlation: CI is just the point estimate
    rLower = r > 0 ? 1.0 : -1.0;
    rUpper = r > 0 ? 1.0 : -1.0;
  } else {
    const z = 0.5 * Math.log((1 + r) / (1 - r));
    const seZ = 1 / Math.sqrt(n - 3);
    const zCrit = Math.abs(normal.quantile(0.025, { mean: 0, sd: 1 }));
    const zLower = z - zCrit * seZ;
    const zUpper = z + zCrit * seZ;

    // Back-transform to correlation scale
    rLower = (Math.exp(2 * zLower) - 1) / (Math.exp(2 * zLower) + 1);
    rUpper = (Math.exp(2 * zUpper) - 1) / (Math.exp(2 * zUpper) + 1);
  }

  return {
    r,
    pValue,
    df,
    tStatistic: t,
    ci95: [rLower, rUpper]
  };
}

/**
 * Spearman rank correlation coefficient with significance test
 * @param {Array<number>} x - First variable
 * @param {Array<number>} y - Second variable
 * @returns {Object} {rho, pValue, df}
 */
export function spearmanCorrelation(x, y) {
  if (x.length !== y.length) {
    throw new Error('x and y must have equal length');
  }

  const n = x.length;
  if (n < 3) {
    throw new Error('Need at least 3 observations');
  }

  // Rank both variables
  const rankX = rank(x);
  const rankY = rank(y);

  // Compute Pearson correlation on ranks
  const result = pearsonCorrelation(rankX, rankY);

  return {
    rho: result.r,
    pValue: result.pValue,
    df: result.df,
    tStatistic: result.tStatistic
  };
}

/**
 * Assign ranks to data (handling ties with average rank)
 */
function rank(data) {
  const indexed = data.map((val, i) => ({ val, index: i }));
  indexed.sort((a, b) => a.val - b.val);

  const ranks = new Array(data.length);
  let i = 0;

  while (i < data.length) {
    let j = i;
    // Find ties
    while (j < data.length && indexed[j].val === indexed[i].val) {
      j++;
    }

    // Average rank for ties
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) {
      ranks[indexed[k].index] = avgRank;
    }

    i = j;
  }

  return ranks;
}

/**
 * Fisher's exact test for 2x2 contingency tables
 * @param {Array<Array<number>>} table - 2x2 contingency table [[a,b],[c,d]]
 * @param {Object} options - {alternative: 'two-sided'|'less'|'greater'}
 * @returns {Object} {pValue, oddsRatio, alternative}
 */
export function fisherExactTest(table, { alternative = 'two-sided' } = {}) {
  if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
    throw new Error('Fisher exact test requires a 2x2 table');
  }

  const [[a, b], [c, d]] = table;

  // Check all values are non-negative integers
  if ([a, b, c, d].some(v => v < 0 || !Number.isInteger(v))) {
    throw new Error('Table entries must be non-negative integers');
  }

  const n = a + b + c + d;
  const row1 = a + b;
  const row2 = c + d;
  const col1 = a + c;
  const col2 = b + d;

  // Hypergeometric probability for a given cell count
  const hypergeoProb = (x) => {
    return (
      factorial(row1) * factorial(row2) * factorial(col1) * factorial(col2) /
      (factorial(x) * factorial(row1 - x) * factorial(col1 - x) *
       factorial(row2 - (col1 - x)) * factorial(n))
    );
  };

  // For two-sided test, sum probabilities <= observed probability
  const observedProb = hypergeoProb(a);
  let pValue;

  if (alternative === 'two-sided') {
    pValue = 0;
    const minA = Math.max(0, col1 - row2);
    const maxA = Math.min(row1, col1);

    for (let x = minA; x <= maxA; x++) {
      const prob = hypergeoProb(x);
      if (prob <= observedProb + 1e-10) {
        pValue += prob;
      }
    }
  } else if (alternative === 'greater') {
    // Greater: test for positive association (OR > 1)
    // p-value = P(X >= observed) = sum from a to max
    pValue = 0;
    for (let x = a; x <= Math.min(row1, col1); x++) {
      pValue += hypergeoProb(x);
    }
  } else {
    // Less: test for negative association (OR < 1)
    // p-value = P(X <= observed) = sum from min to a
    pValue = 0;
    for (let x = Math.max(0, col1 - row2); x <= a; x++) {
      pValue += hypergeoProb(x);
    }
  }

  // Odds ratio
  const oddsRatio = (a * d) / (b * c);

  return {
    pValue: Math.min(pValue, 1.0),
    oddsRatio: isFinite(oddsRatio) ? oddsRatio : (a * d === 0 ? 0 : Infinity),
    alternative
  };
}

/**
 * Factorial function with caching for efficiency
 */
const factorialCache = new Map();
function factorial(n) {
  if (n < 0) throw new Error('Factorial of negative number');
  if (n === 0 || n === 1) return 1;

  if (factorialCache.has(n)) {
    return factorialCache.get(n);
  }

  // Use log-gamma for large n to avoid overflow
  if (n > 170) {
    return Math.exp(logFactorial(n));
  }

  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }

  factorialCache.set(n, result);
  return result;
}

/**
 * Log factorial using Stirling's approximation or gamma function
 */
function logFactorial(n) {
  if (n < 20) {
    return Math.log(factorial(n));
  }

  // Stirling's approximation
  return n * Math.log(n) - n + 0.5 * Math.log(2 * Math.PI * n);
}

// ============= Multiple Testing Corrections =============

/**
 * Bonferroni correction for multiple testing
 * @param {Array<number>} pValues - Array of p-values
 * @param {number} alpha - Family-wise error rate (default 0.05)
 * @returns {Object} {adjustedPValues, rejected, adjustedAlpha}
 */
export function bonferroni(pValues, alpha = 0.05) {
  const m = pValues.length;
  const adjustedAlpha = alpha / m;
  const adjustedPValues = pValues.map(p => Math.min(p * m, 1));
  const rejected = adjustedPValues.map(p => p <= alpha);

  return {
    adjustedPValues,
    rejected,
    adjustedAlpha
  };
}

/**
 * Holm-Bonferroni correction for multiple testing
 * Sequentially rejective Bonferroni procedure (more powerful)
 * @param {Array<number>} pValues - Array of p-values
 * @param {number} alpha - Family-wise error rate (default 0.05)
 * @returns {Object} {adjustedPValues, rejected}
 */
export function holmBonferroni(pValues, alpha = 0.05) {
  const m = pValues.length;

  // Create array with indices
  const indexed = pValues.map((p, i) => ({ p, index: i }));

  // Sort by p-value
  indexed.sort((a, b) => a.p - b.p);

  // Apply Holm-Bonferroni
  const adjustedPValues = new Array(m);
  const rejected = new Array(m);

  for (let i = 0; i < m; i++) {
    const adjustedP = Math.min(indexed[i].p * (m - i), 1);
    // Ensure monotonicity
    if (i > 0) {
      adjustedPValues[indexed[i].index] = Math.max(adjustedP, adjustedPValues[indexed[i - 1].index]);
    } else {
      adjustedPValues[indexed[i].index] = adjustedP;
    }
  }

  // Determine rejections
  for (let i = 0; i < m; i++) {
    rejected[indexed[i].index] = indexed[i].p <= alpha / (m - i);
  }

  return {
    adjustedPValues,
    rejected
  };
}

/**
 * Benjamini-Hochberg FDR correction
 * Controls false discovery rate
 * @param {Array<number>} pValues - Array of p-values
 * @param {number} alpha - False discovery rate (default 0.05)
 * @returns {Object} {adjustedPValues, rejected, criticalValues}
 */
export function fdr(pValues, alpha = 0.05) {
  const m = pValues.length;

  // Create array with indices
  const indexed = pValues.map((p, i) => ({ p, index: i }));

  // Sort by p-value
  indexed.sort((a, b) => a.p - b.p);

  // Apply Benjamini-Hochberg
  const adjustedPValues = new Array(m);
  const rejected = new Array(m).fill(false);
  const criticalValues = new Array(m);

  // Calculate adjusted p-values (from largest to smallest)
  for (let i = m - 1; i >= 0; i--) {
    const rank = i + 1;
    const adjustedP = Math.min((indexed[i].p * m) / rank, 1);

    // Ensure monotonicity
    if (i < m - 1) {
      adjustedPValues[indexed[i].index] = Math.min(adjustedP, adjustedPValues[indexed[i + 1].index]);
    } else {
      adjustedPValues[indexed[i].index] = adjustedP;
    }

    criticalValues[indexed[i].index] = (rank / m) * alpha;
  }

  // Find largest i where p(i) <= (i/m) * alpha
  let maxRejected = -1;
  for (let i = m - 1; i >= 0; i--) {
    if (indexed[i].p <= ((i + 1) / m) * alpha) {
      maxRejected = i;
      break;
    }
  }

  // Reject all hypotheses up to maxRejected
  for (let i = 0; i <= maxRejected; i++) {
    rejected[indexed[i].index] = true;
  }

  return {
    adjustedPValues,
    rejected,
    criticalValues
  };
}
