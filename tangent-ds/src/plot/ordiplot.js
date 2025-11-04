import { attachShow } from './show.js';

/**
 * Unified ordination plot (ordiplot) for PCA, LDA, and RDA
 * Provides a consistent interface for plotting ordination results
 */

/**
 * Generate unified ordination plot configuration
 * Works with PCA, LDA, and RDA results
 *
 * @param {Object} result - Ordination result (from PCA, LDA, or RDA)
 * @param {Object} options - Configuration options
 * @param {string} options.type - Type of ordination ('pca', 'lda', 'rda') - auto-detected if not specified
 * @param {string|null} options.colorBy - Array of colors/groups for points (optional)
 * @param {Array<string>|null} options.labels - Labels for points (optional)
 * @param {boolean} options.showLoadings - Show loading vectors (PCA/RDA only)
 * @param {boolean} options.showCentroids - Show class centroids (LDA only)
 * @param {boolean} options.showConvexHulls - Show convex hulls around groups (optional)
 * @param {number} options.axis1 - First axis to plot (default: 1)
 * @param {number} options.axis2 - Second axis to plot (default: 2)
 * @param {number} options.width - Plot width (default: 640)
 * @param {number} options.height - Plot height (default: 400)
 * @param {number} options.loadingScale - Scale factor for loading vectors (default: 3)
 * @param {number} options.loadingFactor - Multiplier applied to loading vectors (default: 1, set 0 for auto)
 * @param {number|null} options.predictorFactor - Multiplier for predictor arrows (RDA only, default: inherits loadingFactor; set 0 for auto)
 * @returns {Object} Plot configuration
 */
export function ordiplot(result, {
  type = null,
  colorBy = null,
  labels = null,
  showLoadings = true,
  showCentroids = false,
  showConvexHulls = false,
  axis1 = 1,
  axis2 = 2,
  width = 640,
  height = 400,
  loadingScale = 3,
  loadingFactor = 1,
  predictorFactor = null
} = {}) {
  // Auto-detect ordination type if not specified
  if (!type) {
    type = detectOrdinationType(result);
  }

  // Extract scores and construct data based on ordination type
  const { scoresData, loadingsData, centroidsData, axisLabels, predictorData } = extractOrdinationData(
    result,
    type,
    axis1,
    axis2,
    colorBy,
    labels,
    loadingScale
  );

  // Build plot configuration
  const config = {
    type: 'ordiplot',
    ordinationType: type,
    width,
    height,
    data: {
      scores: scoresData
    },
    axes: {
      x: { label: axisLabels.x, grid: true },
      y: { label: axisLabels.y, grid: true }
    },
    marks: []
  };

  // Add convex hulls if requested (must be added first so they're behind points)
  if (showConvexHulls && colorBy) {
    const hullData = computeConvexHulls(scoresData, colorBy);
    if (hullData.length > 0) {
      config.data.hulls = hullData;
      config.marks.push({
        type: 'area',
        data: 'hulls',
        x: 'x',
        y: 'y',
        fill: 'group',
        fillOpacity: 0.2,
        stroke: 'group',
        strokeWidth: 1
      });
    }
  }

  // Add score points
  const hasColorField = scoresData.length > 0 && scoresData.every((d) => typeof d.color !== 'undefined');
  config.marks.push({
    type: 'dot',
    data: 'scores',
    x: 'x',
    y: 'y',
    fill: hasColorField ? 'color' : 'steelblue',
    r: 4,
    fillOpacity: 0.7,
    tip: labels ? true : false
  });

  if (hasColorField) {
    const uniqueColors = Array.from(new Set(scoresData.map((d) => d.color))); // eslint-disable-line no-undef
    config.legends = config.legends || [];
    config.legends.push({
      fill: 'color',
      type: 'symbol',
      title: 'Group',
      categories: uniqueColors
    });
  }

  // Add labels if provided
  if (labels) {
    config.marks.push({
      type: 'text',
      data: 'scores',
      x: 'x',
      y: 'y',
      text: 'label',
      fontSize: 10,
      dx: 8,
      dy: -8
    });
  }

  // Add loadings for PCA, LDA, and RDA
  if (showLoadings && loadingsData && (type === 'pca' || type === 'lda' || type === 'rda')) {
    const computeFactor = (vectors, requestedFactor) => {
      let factor = requestedFactor ?? 1;
      if (factor === 0) {
        const maxScoreRadius = scoresData.reduce(
          (max, point) => Math.max(max, Math.hypot(point.x || 0, point.y || 0)),
          0
        );
        const maxVectorRadius = vectors.reduce(
          (max, vec) => Math.max(max, Math.hypot(vec.x2 || vec.dx || 0, vec.y2 || vec.dy || 0)),
          0
        );
        if (maxVectorRadius > 0) {
          factor = maxScoreRadius > 0 ? maxScoreRadius / maxVectorRadius : 1;
        } else {
          factor = 1;
        }
      }
      return factor;
    };

    const appliedLoadingFactor = computeFactor(loadingsData, loadingFactor);
    const scaledLoadings = appliedLoadingFactor === 1
      ? loadingsData
      : loadingsData.map((loading) => ({
          ...loading,
          x2: (loading.x2 || 0) * appliedLoadingFactor,
          y2: (loading.y2 || 0) * appliedLoadingFactor
        }));

    config.data.loadings = scaledLoadings;

    if (type === 'rda' && predictorData && predictorData.length) {
      const appliedPredictorFactor = computeFactor(
        predictorData,
        predictorFactor === null ? loadingFactor : predictorFactor
      );
      const scaledPredictors = appliedPredictorFactor === 1
        ? predictorData
        : predictorData.map((pred) => ({
            ...pred,
            x2: (pred.x2 || 0) * appliedPredictorFactor,
            y2: (pred.y2 || 0) * appliedPredictorFactor
          }));
      config.data.predictors = scaledPredictors;
    }

    // For RDA triplot: use blue for response loadings
    const loadingColor = (type === 'rda' && predictorData) ? 'blue' : 'red';
    const loadingTextColor = (type === 'rda' && predictorData) ? 'darkblue' : 'darkred';

    config.marks.push({
      type: 'arrow',
      data: 'loadings',
      x1: 'x1',
      y1: 'y1',
      x2: 'x2',
      y2: 'y2',
      stroke: loadingColor,
      strokeWidth: 2,
      headLength: 8
    });
    config.marks.push({
      type: 'text',
      data: 'loadings',
      x: 'x2',
      y: 'y2',
      text: 'variable',
      fontSize: 10,
      fill: loadingTextColor,
      dx: 5,
      dy: 5
    });
  }

  // Add predictor correlations for RDA triplot
  if (predictorData && type === 'rda') {
    if (!config.data.predictors) {
      config.data.predictors = predictorData;
    }
    config.marks.push({
      type: 'arrow',
      data: 'predictors',
      x1: 'x1',
      y1: 'y1',
      x2: 'x2',
      y2: 'y2',
      stroke: 'red',
      strokeWidth: 2,
      headLength: 8
    });
    config.marks.push({
      type: 'text',
      data: 'predictors',
      x: 'x2',
      y: 'y2',
      text: 'variable',
      fontSize: 10,
      fill: 'darkred',
      dx: 5,
      dy: 5
    });
  }

  // Add centroids for LDA
  if (showCentroids && centroidsData && type === 'lda') {
    config.data.centroids = centroidsData;
    config.marks.push({
      type: 'dot',
      data: 'centroids',
      x: 'x',
      y: 'y',
      fill: 'color',
      r: 8,
      stroke: 'black',
      strokeWidth: 2,
      symbol: 'cross'
    });
    config.marks.push({
      type: 'text',
      data: 'centroids',
      x: 'x',
      y: 'y',
      text: 'class',
      fontSize: 12,
      fontWeight: 'bold',
      dy: -15
    });
  }

  return attachShow(config);
}

/**
 * Detect ordination type from result structure
 * @private
 */
function detectOrdinationType(result) {
  // Check for LDA first (most specific - has 'class' field in scores)
  if (result.scores && result.scores[0] && 'class' in result.scores[0]) {
    return 'lda';
  }
  // Check for RDA (has canonical scores/loadings)
  if (result.canonicalScores && result.canonicalLoadings) {
    return 'rda';
  }
  // Check for PCA (has scores, loadings, eigenvalues)
  if (result.scores && result.loadings && result.eigenvalues) {
    return 'pca';
  }
  throw new Error('Cannot detect ordination type. Please specify type option.');
}

/**
 * Extract and format ordination data
 * @private
 */
function extractOrdinationData(result, type, axis1, axis2, colorBy, labels, loadingScale) {
  let scoresData = [];
  let loadingsData = null;
  let centroidsData = null;
  let axisLabels = { x: '', y: '' };

  if (type === 'pca') {
    const { scores, loadings } = result;

    scoresData = scores.map((score, i) => ({
      x: score[`pc${axis1}`],
      y: score[`pc${axis2}`],
      index: i,
      color: colorBy ? colorBy[i] : 'default',
      label: labels ? labels[i] : `${i}`
    }));

    if (loadings) {
      loadingsData = loadings.map((loading, i) => ({
        x1: 0,
        y1: 0,
        x2: loading[`pc${axis1}`] * loadingScale,
        y2: loading[`pc${axis2}`] * loadingScale,
        variable: loading.variable || `Var${i + 1}`,
        index: i
      }));
    }

    axisLabels = {
      x: `PC${axis1}`,
      y: `PC${axis2}`
    };

  } else if (type === 'lda') {
    const { scores, loadings, classMeanScores } = result;

    // Check if we have both axes
    const hasAxis2 = scores[0] && scores[0][`ld${axis2}`] !== undefined;

    scoresData = scores.map((score, i) => ({
      x: score[`ld${axis1}`] || 0,
      y: hasAxis2 ? score[`ld${axis2}`] : 0,
      index: i,
      color: score.class || (colorBy ? colorBy[i] : 'default'),
      label: labels ? labels[i] : `${i}`,
      class: score.class
    }));

    // Extract loadings if available
    if (loadings) {
      loadingsData = loadings.map((loading, i) => ({
        x1: 0,
        y1: 0,
        x2: loading[`ld${axis1}`] * loadingScale,
        y2: hasAxis2 && loading[`ld${axis2}`] !== undefined ? loading[`ld${axis2}`] * loadingScale : 0,
        variable: loading.variable || `Var${i + 1}`,
        index: i
      }));
    }

    // Extract centroids if available
    if (classMeanScores && Array.isArray(classMeanScores)) {
      const classes = [...new Set(scores.map(s => s.class))];
      centroidsData = classMeanScores.map((means, i) => {
        const meanArray = Array.isArray(means) ? means : [means];
        return {
          x: meanArray[axis1 - 1] || 0,
          y: hasAxis2 && meanArray[axis2 - 1] !== undefined ? meanArray[axis2 - 1] : 0,
          class: classes[i] || `Class ${i}`,
          color: classes[i] || `Class ${i}`
        };
      });
    }

    axisLabels = {
      x: `LD${axis1}`,
      y: `LD${axis2}`
    };

  } else if (type === 'rda') {
    const { canonicalScores, canonicalLoadings, predictorCorrelations } = result;

    scoresData = canonicalScores.map((score, i) => ({
      x: score[`rda${axis1}`],
      y: score[`rda${axis2}`],
      index: i,
      color: colorBy ? colorBy[i] : 'default',
      label: labels ? labels[i] : `Site ${i}`
    }));

    // Response loadings (species/response variables)
    if (canonicalLoadings) {
      loadingsData = canonicalLoadings.map((loading, i) => ({
        x1: 0,
        y1: 0,
        x2: loading[`rda${axis1}`] * loadingScale,
        y2: loading[`rda${axis2}`] * loadingScale,
        variable: loading.variable || `Var${i + 1}`,
        index: i,
        type: 'response'
      }));
    }

    // For triplot: use predictor correlations
    let predictorData = null;
    if (predictorCorrelations && predictorCorrelations.length > 0) {
      predictorData = predictorCorrelations.map((corr, i) => ({
        x1: 0,
        y1: 0,
        x2: corr[`rda${axis1}`] * loadingScale,
        y2: corr[`rda${axis2}`] * loadingScale,
        variable: corr.variable || `Pred${i + 1}`,
        index: i
      }));
    }

    axisLabels = {
      x: `RDA${axis1}`,
      y: `RDA${axis2}`
    };

    return { scoresData, loadingsData, centroidsData, axisLabels, predictorData };
  }

  return { scoresData, loadingsData, centroidsData, axisLabels };
}

/**
 * Compute convex hulls for grouped data
 * @private
 */
function computeConvexHulls(scoresData, colorBy) {
  // Group points by color
  const groups = new Map();
  scoresData.forEach((point, i) => {
    const group = colorBy[i];
    if (!groups.has(group)) {
      groups.set(group, []);
    }
    groups.get(group).push(point);
  });

  // Compute convex hull for each group
  const hullData = [];
  for (const [group, points] of groups.entries()) {
    if (points.length < 3) continue; // Need at least 3 points for a hull

    const hull = convexHull(points);
    hull.forEach(point => {
      hullData.push({
        x: point.x,
        y: point.y,
        group
      });
    });
  }

  return hullData;
}

/**
 * Compute convex hull using gift wrapping algorithm
 * @private
 */
function convexHull(points) {
  if (points.length < 3) return points;

  // Find leftmost point
  let leftmost = points[0];
  for (const p of points) {
    if (p.x < leftmost.x || (p.x === leftmost.x && p.y < leftmost.y)) {
      leftmost = p;
    }
  }

  const hull = [];
  let current = leftmost;

  do {
    hull.push(current);
    let next = points[0];

    for (const p of points) {
      if (next === current || isLeftTurn(current, next, p)) {
        next = p;
      }
    }

    current = next;
  } while (current !== leftmost && hull.length < points.length);

  return hull;
}

/**
 * Check if three points make a left turn
 * @private
 */
function isLeftTurn(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
}
