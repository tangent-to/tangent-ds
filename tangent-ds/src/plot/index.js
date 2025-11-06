/**
 * Visualization module
 * Observable Plot configuration generators for data analysis
 */

// Ordination plots (original implementations)
export { plotPCA, plotScree } from './plotPCA.js';
export { plotLDA } from './plotLDA.js';
export { plotHCA, dendrogramLayout } from './plotHCA.js';
export { plotRDA } from './plotRDA.js';

// Unified ordination plot
export { ordiplot } from './ordiplot.js';

// Classification metrics plots
export {
  plotROC,
  plotPrecisionRecall,
  plotConfusionMatrix,
  plotCalibration
} from './classification.js';

// Model interpretation plots
export {
  plotFeatureImportance,
  plotPartialDependence,
  plotCorrelationMatrix,
  plotResiduals,
  plotQQ,
  plotLearningCurve
} from './utils.js';

// Renderer helpers
export { createD3DendrogramRenderer } from './renderers/d3Dendrogram.js';

export { plotSilhouette } from './plotSilhouette.js';

// GLM/GLMM diagnostic plots
export {
  residualPlot,
  scaleLocationPlot,
  qqPlot,
  residualsLeveragePlot,
  diagnosticDashboard,
  effectPlot,
  partialResidualPlot
} from './diagnostics.js';
