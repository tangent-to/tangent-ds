/**
 * Machine Learning module exports
 */

import * as kmeans from './kmeans.js';
import * as hca from './hca.js';
import * as polynomial from './polynomial.js';
import * as silhouette from './silhouette.js';
import * as mlp from './mlp.js';
import * as validation from './validation.js';
import * as preprocessing from './preprocessing.js';
import * as metrics from './metrics.js';
import * as utils from './utils.js';
import * as interpret from './interpret.js';
import * as loss from './loss.js';
import * as train from './train.js';
import * as tuning from './tuning.js';
import { Pipeline, GridSearchCV } from './pipeline.js';

// Class-based estimators (scikit-like)
import { KMeans } from './estimators/KMeans.js';
import { HCA } from './estimators/HCA.js';
import { KNNClassifier, KNNRegressor } from './estimators/KNN.js';
import { DecisionTreeClassifier, DecisionTreeRegressor } from './estimators/DecisionTree.js';
import { RandomForestClassifier, RandomForestRegressor } from './estimators/RandomForest.js';
import { GAMRegressor, GAMClassifier } from './estimators/GAM.js';
import { PolynomialRegressor } from './estimators/PolynomialRegressor.js';
import { MLPRegressor } from './estimators/MLPRegressor.js';

export {
  // K-means clustering (functional and class-based)
  kmeans,
  KMeans,
  hca,
  HCA,
  KNNClassifier,
  KNNRegressor,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
  GAMRegressor,
  GAMClassifier,
  PolynomialRegressor,
  MLPRegressor,

  // Polynomial regression
  polynomial,

  // Silhouette analysis utilities
  silhouette,

  // Multilayer Perceptron
  mlp,

  // Validation utilities
  validation,

  // Preprocessing
  preprocessing,

  // Metrics
  metrics,

  // Utilities
  utils,

  // Model interpretation
  interpret,

  // Loss functions
  loss,

  // Training utilities
  train,

  // Hyperparameter tuning
  tuning,

  // Pipeline
  Pipeline,
  GridSearchCV
};
