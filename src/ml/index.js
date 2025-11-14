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
import { GridSearchCV, Pipeline } from './pipeline.js';
import { Recipe, recipe } from './recipe.js';

// Imputation (missing data handling)
import { SimpleImputer, KNNImputer, simpleImpute, knnImpute } from './impute.js';

// Class-based estimators (scikit-like)
import { KMeans } from './estimators/KMeans.js';
import { HCA } from './estimators/HCA.js';
import { KNNClassifier, KNNRegressor } from './estimators/KNN.js';
import { DecisionTreeClassifier, DecisionTreeRegressor } from './estimators/DecisionTree.js';
import { RandomForestClassifier, RandomForestRegressor } from './estimators/RandomForest.js';
import { GAMClassifier, GAMRegressor } from './estimators/GAM.js';
import { PolynomialRegressor } from './estimators/PolynomialRegressor.js';
import { MLPRegressor } from './estimators/MLPRegressor.js';

export {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GAMClassifier,
  GAMRegressor,
  GridSearchCV,
  HCA,
  hca,
  // Imputation (missing data)
  SimpleImputer,
  KNNImputer,
  simpleImpute,
  knnImpute,
  // Model interpretation
  interpret,
  KMeans,
  // K-means clustering (functional and class-based)
  kmeans,
  KNNClassifier,
  KNNRegressor,
  // Loss functions
  loss,
  // Metrics
  metrics,
  // Multilayer Perceptron
  mlp,
  MLPRegressor,
  // Pipeline
  Pipeline,
  // Polynomial regression
  polynomial,
  PolynomialRegressor,
  // Preprocessing
  preprocessing,
  RandomForestClassifier,
  RandomForestRegressor,
  Recipe,
  // Recipe pattern for inspectable preprocessing
  recipe,
  // Silhouette analysis utilities
  silhouette,
  // Training utilities
  train,
  // Hyperparameter tuning
  tuning,
  // Utilities
  utils,
  // Validation utilities
  validation,
};
