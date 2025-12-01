/**
 * Machine Learning module exports
 */

import * as kmeans from "./kmeans.js";
import * as dbscan from "./dbscan.js";
import * as hca from "./hca.js";
import * as polynomial from "./polynomial.js";
import * as silhouette from "./silhouette.js";
import * as mlp from "./mlp.js";
import * as validation from "./validation.js";
import * as preprocessing from "./preprocessing.js";
import * as metrics from "./metrics.js";
import * as distances from "./distances.js";
import * as criteria from "./criteria.js";
import * as utils from "./utils.js";
import * as interpret from "./interpret.js";
import * as loss from "./loss.js";
import * as train from "./train.js";
import * as tuning from "./tuning.js";
import { GridSearchCV, Pipeline } from "./pipeline.js";
import { Recipe, recipe } from "./recipe.js";
import { BranchPipeline } from "../pipeline/BranchPipeline.js";

// Imputation (missing data handling)
import {
  SimpleImputer,
  KNNImputer,
  IterativeImputer,
  simpleImpute,
  knnImpute,
  iterativeImpute,
} from "./impute.js";

// Outlier detection
import {
  IsolationForest,
  LocalOutlierFactor,
  MahalanobisDistance,
  isolationForest,
  localOutlierFactor,
  mahalanobisDistance,
} from "./outliers.js";

// Class-based estimators (scikit-like)
import { KMeans } from "./estimators/KMeans.js";
import { DBSCAN } from "./estimators/DBSCAN.js";
import { HCA } from "./estimators/HCA.js";
import { ConsensusCluster } from "../clustering/ConsensusCluster.js";
import { KNNClassifier, KNNRegressor } from "./estimators/KNN.js";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
} from "./estimators/DecisionTree.js";
import {
  RandomForestClassifier,
  RandomForestRegressor,
} from "./estimators/RandomForest.js";
import { GAMClassifier, GAMRegressor } from "./estimators/GAM.js";
import { PolynomialRegressor } from "./estimators/PolynomialRegressor.js";
import { MLPRegressor } from "./estimators/MLPRegressor.js";
import { GaussianProcessRegressor } from "./estimators/GaussianProcessRegressor.js";

// Kernels for Gaussian Processes
import { Kernel, RBF, Periodic, RationalQuadratic } from "./kernels/index.js";

export {
  // Clustering algorithms
  DBSCAN,
  dbscan,
  ConsensusCluster,
  BranchPipeline,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GAMClassifier,
  GAMRegressor,
  // Gaussian Process
  GaussianProcessRegressor,
  Kernel,
  RBF,
  Periodic,
  RationalQuadratic,
  GridSearchCV,
  HCA,
  hca,
  // Imputation (missing data)
  SimpleImputer,
  KNNImputer,
  IterativeImputer,
  simpleImpute,
  knnImpute,
  iterativeImpute,
  // Outlier detection
  IsolationForest,
  LocalOutlierFactor,
  MahalanobisDistance,
  isolationForest,
  localOutlierFactor,
  mahalanobisDistance,
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
  // Distance metrics
  distances,
  // Impurity criteria
  criteria,
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
