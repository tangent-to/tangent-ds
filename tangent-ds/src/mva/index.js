/**
 * Multivariate Analysis module exports
 *
 * Provide both the functional namespaces (pca, hca, lda, rda) and the
 * class-based PCA and LDA estimators. Import the classes and re-export them
 * together with the functional namespaces to ensure consumers can import
 * either the functional API or the class-based estimators.
 */

import * as pca from './pca.js';
import * as lda from './lda.js';
import * as rda from './rda.js';
import * as cca from './cca.js';

import PCA from './estimators/PCA.js';
import { LDA } from './estimators/LDA.js';
import RDA from './estimators/RDA.js';
import { CCA } from './estimators/CCA.js';

export {
  // Functional namespaces
  pca,
  lda,
  rda,
  cca,

  // Class-based estimators
  PCA,
  LDA,
  RDA,
  CCA
};
