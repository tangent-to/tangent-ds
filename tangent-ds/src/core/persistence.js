/**
 * Model Persistence
 * Save and load models in JSON format
 */

/**
 * Save model to JSON
 * @param {Object} model - Model object with toJSON method or estimator instance
 * @returns {string} JSON string representation of the model
 */
export function saveModel(model) {
  if (!model) {
    throw new Error('Model cannot be null or undefined');
  }

  // Check if model has a toJSON method
  if (typeof model.toJSON === 'function') {
    const json = model.toJSON();
    return JSON.stringify(json, null, 2);
  }

  // Check if it's an estimator with model property
  if (model.model && typeof model.model === 'object') {
    return saveModel(model.model);
  }

  // For plain objects, add metadata and serialize
  const serialized = {
    __tangentds__: true,
    version: '0.7.0',
    timestamp: new Date().toISOString(),
    modelType: detectModelType(model),
    data: model
  };

  return JSON.stringify(serialized, null, 2);
}

/**
 * Load model from JSON
 * @param {string} json - JSON string representation
 * @returns {Object} Reconstructed model object
 */
export function loadModel(json) {
  if (typeof json !== 'string') {
    throw new Error('JSON must be a string');
  }

  const parsed = JSON.parse(json);

  // Check if it's a tangent-ds model
  if (parsed.__tangentds__) {
    return parsed.data;
  }

  // Otherwise return as-is
  return parsed;
}

/**
 * Detect model type from structure
 * @private
 */
function detectModelType(model) {
  // Statistics models
  if (model.coefficients && model.residuals) {
    return 'linear_model';
  }
  if (model.coefficients && model.logLikelihood !== undefined) {
    return 'logistic_regression';
  }

  // MVA models
  if (model.constraintScores && model.eigenvalues) {
    return 'rda';
  }
  if (model.scores && model.loadings && model.eigenvalues) {
    return 'pca';
  }
  if (model.scores && model.discriminantAxes) {
    return 'lda';
  }
  if (model.canonicalScores && model.canonicalLoadings) {
    return 'rda';
  }
  if (model.dendrogram) {
    return 'hca';
  }

  // ML models
  if (model.centroids && model.labels) {
    return 'kmeans';
  }
  if (model.trainX && model.trainY && model.k) {
    return 'knn';
  }
  if (model.root && model.maxDepth !== undefined) {
    return 'decision_tree';
  }
  if (model.trees && Array.isArray(model.trees)) {
    return 'random_forest';
  }

  return 'unknown';
}

/**
 * Add toJSON method to an estimator class prototype
 * This allows models to define their own serialization logic
 * @param {Function} EstimatorClass - Estimator class
 * @param {Function} toJSONFn - Custom toJSON function
 */
export function addSerializationSupport(EstimatorClass, toJSONFn) {
  EstimatorClass.prototype.toJSON = toJSONFn;
}

/**
 * Serialize model to file-safe object
 * Handles special types like undefined, Infinity, NaN
 * @param {any} value - Value to serialize
 * @returns {any} Serializable value
 */
export function serializeValue(value) {
  if (value === undefined) {
    return { __type__: 'undefined' };
  }
  if (value === Infinity) {
    return { __type__: 'Infinity' };
  }
  if (value === -Infinity) {
    return { __type__: '-Infinity' };
  }
  if (Number.isNaN(value)) {
    return { __type__: 'NaN' };
  }
  if (Array.isArray(value)) {
    return value.map(serializeValue);
  }
  if (value && typeof value === 'object' && value.constructor === Object) {
    const serialized = {};
    for (const [key, val] of Object.entries(value)) {
      serialized[key] = serializeValue(val);
    }
    return serialized;
  }
  return value;
}

/**
 * Deserialize value from file-safe object
 * @param {any} value - Value to deserialize
 * @returns {any} Deserialized value
 */
export function deserializeValue(value) {
  if (value && typeof value === 'object' && value.__type__) {
    switch (value.__type__) {
      case 'undefined':
        return undefined;
      case 'Infinity':
        return Infinity;
      case '-Infinity':
        return -Infinity;
      case 'NaN':
        return NaN;
    }
  }
  if (Array.isArray(value)) {
    return value.map(deserializeValue);
  }
  if (value && typeof value === 'object' && value.constructor === Object) {
    const deserialized = {};
    for (const [key, val] of Object.entries(value)) {
      deserialized[key] = deserializeValue(val);
    }
    return deserialized;
  }
  return value;
}

/**
 * Create a saveable model wrapper
 * Adds save() method to any model object
 * @param {Object} model - Model object
 * @returns {Object} Model with save() method
 */
export function makeSaveable(model) {
  return {
    ...model,
    save() {
      return saveModel(this);
    }
  };
}
