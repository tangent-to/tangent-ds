/**
 * Linear algebra operations wrapping ml-matrix
 * This abstraction allows future backend swaps (e.g., WASM)
 */

import { Matrix, SingularValueDecomposition, EigenvalueDecomposition, inverse as mlInverse, pseudoInverse as mlPseudoInverse } from 'ml-matrix';

/**
 * Convert array-like structure to Matrix
 * @param {Array<Array<number>>|Matrix} data - Input data
 * @returns {Matrix} Matrix object
 */
export function toMatrix(data) {
  if (data instanceof Matrix) {
    return data;
  }
  return new Matrix(data);
}

/**
 * Solve least squares problem: minimize ||Ax - b||^2
 * @param {Array<Array<number>>|Matrix} A - Design matrix
 * @param {Array<number>|Array<Array<number>>|Matrix} b - Target vector/matrix
 * @returns {Matrix} Solution x
 */
export function solveLeastSquares(A, b) {
  const matA = toMatrix(A);
  const matB = Array.isArray(b) && !Array.isArray(b[0]) 
    ? Matrix.columnVector(b) 
    : toMatrix(b);
  
  // Solve using normal equations: (A'A)x = A'b
  const AtA = matA.transpose().mmul(matA);
  const Atb = matA.transpose().mmul(matB);
  
  try {
    return AtA.solve(Atb);
  } catch (e) {
    // If singular, use pseudoinverse via SVD
    const svd = new SingularValueDecomposition(matA);
    return svd.solve(matB);
  }
}

/**
 * Compute covariance matrix
 * @param {Array<Array<number>>|Matrix} data - Data matrix (rows = observations)
 * @param {boolean} center - If true, center the data
 * @returns {Matrix} Covariance matrix
 */
export function covarianceMatrix(data, center = true) {
  let mat = toMatrix(data);
  
  if (center) {
    // Center each column
    const means = mat.mean('column');
    mat = mat.clone();
    for (let i = 0; i < mat.rows; i++) {
      for (let j = 0; j < mat.columns; j++) {
        mat.set(i, j, mat.get(i, j) - means[j]);
      }
    }
  }
  
  const n = mat.rows;
  return mat.transpose().mmul(mat).div(n - 1);
}

/**
 * Singular Value Decomposition
 * @param {Array<Array<number>>|Matrix} data - Input matrix
 * @returns {Object} {U, s, V} where data ≈ U * diag(s) * V'
 */
export function svd(data) {
  const mat = toMatrix(data);
  const decomp = new SingularValueDecomposition(mat);
  return {
    U: decomp.leftSingularVectors,
    s: decomp.diagonal,
    V: decomp.rightSingularVectors
  };
}

/**
 * Eigenvalue decomposition
 * @param {Array<Array<number>>|Matrix} data - Square matrix
 * @returns {Object} {values, vectors}
 */
export function eig(data) {
  const mat = toMatrix(data);
  const decomp = new EigenvalueDecomposition(mat);
  return {
    values: decomp.realEigenvalues,
    vectors: decomp.eigenvectorMatrix
  };
}

/**
 * Matrix multiplication
 * @param {Array<Array<number>>|Matrix} A - First matrix
 * @param {Array<Array<number>>|Matrix} B - Second matrix
 * @returns {Matrix} A * B
 */
export function mmul(A, B) {
  return toMatrix(A).mmul(toMatrix(B));
}

/**
 * Matrix transpose
 * @param {Array<Array<number>>|Matrix} data - Input matrix
 * @returns {Matrix} Transposed matrix
 */
export function transpose(data) {
  return toMatrix(data).transpose();
}

/**
 * Matrix inverse
 * @param {Array<Array<number>>|Matrix} data - Square matrix
 * @returns {Matrix} Inverse matrix
 */
export function inverse(data) {
  const mat = toMatrix(data);
  // Use the static inverse function from ml-matrix
  return mlInverse(mat);
}

export function pseudoInverse(data) {
  const mat = toMatrix(data);
  return mlPseudoInverse(mat);
}

export { Matrix };
