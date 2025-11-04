#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required. It ships with base R but must be available.")
  }
  if (!requireNamespace("vegan", quietly = TRUE)) {
    stop("Package 'vegan' is required. Install with install.packages('vegan').")
  }
})

library(jsonlite)
library(MASS)
library(vegan)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: penguins_reference.R <input_json> <output_json>")
}

matrix_to_rows <- function(mat) {
  if (is.null(mat)) {
    return(list())
  }
  mat <- as.matrix(mat)
  lapply(seq_len(nrow(mat)), function(i) {
    as.numeric(mat[i, , drop = TRUE])
  })
}

format_matrix <- function(mat, row_names = NULL, col_names = NULL) {
  if (is.null(mat)) {
    return(list(data = list(), rows = list(), columns = list()))
  }
  mat <- as.matrix(mat)
  if (!is.null(row_names)) {
    rownames(mat) <- row_names
  }
  if (!is.null(col_names)) {
    colnames(mat) <- col_names
  }
  if (is.null(rownames(mat))) {
    rownames(mat) <- as.character(seq_len(nrow(mat)))
  }
  if (is.null(colnames(mat))) {
    colnames(mat) <- paste0("col", seq_len(ncol(mat)))
  }
  list(
    data = matrix_to_rows(mat),
    rows = as.list(rownames(mat)),
    columns = as.list(colnames(mat))
  )
}

payload <- fromJSON(args[[1]], simplifyVector = TRUE)

df <- as.data.frame(payload$data, stringsAsFactors = FALSE)
numeric_cols <- unlist(payload$numericColumns, use.names = FALSE)
class_col <- payload$classColumn
response_cols <- unlist(payload$responseColumns, use.names = FALSE)
predictor_cols <- unlist(payload$predictorColumns, use.names = FALSE)
pca_opts <- payload$pca
lda_opts <- payload$lda
rda_opts <- payload$rda

if (is.null(df) || !nrow(df)) {
  stop("Input data frame is empty.")
}

for (col in predictor_cols) {
  df[[col]] <- factor(df[[col]], levels = unique(df[[col]]))
}
df[[class_col]] <- factor(df[[class_col]], levels = unique(df[[class_col]]))

result <- list()

## PCA section ---------------------------------------------------------------
numeric_mat <- as.matrix(df[, numeric_cols, drop = FALSE])
rownames(numeric_mat) <- as.character(seq_len(nrow(numeric_mat)))

pca_center <- if (isTRUE(pca_opts$center)) colMeans(numeric_mat) else rep(0, ncol(numeric_mat))
centered <- if (isTRUE(pca_opts$center)) sweep(numeric_mat, 2, pca_center, "-") else numeric_mat

if (isTRUE(pca_opts$scale)) {
  pca_scale <- apply(centered, 2, function(col) {
    sqrt(mean(col^2))
  })
  pca_scale[pca_scale == 0] <- 1
  pca_input <- sweep(centered, 2, pca_scale, "/")
} else {
  pca_scale <- rep(1, length(pca_center))
  pca_input <- centered
}

pca_fit <- prcomp(pca_input, center = FALSE, scale. = FALSE)
n_samples <- nrow(pca_input)
score_cols <- paste0("pc", seq_len(ncol(pca_fit$x)))
loading_cols <- score_cols
rownames(pca_fit$x) <- rownames(numeric_mat)
colnames(pca_fit$x) <- score_cols
colnames(pca_fit$rotation) <- loading_cols

singular_values <- pca_fit$sdev * sqrt(max(n_samples - 1, 1))
eigenvalues <- (pca_fit$sdev)^2

pca_scalings <- list()
for (sc in unlist(pca_opts$scaling, use.names = FALSE)) {
  sc_name <- paste0("scaling_", sc)
  scores <- pca_fit$x
  loadings <- pca_fit$rotation
  if (sc == 1) {
    scores <- scores / sqrt(max(n_samples - 1, 1))
  } else if (sc == 2) {
    scores <- sweep(scores, 2, singular_values, FUN = "/")
    loadings <- loadings %*% diag(pca_fit$sdev)
  }
  pca_scalings[[sc_name]] <- list(
    scores = format_matrix(scores),
    loadings = format_matrix(loadings, rownames(pca_fit$rotation), colnames(pca_fit$rotation))
  )
}

result$pca <- list(
  columns = as.list(numeric_cols),
  center = as.numeric(pca_center),
  scale = as.numeric(pca_scale),
  eigenvalues = as.numeric(eigenvalues),
  singularValues = as.numeric(singular_values),
  scalings = pca_scalings
)

## LDA section ---------------------------------------------------------------
lda_mat <- as.matrix(df[, numeric_cols, drop = FALSE])
rownames(lda_mat) <- as.character(seq_len(nrow(lda_mat)))
lda_center <- colMeans(lda_mat)
lda_centered <- sweep(lda_mat, 2, lda_center, "-")

if (isTRUE(lda_opts$scale)) {
  lda_scale <- apply(lda_centered, 2, function(col) sqrt(mean(col^2)))
  lda_scale[lda_scale == 0] <- 1
  lda_input <- sweep(lda_centered, 2, lda_scale, "/")
} else {
  lda_scale <- rep(1, length(lda_center))
  lda_input <- lda_centered
}

lda_fit <- lda(x = lda_input, grouping = df[[class_col]])
lda_pred <- predict(lda_fit)
lda_scores <- as.matrix(lda_pred$x)
if (is.null(lda_scores)) {
  lda_scores <- matrix(0, nrow = nrow(lda_input), ncol = 0)
}
if (ncol(lda_scores) > 0) {
  colnames(lda_scores) <- paste0("ld", seq_len(ncol(lda_scores)))
}
rownames(lda_scores) <- rownames(lda_mat)

lda_loadings <- as.matrix(lda_fit$scaling)
if (!is.null(lda_loadings) && ncol(lda_loadings) > 0) {
  colnames(lda_loadings) <- paste0("ld", seq_len(ncol(lda_loadings)))
  rownames(lda_loadings) <- numeric_cols
}

lda_singular <- as.numeric(lda_fit$svd)
lda_eigenvalues <- lda_singular^2

lda_scalings <- list()
for (sc in unlist(lda_opts$scaling, use.names = FALSE)) {
  sc_name <- paste0("scaling_", sc)
  scores <- lda_scores
  loadings <- lda_loadings
  if (!is.null(scores) && ncol(scores) > 0) {
    if (sc == 1) {
      scores <- scores / sqrt(max(nrow(lda_input) - 1, 1))
    } else if (sc == 2) {
      divisor <- ifelse(lda_singular == 0, 1, lda_singular)
      scores <- sweep(scores, 2, divisor, "/")
      if (!is.null(loadings)) {
        loadings <- loadings %*% diag(divisor)
      }
    }
  }
  lda_scalings[[sc_name]] <- list(
    scores = format_matrix(scores),
    loadings = format_matrix(loadings, rownames(lda_loadings), colnames(lda_loadings))
  )
}

result$lda <- list(
  columns = as.list(numeric_cols),
  classColumn = class_col,
  center = as.numeric(lda_center),
  scale = as.numeric(lda_scale),
  singularValues = lda_singular,
  eigenvalues = lda_eigenvalues,
  scalings = lda_scalings
)

## RDA section ---------------------------------------------------------------
response_mat <- as.matrix(df[, response_cols, drop = FALSE])
rownames(response_mat) <- as.character(seq_len(nrow(response_mat)))
rda_center <- colMeans(response_mat)
rda_centered <- sweep(response_mat, 2, rda_center, "-")

if (isTRUE(rda_opts$scale)) {
  rda_scale <- apply(rda_centered, 2, function(col) sqrt(mean(col^2)))
  rda_scale[rda_scale == 0] <- 1
  rda_response <- sweep(rda_centered, 2, rda_scale, "/")
} else {
  rda_scale <- rep(1, length(rda_center))
  rda_response <- rda_centered
}

predictor_formula <- as.formula(
  paste("~ 0 +", paste(predictor_cols, collapse = " + "))
)
predictor_matrix <- model.matrix(predictor_formula, data = df)
rename_ohe <- function(name) {
  for (base_col in predictor_cols) {
    prefix_len <- nchar(base_col)
    if (startsWith(name, base_col)) {
      level <- substr(name, prefix_len + 1, nchar(name))
      return(paste0(base_col, "_", level))
    }
  }
  name
}
colnames(predictor_matrix) <- vapply(colnames(predictor_matrix), rename_ohe, character(1))

rda_fit <- rda(rda_response, predictor_matrix, scale = FALSE)
rda_eigen <- as.numeric(rda_fit$CCA$eig)

rda_scalings <- list()
sc_values <- unlist(rda_opts$scaling, use.names = FALSE)
if (length(sc_values) == 0) {
  sc_values <- c(0)
}
for (sc in sc_values) {
  sc_name <- paste0("scaling_", sc)
  site_scores <- scores(rda_fit, display = "sites", scaling = sc, choices = seq_along(rda_eigen))
  species_scores <- scores(rda_fit, display = "species", scaling = sc, choices = seq_along(rda_eigen))
  if (!is.null(site_scores)) {
    colnames(site_scores) <- paste0("rda", seq_len(ncol(site_scores)))
  }
  if (!is.null(species_scores)) {
    colnames(species_scores) <- paste0("rda", seq_len(ncol(species_scores)))
  }
  rda_scalings[[sc_name]] <- list(
    sites = format_matrix(site_scores),
    species = format_matrix(species_scores, response_cols, colnames(species_scores))
  )
}

result$rda <- list(
  responseColumns = as.list(response_cols),
  predictorColumns = as.list(colnames(predictor_matrix)),
  center = as.numeric(rda_center),
  scale = as.numeric(rda_scale),
  eigenvalues = rda_eigen,
  scalings = rda_scalings
)

write_json(result, args[[2]], auto_unbox = TRUE, digits = 15)
