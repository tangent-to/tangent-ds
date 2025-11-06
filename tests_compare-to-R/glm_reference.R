#!/usr/bin/env Rscript

#'
#' R Reference Implementations for GLM and GLMM Testing
#'
#' This script generates reference outputs using:
#' - stats::glm() for fixed-effects GLM
#' - lme4::glmer() for mixed-effects GLMM
#'

library(jsonlite)

# Install lme4 if not available
if (!require("lme4", quietly = TRUE)) {
  install.packages("lme4", repos = "https://cloud.r-project.org/")
  library(lme4)
}

# ==============================================================================
# Test Dataset 1: Simple Linear Regression (Gaussian)
# ==============================================================================

test_gaussian_glm <- function() {
  # Simple linear regression
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 4, 3, 5, 6, 5, 7, 8, 9)
  y <- c(3, 5, 7, 6, 9, 11, 10, 13, 15, 17)

  df <- data.frame(y = y, X1 = X1, X2 = X2)

  # Fit GLM with Gaussian family
  model <- glm(y ~ X1 + X2, data = df, family = gaussian())

  # Extract results
  list(
    test = "gaussian_glm",
    family = "gaussian",
    link = "identity",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model)),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    bic = BIC(model),
    df_residual = df.residual(model),
    converged = model$converged
  )
}

# ==============================================================================
# Test Dataset 2: Logistic Regression (Binomial)
# ==============================================================================

test_binomial_glm <- function() {
  # Logistic regression
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 2, 4, 3, 5, 4, 6, 5, 7)
  y <- c(0, 0, 0, 0, 1, 0, 1, 1, 1, 1)

  df <- data.frame(y = y, X1 = X1, X2 = X2)

  # Fit GLM with binomial family
  model <- glm(y ~ X1 + X2, data = df, family = binomial(link = "logit"))

  list(
    test = "binomial_glm",
    family = "binomial",
    link = "logit",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model, type = "response")),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    bic = BIC(model),
    df_residual = df.residual(model),
    converged = model$converged,
    iterations = model$iter
  )
}

# ==============================================================================
# Test Dataset 3: Poisson Regression (Count Data)
# ==============================================================================

test_poisson_glm <- function() {
  # Poisson regression
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 4, 3, 5, 6, 5, 7, 8, 9)
  y <- c(2, 3, 5, 4, 7, 9, 8, 11, 13, 15)

  df <- data.frame(y = y, X1 = X1, X2 = X2)

  # Fit GLM with Poisson family
  model <- glm(y ~ X1 + X2, data = df, family = poisson(link = "log"))

  list(
    test = "poisson_glm",
    family = "poisson",
    link = "log",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model, type = "response")),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    bic = BIC(model),
    df_residual = df.residual(model),
    converged = model$converged
  )
}

# ==============================================================================
# Test Dataset 4: Gamma Regression (Positive Continuous Data)
# ==============================================================================

test_gamma_glm <- function() {
  # Gamma regression
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 4, 3, 5, 6, 5, 7, 8, 9)
  y <- c(1.2, 2.3, 3.1, 2.8, 4.5, 5.9, 5.2, 7.3, 8.7, 10.1)

  df <- data.frame(y = y, X1 = X1, X2 = X2)

  # Fit GLM with Gamma family
  model <- glm(y ~ X1 + X2, data = df, family = Gamma(link = "inverse"))

  list(
    test = "gamma_glm",
    family = "gamma",
    link = "inverse",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model, type = "response")),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    bic = BIC(model),
    df_residual = df.residual(model),
    converged = model$converged
  )
}

# ==============================================================================
# Test Dataset 5: Inverse Gaussian Regression
# ==============================================================================

test_inverse_gaussian_glm <- function() {
  # Inverse Gaussian regression
  X1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  X2 <- c(2, 3, 4, 3, 5, 6, 5, 7, 8, 9)
  y <- c(0.8, 1.5, 2.2, 1.9, 3.1, 3.8, 3.5, 4.9, 5.7, 6.5)

  df <- data.frame(y = y, X1 = X1, X2 = X2)

  # Fit GLM with inverse.gaussian family
  model <- glm(y ~ X1 + X2, data = df, family = inverse.gaussian(link = "1/mu^2"))

  list(
    test = "inverse_gaussian_glm",
    family = "inverse_gaussian",
    link = "inverse_squared",
    coefficients = as.numeric(coef(model)),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model, type = "response")),
    standard_errors = as.numeric(summary(model)$coefficients[, "Std. Error"]),
    deviance = deviance(model),
    null_deviance = model$null.deviance,
    aic = AIC(model),
    bic = BIC(model),
    df_residual = df.residual(model),
    converged = model$converged
  )
}

# ==============================================================================
# Test Dataset 6: Linear Mixed Model (Gaussian GLMM with random intercepts)
# ==============================================================================

test_gaussian_glmm <- function() {
  # Random intercept model
  X1 <- rep(1:10, each = 3)
  X2 <- c(2,3,4, 3,4,5, 4,5,6, 3,4,5, 5,6,7, 6,7,8, 5,6,7, 7,8,9, 8,9,10, 9,10,11)
  group <- rep(c("A", "B", "C"), 10)
  y <- c(3,4,5, 5,6,7, 7,8,9, 6,7,8, 9,10,11, 11,12,13, 10,11,12, 13,14,15, 15,16,17, 17,18,19)

  df <- data.frame(y = y, X1 = X1, X2 = X2, group = group)

  # Fit GLMM with random intercepts
  model <- lmer(y ~ X1 + X2 + (1 | group), data = df, REML = FALSE)

  # Extract variance components
  vc <- as.data.frame(VarCorr(model))

  list(
    test = "gaussian_glmm",
    family = "gaussian",
    link = "identity",
    fixed_effects = as.numeric(fixef(model)),
    random_effects = as.numeric(ranef(model)$group[, 1]),
    variance_components = list(
      group_intercept = vc$vcov[1],
      residual = vc$vcov[2]
    ),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model)),
    logLik = as.numeric(logLik(model)),
    aic = AIC(model),
    bic = BIC(model),
    ngroups = length(unique(df$group)),
    converged = TRUE
  )
}

# ==============================================================================
# Test Dataset 7: Logistic Mixed Model (Binomial GLMM with random intercepts)
# ==============================================================================

test_binomial_glmm <- function() {
  # Random intercept logistic regression
  X1 <- rep(1:10, each = 3)
  X2 <- rep(c(2, 3, 4), 10)
  group <- rep(c("A", "B", "C"), 10)
  y <- c(0,0,1, 0,0,1, 0,1,1, 0,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1)

  df <- data.frame(y = y, X1 = X1, X2 = X2, group = group)

  # Fit GLMM with random intercepts
  model <- glmer(y ~ X1 + X2 + (1 | group), data = df, family = binomial(link = "logit"))

  # Extract variance components
  vc <- as.data.frame(VarCorr(model))

  list(
    test = "binomial_glmm",
    family = "binomial",
    link = "logit",
    fixed_effects = as.numeric(fixef(model)),
    random_effects = as.numeric(ranef(model)$group[, 1]),
    variance_components = list(
      group_intercept = vc$vcov[1]
    ),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model, type = "response")),
    logLik = as.numeric(logLik(model)),
    aic = AIC(model),
    bic = BIC(model),
    ngroups = length(unique(df$group)),
    converged = is.null(model@optinfo$conv$lme4$code)
  )
}

# ==============================================================================
# Test Dataset 8: Poisson Mixed Model (Poisson GLMM with random intercepts)
# ==============================================================================

test_poisson_glmm <- function() {
  # Random intercept Poisson regression
  X1 <- rep(1:10, each = 3)
  X2 <- rep(c(2, 3, 4), 10)
  group <- rep(c("A", "B", "C"), 10)
  y <- c(2,3,4, 3,4,5, 5,6,7, 4,5,6, 7,8,9, 9,10,11, 8,9,10, 11,12,13, 13,14,15, 15,16,17)

  df <- data.frame(y = y, X1 = X1, X2 = X2, group = group)

  # Fit GLMM with random intercepts
  model <- glmer(y ~ X1 + X2 + (1 | group), data = df, family = poisson(link = "log"))

  # Extract variance components
  vc <- as.data.frame(VarCorr(model))

  list(
    test = "poisson_glmm",
    family = "poisson",
    link = "log",
    fixed_effects = as.numeric(fixef(model)),
    random_effects = as.numeric(ranef(model)$group[, 1]),
    variance_components = list(
      group_intercept = vc$vcov[1]
    ),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model, type = "response")),
    logLik = as.numeric(logLik(model)),
    aic = AIC(model),
    bic = BIC(model),
    ngroups = length(unique(df$group)),
    converged = is.null(model@optinfo$conv$lme4$code)
  )
}

# ==============================================================================
# Test Dataset 9: Mixed Model with Random Slopes
# ==============================================================================

test_gaussian_glmm_slopes <- function() {
  # Random intercept and slope model
  time <- rep(0:9, each = 3)
  X2 <- c(2,3,4, 3,4,5, 4,5,6, 3,4,5, 5,6,7, 6,7,8, 5,6,7, 7,8,9, 8,9,10, 9,10,11)
  group <- rep(c("A", "B", "C"), 10)
  y <- c(3,4,5, 5,6,7, 7,8,9, 6,7,8, 9,10,11, 11,12,13, 10,11,12, 13,14,15, 15,16,17, 17,18,19)

  df <- data.frame(y = y, time = time, X2 = X2, group = group)

  # Fit GLMM with random intercepts and slopes
  model <- lmer(y ~ time + X2 + (1 + time | group), data = df, REML = FALSE)

  # Extract variance components
  vc <- as.data.frame(VarCorr(model))

  # Extract random effects
  re <- ranef(model)$group

  list(
    test = "gaussian_glmm_slopes",
    family = "gaussian",
    link = "identity",
    fixed_effects = as.numeric(fixef(model)),
    random_effects_intercept = as.numeric(re[, 1]),
    random_effects_slope = as.numeric(re[, 2]),
    variance_components = list(
      group_intercept = vc$vcov[1],
      group_slope = vc$vcov[2],
      residual = vc$vcov[3]
    ),
    fitted_values = as.numeric(fitted(model)),
    residuals = as.numeric(residuals(model)),
    logLik = as.numeric(logLik(model)),
    aic = AIC(model),
    bic = BIC(model),
    ngroups = length(unique(df$group)),
    converged = TRUE
  )
}

# ==============================================================================
# Run All Tests and Export to JSON
# ==============================================================================

run_all_tests <- function() {
  results <- list(
    gaussian_glm = test_gaussian_glm(),
    binomial_glm = test_binomial_glm(),
    poisson_glm = test_poisson_glm(),
    gamma_glm = test_gamma_glm(),
    inverse_gaussian_glm = test_inverse_gaussian_glm(),
    gaussian_glmm = test_gaussian_glmm(),
    binomial_glmm = test_binomial_glmm(),
    poisson_glmm = test_poisson_glmm(),
    gaussian_glmm_slopes = test_gaussian_glmm_slopes()
  )

  # Write to JSON file
  json_output <- toJSON(results, pretty = TRUE, auto_unbox = TRUE)
  write(json_output, file = "glm_reference_results.json")

  cat("R reference results written to glm_reference_results.json\n")

  return(results)
}

# Run if executed as script
if (!interactive()) {
  run_all_tests()
}
