context("Forecasting")

library(gbcode)
library(forecast)

#Set up - Data
X <- read.csv("testdata/Sigma4.ssv",sep=" ",header = TRUE)
splitting_point <- round(2*nrow(X)/3)
X_train <- scale(X[1:splitting_point,])

# Set up - Parameters
parameters <- list()
k <- 3
h <- 5
parameters$m <- 3
parameters$C <- 3
parameters$Kmin <- 2
parameters$FF <- 0

# Set up - Components
PCA_dec_results <- dimensionalityReduction(X_train,k,family="PCA")
Z_train <- PCA_dec_results$Z

test_that("[Forecast] - ES", {
  parameters$method <- "ES"
  forecast_results <- forecaster(Z_train,"M4Methods",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] Holt-Winters", {
  parameters$method <- "holt_winters"
  forecast_results <- forecaster(Z_train,"M4Methods",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] Holt-Winters Damped", {
  parameters$method <- "holt_winters_damped"
  forecast_results <- forecaster(Z_train,"M4Methods",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] Theta", {
  parameters$method <- "theta"
  forecast_results <- forecaster(Z_train,"M4Methods",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] Combined", {
  parameters$method <- "combined"
  forecast_results <- forecaster(Z_train,"M4Methods",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

#c("arima","direct","iter","lazydirect","lazyiter","rfdirect","rfiter","mimo","mimo.comb","mimo.acf","mimo.acf.lin","mimo.pls","mimo.lin.pls")
test_that("[Forecast] ARIMA", {
  parameters$method <- "arima"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] KNN direct", {
  parameters$method <- "direct"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] KNN recursive", {
  parameters$method <- "iter"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] Lazy learning direct", {
  parameters$method <- "lazydirect"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] Lazy learning recursive", {
  parameters$method <- "lazyiter"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] RF direct", {
  parameters$method <- "rfdirect"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] MIMO", {
  parameters$method <- "mimo"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] MIMO Combined", {
  parameters$method <- "mimo.comb"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] MIMO ACF", {
  parameters$method <- "mimo.acf"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] MIMO ACF Linear", {
  parameters$method <- "mimo.acf.lin"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] MIMO PLS", {
  parameters$method <- "mimo.pls"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] MIMO ACF Linear", {
  parameters$method <- "mimo.lin.pls"
  forecast_results <- forecaster(Z_train,"multistepAhead",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})

test_that("[Forecast] VAR", {
  forecast_results <- forecaster(Z_train,"VAR",parameters,h)
  expect_true(("matrix" %in% class(forecast_results$Z_hat)))
  expect_equal(dim(forecast_results$Z_hat), c(h,k))
  expect_true(is.numeric(forecast_results$time_forecast))
})
