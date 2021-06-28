context("DFML - Test M4")

library(keras)
library(dplyr)
library(forecast)
library(MEMTS)

#Set up - Data
X <- read.csv("testdata/Sigma4.ssv",sep=" ",header = TRUE)
X <- as.matrix(X)
splitting_point <- round(2*nrow(X)/3)
X_train <- scale(X[1:splitting_point,])

components <- 3
horizon <- 2
time_window <- horizon

ss_results.df <- data.frame(DimensionalityMethod=character(),ForecastingMethod=character(),Dataset=character(),Horizon=numeric(),Columns=numeric(),Time=numeric(),MSE=numeric(),Samples=numeric(),stringsAsFactors = FALSE)

# If there is an error with the Arima package, check
# detach("package:dse", unload=TRUE)

test_that("[DFML] - PCA + M4 methods", {
  forecast_params <- list()
  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "PCA",
                                  "M4Methods",
                                  dimensionality_parameters = NULL,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="PCA",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - Base autoencoder + M4 methods", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "base"
  dim_params$epochs <- 10

  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Base Autoencoder",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep autoencoder + M4 methods", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "deep"
  dim_params$deep_layers <- c(10,5,3)
  dim_params$epochs <- 10

  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep Autoencoder",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - LSTM autoencoder + M4 methods", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "lstm"
  dim_params$time_window <- time_window
  dim_params$epochs <- 10

  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="LSTM Autoencoder",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep LSTM autoencoder + M4 methods", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "deep_lstm"
  dim_params$time_window <- time_window
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- 10

  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep LSTM Autoencoder",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - GRU autoencoder + M4 methods", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "gru"
  dim_params$time_window <- time_window
  dim_params$epochs <- 10

  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="GRU Autoencoder",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep GRU autoencoder + M4 methods", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "deep_gru"
  dim_params$time_window <- time_window
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- 10

  for (forecasting_method in ExtendedDFML::M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <-  MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep GRU Autoencoder",
                                          ForecastingMethod=forecasting_method,
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})
