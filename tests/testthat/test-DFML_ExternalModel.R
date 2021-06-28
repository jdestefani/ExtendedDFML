context("DFML - Test DFML with external dimensionality model")

library(keras)
library(dplyr)
library(MEMTS)

#Set up - Data
X <- EuStockMarkets
X <- as.matrix(X)
splitting_point <- round(2*nrow(X)/3)
X_train <- scale(X[1:splitting_point,])
components <- 3
horizon <- 5
epochs <- 10

ss_results.df <- data.frame(DimensionalityMethod=character(),ForecastingMethod=character(),Dataset=character(),Horizon=numeric(),Columns=numeric(),Time=numeric(),MSE=numeric(),Samples=numeric(),stringsAsFactors = FALSE)

test_that("[DFML] - PCA External model", {

  forecast_params <- list()
  dim_params <- list()

  dim_res <- dimensionalityReduction(X_train,components,"PCA",NULL)
  dim_params$model <- dim_res$model
  dim_params$time_dim <- dim_res$time_dim

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "PCA",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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

test_that("[DFML] - Base autoencoder external", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "base"
  dim_params$epochs <- epochs
  
  dim_res <- dimensionalityReduction(X_train,components,"Autoencoder_Keras",dim_params)
  dim_params$model <- dim_res$model
  dim_params$time_dim <- dim_res$time_dim

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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

test_that("[DFML] - Deep autoencoder external", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "deep"
  dim_params$deep_layers <- c(10,5,3)
  dim_params$epochs <- epochs

  dim_res <- dimensionalityReduction(X_train,components,"Autoencoder_Keras",dim_params)
  dim_params$model <- dim_res$model
  dim_params$time_dim <- dim_res$time_dim

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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

test_that("[DFML] - LSTM autoencoder external", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "lstm"
  dim_params$time_window <- 5
  dim_params$epochs <- epochs

  dim_res <- dimensionalityReduction(X_train,components,"Autoencoder_Keras",dim_params)
  dim_params$model <- dim_res$model
  dim_params$time_dim <- dim_res$time_dim

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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

test_that("[DFML] - Deep LSTM autoencoder external", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "deep_lstm"
  dim_params$time_window <- 5
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- epochs

  dim_res <- dimensionalityReduction(X_train,components,"Autoencoder_Keras",dim_params)
  dim_params$model <- dim_res$model
  dim_params$time_dim <- dim_res$time_dim

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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

test_that("[DFML] - GRU autoencoder external", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "gru"
  dim_params$time_window <- 5
  dim_params$epochs <- epochs

  dim_res <- dimensionalityReduction(X_train,components,"Autoencoder_Keras",dim_params)
  dim_params$model <- dim_res$model

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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

test_that("[DFML] - Deep LSTM autoencoder external", {
  forecast_params <- list()
  dim_params <- list()
  dim_params$method <- "deep_gru"
  dim_params$time_window <- 5
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- epochs

  dim_res <- dimensionalityReduction(X_train,components,"Autoencoder_Keras",dim_params)
  dim_params$model <- dim_res$model

  for (forecasting_method in M4_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "M4Methods",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
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
