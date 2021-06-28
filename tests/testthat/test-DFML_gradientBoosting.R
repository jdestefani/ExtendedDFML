context("DFML - Test gradient boosting")

library(keras)
library(dplyr)
library(forecast)
library(parallel)
library(xgboost)
library(lightgbm)
library(MEMTS)

#Set up - Data
X <- read.csv("testdata/Sigma4.ssv",sep=" ",header = TRUE)
X <- as.matrix(X)
splitting_point <- round(2*nrow(X)/3)
X_train <- scale(X[1:splitting_point,])

components <- 3
horizon <- 5
time_window <- horizon

ss_results.df <- data.frame(DimensionalityMethod=character(),ForecastingMethod=character(),Dataset=character(),Horizon=numeric(),Columns=numeric(),Time=numeric(),MSE=numeric(),Samples=numeric(),stringsAsFactors = FALSE)

# If there is an error with the Arima package, check
#detach("package:dse", unload=TRUE)

test_that("[DFML] - PCA + gradientBoosting methods direct + 8 threads", {
  forecast_params <- list()
  forecast_params$m <- 3
  forecast_params$forecasting_params <- list()
  forecast_params$forecasting_params$n_threads <- 2
  multistep_method <- "direct"

  for (forecasting_method in ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "PCA",
                                  "gradientBoosting",
                                  dimensionality_parameters = NULL,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="PCA",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})



test_that("[DFML] - PCA + gradientBoosting methods direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  multistep_method <- "direct"

  for (forecasting_method in ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "PCA",
                                  "gradientBoosting",
                                  dimensionality_parameters = NULL,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="PCA",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - PCA + gradientBoosting methods recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "PCA",
                                  "gradientBoosting",
                                  dimensionality_parameters = NULL,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="PCA",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - Base autoencoder + gradientBoosting methods direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "base"
  dim_params$epochs <- 10
  multistep_method <- "direct"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Base Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - Base autoencoder + gradientBoosting methods recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "base"
  dim_params$epochs <- 10
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Base Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep autoencoder + gradientBoosting methods direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "deep"
  dim_params$deep_layers <- c(10,5,3)
  dim_params$epochs <- 10
  multistep_method <- "direct"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - Deep autoencoder + gradientBoosting methods recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "deep"
  dim_params$deep_layers <- c(10,5,3)
  dim_params$epochs <- 10
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - LSTM autoencoder + gradientBoosting methods direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "lstm"
  dim_params$time_window <- time_window
  dim_params$epochs <- 10
  multistep_method <- "direct"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="LSTM Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - LSTM autoencoder + gradientBoosting methods recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "lstm"
  dim_params$time_window <- time_window
  dim_params$epochs <- 10
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$method <- forecasting_method
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="LSTM Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep LSTM autoencoder + gradientBoosting methods direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "deep_lstm"
  dim_params$time_window <- time_window
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- 10
  multistep_method <- "direct"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep LSTM Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep LSTM autoencoder + gradientBoosting methods recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "deep_lstm"
  dim_params$time_window <- time_window
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- 10
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep LSTM Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - GRU autoencoder + gradientBoosting methods direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "gru"
  dim_params$time_window <- time_window
  dim_params$epochs <- 10
  multistep_method <- "direct"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="GRU Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})

test_that("[DFML] - GRU autoencoder + gradientBoosting methods recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "gru"
  dim_params$time_window <- time_window
  dim_params$epochs <- 10
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="GRU Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep GRU autoencoder + gradientBoosting direct", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "deep_gru"
  dim_params$time_window <- time_window
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- 10
  multistep_method <- "direct"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep GRU Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})


test_that("[DFML] - Deep GRU autoencoder + gradientBoosting recursive", {
  forecast_params <- list()
  forecast_params$m <- 3
  dim_params <- list()
  dim_params$method <- "deep_gru"
  dim_params$time_window <- time_window
  dim_params$deep_layers <- c(10,3)
  dim_params$epochs <- 10
  multistep_method <- "recursive"

  for (forecasting_method in  ExtendedDFML::GRADIENT_BOOSTING_METHODS) {
    print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
    forecast_params$forecasting_method <- forecasting_method
    forecast_params$multistep_method <- multistep_method
    results <- ExtendedDFML::DFML(X_train,
                                  "Autoencoder_Keras",
                                  "gradientBoosting",
                                  dimensionality_parameters = dim_params,
                                  forecast_params,
                                  components,
                                  horizon)
    MSE_forecast <- MMSE(X[(splitting_point+1):(splitting_point+horizon),],results$X_hat)
    ss_results.df <- bind_rows(ss_results.df,
                               data.frame(DimensionalityMethod="Deep GRU Autoencoder",
                                          ForecastingMethod=paste(forecasting_method,multistep_method,sep="_"),
                                          Dataset="Sigma 4",
                                          Horizon=as.numeric(horizon),
                                          Columns=as.numeric(ncol(X)),
                                          Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                          MSE=as.numeric(MSE_forecast$mean),
                                          Samples=as.numeric(splitting_point)))
  }
  print(ss_results.df)
})
