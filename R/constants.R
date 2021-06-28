# Dimenstionality reduction constants - Method categories and names
DIMENSIONALITY_METHODS <- c("PCA","Autoencoder","Autoencoder_Keras")
INCREMENTAL_DIMENSIONALITY_METHODS <- c("PCA","Autoencoder_Keras")
INCREMENTAL_PCA_METHODS <- c("incrpca","ccipca","ghapca","perturbationpca","secularpca","sgapca","snlpca","incrpca_block")
NO_EMBEDDING_METHODS <- c("base","base_regularized","deep")
REQUIRE_EMBEDDING_METHODS <- c("convolutional_1D","convolutional_2D","lstm","lstm_dense","deep_lstm","gru","deep_gru","lstm_convolutional_1D")
AUTOENCODER_METHODS <- c(NO_EMBEDDING_METHODS,REQUIRE_EMBEDDING_METHODS)

# Forecasting constants - Method categories and names
FORECAST_FAMILY <- c("multistepAhead","M4Methods","VAR","gradientBoosting")
MULTISTEPAHEAD_METHODS <- c("arima","direct","iter","lazydirect","lazyiter","rfdirect","rfiter","mimo","mimo.comb","mimo.acf","mimo.acf.lin","mimo.pls","mimo.lin.pls")
M4_METHODS <- c("ES","holt_winters","holt_winters_damped","theta","combined")
GRADIENT_BOOSTING_METHODS <- c('xgboost',"lightgbm")
FORECAST_METHODS <- c(MULTISTEPAHEAD_METHODS,M4_METHODS,GRADIENT_BOOSTING_METHODS)
