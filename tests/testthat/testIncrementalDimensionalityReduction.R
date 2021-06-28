context("Incremental Dimensionality Reduction")
library(onlinePCA)

#Set up - Data
X <- EuStockMarkets
X <- as.matrix(X)
splitting_point <- round(2*nrow(X)/3)
X_train <- X[1:splitting_point,]
X_update <- X[(splitting_point+1):(splitting_point+100),]

#Set up - Parameters
k <- 3
h <- 5
time_window <- 5
init_sample_size = 0.5
epochs <- 10

#Set up - Precomputed results
PCA_dec_results <- dimensionalityReduction(X_train,k,family="PCA")
params <- list(pca_type="incrpca")
PCA_incremental_dec_results <- incrementalDimensionalityReduction(X_train,init_sample_size,k,family="PCA",params=params)
params <- list(pca_type="incrpca",model=PCA_dec_results$model)
PCA_incremental_update_dec_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                               X_update,
                                                                               k=k,
                                                                               family="PCA",
                                                                               params=params)

#autoencoder_dec_results <- dimensionalityReduction(X_train,k,family="Autoencoder")
params <- list(method="base",epochs=epochs)
autoencoder_keras_base_results <- dimensionalityReduction(X_train,k,family="Autoencoder_Keras",params)
params$model <- autoencoder_keras_base_results$model
autoencoder_keras_incremental_update_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                                         X_update,
                                                                                         k=k,
                                                                                         family="Autoencoder_Keras",
                                                                                         params=params)
# TO DO - Implement other tests
params <- list(method="deep",epochs=epochs)
autoencoder_keras_deep_results <- dimensionalityReduction(X_train,c(10,5,k),family="Autoencoder_Keras",params)
params$model <- autoencoder_keras_deep_results$model
autoencoder_keras_incremental_update_deep_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                                         X_update,
                                                                                         k=k,
                                                                                         family="Autoencoder_Keras",
                                                                                         params=params)
params <- list(method="lstm",time_window=time_window,epochs=epochs)
autoencoder_keras_lstm_results <- dimensionalityReduction(X_train,k,family="Autoencoder_Keras",params)
params$model <- autoencoder_keras_lstm_results$model
autoencoder_keras_incremental_update_lstm_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                                              X_update,
                                                                                              k=k,
                                                                                              family="Autoencoder_Keras",
                                                                                              params=params)
params <- list(method="deep_lstm",time_window=time_window,epochs=epochs)
autoencoder_keras_deep_lstm_results <- dimensionalityReduction(X_train,c(10,k),family="Autoencoder_Keras",params)
params$model <- autoencoder_keras_deep_lstm_results$model
autoencoder_keras_incremental_update_deep_lstm_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                                              X_update,
                                                                                              k=k,
                                                                                              family="Autoencoder_Keras",
                                                                                              params=params)
params <- list(method="gru",time_window=time_window,epochs=epochs)
autoencoder_keras_gru_results <- dimensionalityReduction(X_train,k,family="Autoencoder_Keras",params)
params$model <- autoencoder_keras_gru_results$model
autoencoder_keras_incremental_update_gru_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                                              X_update,
                                                                                              k=k,
                                                                                              family="Autoencoder_Keras",
                                                                                              params=params)
params <- list(method="deep_gru",time_window=time_window,epochs=epochs)
autoencoder_keras_deep_gru_results <- dimensionalityReduction(X_train,c(10,k),family="Autoencoder_Keras",params)
params$model <- autoencoder_keras_deep_gru_results$model
autoencoder_keras_incremental_update_deep_gru_results <- incrementalDimensionalityReductionUpdate(X_train,
                                                                                              X_update,
                                                                                              k=k,
                                                                                              family="Autoencoder_Keras",
                                                                                              params=params)

PCA_inc_results <- dimensionalityIncrease(PCA_incremental_dec_results$Z,
                                          family = "PCA",
                                          PCA_dec_results$model,
                                          PCA_incremental_dec_results$Z)

PCA_inc_update_results <- dimensionalityIncrease(PCA_incremental_update_dec_results$Z_update,
                                                 family = "PCA",
                                                 PCA_incremental_update_dec_results$model,
                                                 PCA_incremental_update_dec_results$Z_update)
# autoencoder_inc_results <- dimensionalityIncrease(autoencoder_dec_results$Z,method = "Autoencoder",autoencoder_dec_results$model)
#
autoencoder_inc_keras_base_results <- dimensionalityIncrease(autoencoder_keras_incremental_update_results$Z_update,
                                                             family ="Autoencoder_Keras",
                                                             autoencoder_keras_incremental_update_results$model)

autoencoder_inc_keras_deep_results <- dimensionalityIncrease(autoencoder_keras_incremental_update_deep_results$Z_update,
                                                             family ="Autoencoder_Keras",
                                                             autoencoder_keras_incremental_update_deep_results$model)

params <- list(method="lstm",time_window=time_window)
autoencoder_inc_keras_lstm_results <- dimensionalityIncrease(autoencoder_keras_incremental_update_lstm_results$Z_update,
                                                             family ="Autoencoder_Keras",
                                                             autoencoder_keras_incremental_update_lstm_results$model,
                                                             params=params)

params <- list(method="deep_lstm",time_window=time_window)
autoencoder_inc_keras_deep_lstm_results <- dimensionalityIncrease(autoencoder_keras_incremental_update_deep_lstm_results$Z_update,
                                                                  family ="Autoencoder_Keras",
                                                                  autoencoder_keras_incremental_update_deep_lstm_results$model,
                                                                  params=params)

params <- list(method="gru",time_window=time_window)
autoencoder_inc_keras_gru_results <- dimensionalityIncrease(autoencoder_keras_incremental_update_gru_results$Z_update,
                                                            family ="Autoencoder_Keras",
                                                            autoencoder_keras_incremental_update_gru_results$model,
                                                            params=params)

params <- list(method="deep_gru",time_window=time_window)
autoencoder_inc_keras_deep_gru_results <- dimensionalityIncrease(autoencoder_keras_incremental_update_deep_gru_results$Z_update,
                                                                 family ="Autoencoder_Keras",
                                                                 autoencoder_keras_incremental_update_deep_gru_results$model,
                                                                 params = params)

test_that("Dimensionality reduction fails when called with wrong method", {
  throws_error(dimensionalityReduction(X_train,k,dimensionality_method="anything"))
})

test_that("Dimensionality increase fails when called with wrong method", {
  throws_error(dimensionalityReduction(X_train,k,dimensionality_method="anything"))
})


test_that("[Decrease] Incremental PCA return has correct dimension", {
   expect_equal(dim(PCA_incremental_dec_results$Z), c(dim(X_train)[1],k))
   expect_true(is.numeric(PCA_incremental_dec_results$time_dim))
   expect_equal(dim(PCA_incremental_dec_results$model$eigenvectors), c(k,dim(X_train)[2]))
})

test_that("[Decrease] Incremental PCA works while passing an external model", {
  PCA_dec_model_results <- incrementalDimensionalityReduction(X_train,init_sample_size,k,family="PCA",params=list(pca_type="incrpca",model=PCA_incremental_dec_results$model))
  expect_equal(dim(PCA_dec_model_results$Z), c(dim(X_train)[1],k))
  expect_true(is.numeric(PCA_dec_model_results$time_dim))
  expect_equal(dim(PCA_dec_model_results$model$eigenvectors), c(k,dim(X_train)[2]))
  expect_true(sum(PCA_dec_model_results$Z - PCA_dec_results$Z) == 0) #Components are equals
  expect_true(sum(PCA_dec_model_results$model$eigenvectors - PCA_dec_results$model$eigenvectors) == 0) # Matrix models are equals
})

test_that("[Decrease] Incremental PCA update implementation", {
    expect_equal(length(PCA_incremental_update_dec_results$model$eigenvalues), k)
    expect_equal(dim(PCA_incremental_update_dec_results$model$eigenvectors), c(k,dim(X_train)[2]))
    expect_true(is.numeric(PCA_incremental_update_dec_results$time_dim))
    expect_equal(dim(PCA_incremental_update_dec_results$Z_update), c(dim(X_update)[1],k))
})

test_that("[Decrease] Incremental PCA core function works while passing external data", {

  for(pca_type in INCREMENTAL_PCA_METHODS){
    print(paste("[INFO] - Launching", pca_type))
    X <- as.matrix(X)
    pca <- list(values=PCA_incremental_dec_results$model$eigenvalues,vectors=t(PCA_incremental_dec_results$model$eigenvectors))
    xbar <- apply(as.matrix(X_train), 2, mean)

    for(i in (splitting_point+1):(splitting_point+h)){
      iterative_pca_results <- iterativePCASingleStep(X[i,],i,pca_type,xbar,pca,NA,k)
      xbar <- iterative_pca_results$xbar
      pca <- iterative_pca_results$pca
    }
    expect_equal(length(pca$values), k)
    expect_equal(dim(t(pca$vectors)), c(k,dim(X_train)[2]))
  }

})

# test_that("[Decrease] Autoencoder return has correct dimension", {
#   expect_equal(dim(autoencoder_dec_results$Z), c(dim(X_train)[1],k))
#   expect_true(is.numeric(autoencoder_dec_results$time_dim))
#   expect_equal(class(autoencoder_dec_results$model), "autoencoder")
# })

test_that("[Decrease] Autoencoder Keras - Base update return has correct dimension", {
  print("[INFO] - Autoencoder Keras Base")
  print(dim(autoencoder_keras_incremental_update_results$Z_update))
  print(head(autoencoder_keras_incremental_update_results$Z_update))
  expect_equal(dim(autoencoder_keras_incremental_update_results$Z_update), c(dim(X_update)[1],k))
  expect_true(is.numeric(autoencoder_keras_incremental_update_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - Deep update return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep")
  print(dim(autoencoder_keras_incremental_update_deep_results$Z_update))
  print(head(autoencoder_keras_incremental_update_deep_results$Z_update))
  expect_equal(dim(autoencoder_keras_incremental_update_deep_results$Z_update), c(dim(X_update)[1],k))
  expect_true(is.numeric(autoencoder_keras_incremental_update_deep_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - LSTM update return has correct dimension", {
  print("[INFO] - Autoencoder Keras LSTM")
  print(dim(autoencoder_keras_incremental_update_lstm_results$Z_update))
  print(head(autoencoder_keras_incremental_update_lstm_results$Z_update))
  expect_equal(dim(autoencoder_keras_incremental_update_lstm_results$Z_update), c(dim(X_update)[1],k))
  expect_true(is.numeric(autoencoder_keras_incremental_update_lstm_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - Deep LSTM update return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep LSTM")
  print(dim(autoencoder_keras_incremental_update_deep_lstm_results$Z_update))
  print(head(autoencoder_keras_incremental_update_deep_lstm_results$Z_update))
  expect_equal(dim(autoencoder_keras_incremental_update_deep_lstm_results$Z_update), c(dim(X_update)[1],k))
  expect_true(is.numeric(autoencoder_keras_incremental_update_deep_lstm_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - GRU update return has correct dimension", {
  print("[INFO] - Autoencoder Keras GRU")
  print(dim(autoencoder_keras_incremental_update_gru_results$Z_update))
  print(head(autoencoder_keras_incremental_update_gru_results$Z_update))
  expect_equal(dim(autoencoder_keras_incremental_update_gru_results$Z_update), c(dim(X_update)[1],k))
  expect_true(is.numeric(autoencoder_keras_incremental_update_gru_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - Deep GRU update return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep GRU")
  print(dim(autoencoder_keras_incremental_update_deep_gru_results$Z_update))
  print(head(autoencoder_keras_incremental_update_deep_gru_results$Z_update))
  expect_equal(dim(autoencoder_keras_incremental_update_deep_gru_results$Z_update), c(dim(X_update)[1],k))
  expect_true(is.numeric(autoencoder_keras_incremental_update_gru_results$time_dim))
})


test_that("[Increase] PCA return has correct dimension", {
  expect_equal(dim(PCA_inc_results$X_hat), dim(X_train))
  expect_true(is.numeric(PCA_inc_results$time_dim))
})

# test_that("[Increase] Autoencoder return has correct dimension", {
#   expect_equal(dim(autoencoder_inc_results$X_hat), dim(X_train))
#   expect_true(is.numeric(autoencoder_inc_results$time_dim))
# })
#
test_that("[Increase] Autoencoder Keras - Base return has correct dimension", {
  print("[INFO] - Autoencoder Keras Base")
  print(dim(autoencoder_inc_keras_base_results$X_hat))
  print(head(autoencoder_inc_keras_base_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_base_results$X_hat), c(dim(X_update)))
  expect_true(is.numeric(autoencoder_inc_keras_base_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Deep return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep")
  print(dim(autoencoder_inc_keras_deep_results$X_hat))
  print(head(autoencoder_inc_keras_deep_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_deep_results$X_hat), c(dim(X_update)))
  expect_true(is.numeric(autoencoder_inc_keras_deep_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - LSTM return has correct dimension", {
  print("[INFO] - Autoencoder Keras LSTM")
  print(dim(autoencoder_inc_keras_lstm_results$X_hat))
  print(head(autoencoder_inc_keras_lstm_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_lstm_results$X_hat), c(dim(X_update)))
  expect_true(is.numeric(autoencoder_inc_keras_lstm_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Deep LSTM return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep LSTM")
  print(dim(autoencoder_inc_keras_deep_lstm_results$X_hat))
  print(head(autoencoder_inc_keras_deep_lstm_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_deep_lstm_results$X_hat), c(dim(X_update)))
  expect_true(is.numeric(autoencoder_inc_keras_deep_lstm_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - GRU return has correct dimension", {
  print("[INFO] - Autoencoder Keras GRU")
  print(dim(autoencoder_inc_keras_gru_results$X_hat))
  print(head(autoencoder_inc_keras_gru_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_gru_results$X_hat),dim(X_update))
  expect_true(is.numeric(autoencoder_inc_keras_gru_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Deep GRU return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep GRU")
  print(dim(autoencoder_inc_keras_deep_gru_results$X_hat))
  print(head(autoencoder_inc_keras_deep_gru_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_deep_gru_results$X_hat), c(dim(X_update)))
  expect_true(is.numeric(autoencoder_inc_keras_deep_gru_results$time_dim))
})


# Teardown
#rm(PCA_dec_results)
#rm(autoencoder_dec_results)
#rm(PCA_inc_results)
#rm(autoencoder_inc_results)
