context("Dimensionality Reduction/Increase")

library(ExtendedDFML)

#Set up - Data
X <- read.csv("testdata/Sigma4.ssv",sep=" ",header = TRUE)
splitting_point <- round(2*nrow(X)/3)
X_train <- X[1:splitting_point,]
X_train <- as.matrix(X_train)

#Set up - Parameters
k <- 3
time_window <- 5
epochs <- 10

#Set up - Precomputed results
PCA_dec_results <- dimensionalityReduction(X_train,k,family="PCA")
autoencoder_dec_results <- dimensionalityReduction(X_train,k,family="Autoencoder")
params <- list(method="base",epochs=epochs)
autoencoder_keras_base_results <- dimensionalityReduction(X_train,k,family="Autoencoder_Keras",params)
params <- list(method="deep",epochs=epochs)
autoencoder_keras_deep_results <- dimensionalityReduction(X_train,c(10,5,k),family="Autoencoder_Keras",params)
params <- list(method="lstm",time_window=time_window,epochs=epochs)
autoencoder_keras_lstm_results <- dimensionalityReduction(X_train,k,family="Autoencoder_Keras",params)
params <- list(method="deep_lstm",time_window=time_window,epochs=epochs)
autoencoder_keras_deep_lstm_results <- dimensionalityReduction(X_train,c(10,k),family="Autoencoder_Keras",params)
params <- list(method="gru",time_window=time_window,epochs=epochs)
autoencoder_keras_gru_results <- dimensionalityReduction(X_train,k,family="Autoencoder_Keras",params)
params <- list(method="deep_gru",time_window=time_window,epochs=epochs)
autoencoder_keras_deep_gru_results <- dimensionalityReduction(X_train,c(10,k),family="Autoencoder_Keras",params)

PCA_inc_results <- dimensionalityIncrease(PCA_dec_results$Z,family = "PCA",PCA_dec_results$model,PCA_dec_results$Z)
autoencoder_inc_results <- dimensionalityIncrease(autoencoder_dec_results$Z,family = "Autoencoder",autoencoder_dec_results$model)

autoencoder_inc_keras_base_results <- dimensionalityIncrease(autoencoder_keras_base_results$Z,family="Autoencoder_Keras",model=autoencoder_keras_base_results$model)
autoencoder_inc_keras_deep_results <- dimensionalityIncrease(autoencoder_keras_deep_results$Z,family="Autoencoder_Keras",model=autoencoder_keras_deep_results$model)
params <- list(method="lstm",time_window=time_window)
autoencoder_inc_keras_lstm_results <- dimensionalityIncrease(autoencoder_keras_lstm_results$Z,family="Autoencoder_Keras",model=autoencoder_keras_lstm_results$model,params = params)
params <- list(method="deep_lstm",time_window=time_window)
autoencoder_inc_keras_deep_lstm_results <- dimensionalityIncrease(autoencoder_keras_deep_lstm_results$Z,family="Autoencoder_Keras",model=autoencoder_keras_deep_lstm_results$model,params = params)
params <- list(method="gru",time_window=time_window)
autoencoder_inc_keras_gru_results <- dimensionalityIncrease(autoencoder_keras_gru_results$Z,family="Autoencoder_Keras",model=autoencoder_keras_gru_results$model,params = params)
params <- list(method="deep_gru",time_window=time_window)
autoencoder_inc_keras_deep_gru_results <- dimensionalityIncrease(autoencoder_keras_deep_gru_results$Z,family="Autoencoder_Keras",model=autoencoder_keras_deep_gru_results$model,params = params)

test_that("Dimensionality reduction fails when called with wrong method", {
  throws_error(dimensionalityReduction(X_train,k,dimensionality_method="anything"))
})

test_that("Dimensionality increase fails when called with wrong method", {
  throws_error(dimensionalityReduction(X_train,k,dimensionality_method="anything"))
})

test_that("[Decrease] PCA refactor correctly executed", {
  #Old implementation
  C <- cov(X_train)
  V <- t(eigen(C,TRUE)$vectors[,1:k])

  expect_equal(dim(PCA_dec_results$model$eigenvectors), dim(V))
  expect_true(sum(PCA_dec_results$model$eigenvectors - V) == 0)
})

test_that("[Decrease] PCA return has correct dimension", {
   expect_equal(dim(PCA_dec_results$Z), c(dim(X_train)[1],k))
   expect_true(is.numeric(PCA_dec_results$time_dim))
   expect_equal(dim(PCA_dec_results$model$eigenvectors), c(k,dim(X_train)[2]))
})

test_that("[Decrease] PCA works while passing an external model", {
  PCA_dec_model_results <- dimensionalityReduction(X_train,k,family="PCA",PCA_dec_results)
  expect_equal(dim(PCA_dec_model_results$Z), c(dim(X_train)[1],k))
  expect_true(is.numeric(PCA_dec_model_results$time_dim))
  expect_equal(dim(PCA_dec_model_results$model$eigenvectors), c(k,dim(X_train)[2]))
  expect_true(sum(PCA_dec_model_results$Z - PCA_dec_results$Z) == 0) #Components are equals
  expect_true(sum(PCA_dec_model_results$model$eigenvectors - PCA_dec_results$model$eigenvectors) == 0) # Matrix models are equals
})

test_that("[Decrease] Autoencoder return has correct dimension", {
  expect_equal(dim(autoencoder_dec_results$Z), c(dim(X_train)[1],k))
  expect_true(is.numeric(autoencoder_dec_results$time_dim))
  expect_equal(class(autoencoder_dec_results$model), "autoencoder")
})

test_that("[Decrease] Autoencoder Keras - Base return has correct dimension", {
  print("[INFO] - Autoencoder Keras Base")
  print(dim(autoencoder_keras_base_results$Z))
  print(head(autoencoder_keras_base_results$Z))
  expect_equal(dim(autoencoder_keras_base_results$Z), c(dim(X_train)[1],k))
  expect_true(is.numeric(autoencoder_keras_base_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - Deep return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep")
  print(dim(autoencoder_keras_deep_results$Z))
  print(head(autoencoder_keras_deep_results$Z))
  expect_equal(dim(autoencoder_keras_deep_results$Z), c(dim(X_train)[1],k))
  expect_true(is.numeric(autoencoder_keras_deep_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - LSTM return has correct dimension", {
  print("[INFO] - Autoencoder Keras LSTM")
  print(dim(autoencoder_keras_lstm_results$Z))
  print(head(autoencoder_keras_lstm_results$Z))
  expect_equal(dim(autoencoder_keras_lstm_results$Z), c(dim(X_train)[1]-time_window+1,k))
  expect_true(is.numeric(autoencoder_keras_lstm_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - Deep LSTM return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep LSTM")
  print(dim(autoencoder_keras_deep_lstm_results$Z))
  print(head(autoencoder_keras_deep_lstm_results$Z))
  expect_equal(dim(autoencoder_keras_deep_lstm_results$Z), c(dim(X_train)[1]-time_window+1,k))
  expect_true(is.numeric(autoencoder_keras_deep_lstm_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - GRU return has correct dimension", {
  print("[INFO] - Autoencoder Keras GRU")
  print(dim(autoencoder_keras_gru_results$Z))
  print(head(autoencoder_keras_gru_results$Z))
  expect_equal(dim(autoencoder_keras_gru_results$Z), c(dim(X_train)[1]-time_window+1,k))
  expect_true(is.numeric(autoencoder_keras_gru_results$time_dim))
})

test_that("[Decrease] Autoencoder Keras - Deep GRU return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep GRU")
  print(dim(autoencoder_keras_deep_gru_results$Z))
  print(head(autoencoder_keras_deep_gru_results$Z))
  expect_equal(dim(autoencoder_keras_deep_gru_results$Z), c(dim(X_train)[1]-time_window+1,k))
  expect_true(is.numeric(autoencoder_keras_deep_gru_results$time_dim))
})

test_that("[Increase] PCA return has correct dimension", {
  expect_equal(dim(PCA_inc_results$X_hat), dim(X_train))
  expect_true(is.numeric(PCA_inc_results$time_dim))
})

test_that("[Increase] Autoencoder return has correct dimension", {
  expect_equal(dim(autoencoder_inc_results$X_hat), dim(X_train))
  expect_true(is.numeric(autoencoder_inc_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Base return has correct dimension", {
  print("[INFO] - Autoencoder Keras Base")
  print(dim(autoencoder_inc_keras_base_results$X_hat))
  print(head(autoencoder_inc_keras_base_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_base_results$X_hat), c(dim(X_train)))
  expect_true(is.numeric(autoencoder_inc_keras_base_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Deep return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep")
  print(dim(autoencoder_inc_keras_deep_results$X_hat))
  print(head(autoencoder_inc_keras_deep_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_deep_results$X_hat), c(dim(X_train)))
  expect_true(is.numeric(autoencoder_inc_keras_deep_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - LSTM return has correct dimension", {
  print("[INFO] - Autoencoder Keras LSTM")
  print(dim(autoencoder_inc_keras_lstm_results$X_hat))
  print(head(autoencoder_inc_keras_lstm_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_lstm_results$X_hat), c(dim(X_train)))
  expect_true(is.numeric(autoencoder_inc_keras_lstm_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Deep LSTM return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep LSTM")
  print(dim(autoencoder_inc_keras_deep_lstm_results$X_hat))
  print(head(autoencoder_inc_keras_deep_lstm_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_deep_lstm_results$X_hat), c(dim(X_train)))
  expect_true(is.numeric(autoencoder_inc_keras_deep_lstm_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - GRU return has correct dimension", {
  print("[INFO] - Autoencoder Keras GRU")
  print(dim(autoencoder_inc_keras_gru_results$X_hat))
  print(head(autoencoder_inc_keras_gru_results$X_hat))
  expect_equal(dim(tensor3D2matrix(autoencoder_inc_keras_deep_gru_results$X_hat,shift = time_window)),
               c(dim(X_train)))
  expect_true(is.numeric(autoencoder_inc_keras_gru_results$time_dim))
})

test_that("[Increase] Autoencoder Keras - Deep GRU return has correct dimension", {
  print("[INFO] - Autoencoder Keras Deep GRU")
  print(dim(autoencoder_inc_keras_deep_gru_results$X_hat))
  print(head(autoencoder_inc_keras_deep_gru_results$X_hat))
  expect_equal(dim(autoencoder_inc_keras_deep_gru_results$X_hat), c(dim(X_train)))
  expect_true(is.numeric(autoencoder_inc_keras_deep_gru_results$time_dim))
})


# Teardown
#rm(PCA_dec_results)
#rm(autoencoder_dec_results)
#rm(PCA_inc_results)
#rm(autoencoder_inc_results)
