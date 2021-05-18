context("Autoencoder tests")

library(keras)

#Set up - Data
X <- EuStockMarkets
splitting_point <- round(2*nrow(X)/3)

latent_dim <- 3
time_window <- 5
epochs <- 10

X_train <- scale(X[1:splitting_point,])
X_train_windowed <- matrix2Tensor3D(X_train,time_window,time_window)

test_that("[Autoencoder] - Error with unexisting method", {
  throws_error(dimensionalityReduction(X_train,method="anything"))
})

test_that("[Autoencoder] - Base autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = latent_dim, method = "base", epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train),latent_dim))
  expect_equal(dim(X_train),dim(X_train_hat))
})

test_that("[Autoencoder] - Base regularized autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = latent_dim, method = "base_regularized", epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train),latent_dim))
  expect_equal(dim(X_train),dim(X_train_hat))
})


# # test_that("[Autoencoder] - Convolutional 1D", {
# #   # Due to max pooling the time window must have an even size
# #   autoencoder_models <- autoencoder_keras(X_train,method = "convolutional_1D",epochs = epochs,time_window = 4)
# #   Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
# #   X_train_hat <- predict(autoencoder_models$decoder, Z_train)
# #   print(dim(Z_train))
# #   print(head(Z_train))
# #   #expect_equal(dim(Z_train),c(nrow(X_train),latent_dim))
# #   #expect_equal(dim(X_train_windowed),dim(X_train_hat))
# # })
# #
# # test_that("[Autoencoder] - Convolutional 2D", {
# #   # Due to max pooling the time window must have an even size
# #   autoencoder_models <- autoencoder_keras(X_train,method = "convolutional_2D",epochs = epochs,time_window = 4)
# #   Z_train <- predict(autoencoder_models$encoder,array(X_train_windowed,dim=c(dim(X_train_windowed),1)))
# #   print(dim(Z_train))
# #   print(head(Z_train))
# # })
# #
#
test_that("[Autoencoder] - Deep autoencoder", {
   autoencoder_models <- autoencoder_keras(X_train, latent_dim = c(10,5,latent_dim), method = "deep",epochs = epochs)
   Z_train <- predict(autoencoder_models$encoder, X_train)
   X_train_hat <- predict(autoencoder_models$decoder, Z_train)
   print(dim(Z_train))
   print(head(Z_train))
   expect_equal(dim(Z_train),c(nrow(X_train),latent_dim))
   expect_equal(dim(X_train),dim(X_train_hat))
})



test_that("[Autoencoder] - LSTM autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = latent_dim, time_window = 5,method = "lstm",epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train_windowed),time_window,latent_dim))
  expect_equal(dim(X_train_windowed),dim(X_train_hat))
})

test_that("[Autoencoder] - LSTM Dense autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = latent_dim, time_window = 5,method = "lstm_dense",epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train_windowed),time_window,latent_dim))
  expect_equal(dim(X_train_windowed),dim(X_train_hat))
})


test_that("[Autoencoder] - Deep LSTM autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = c(10,latent_dim), time_window = time_window, method = "deep_lstm", epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train_windowed),time_window,latent_dim))
  expect_equal(dim(X_train_windowed),dim(X_train_hat))
})

test_that("[Autoencoder] - GRU autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = latent_dim, time_window = time_window, method = "gru", epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train_windowed),time_window,latent_dim))
  expect_equal(dim(X_train_windowed),dim(X_train_hat))
})

test_that("[Autoencoder] - Deep GRU autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = c(10,latent_dim), time_window = time_window, method = "deep_gru", epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train_windowed),time_window,latent_dim))
  expect_equal(dim(X_train_windowed),dim(X_train_hat))
})

test_that("[Autoencoder] - LSTM Convlolutional 1D autoencoder", {
  autoencoder_models <- autoencoder_keras(X_train, latent_dim = c(32,32,latent_dim), time_window = time_window, method = "lstm_convolutional_1D", epochs = epochs)
  Z_train <- predict(autoencoder_models$encoder, X_train_windowed)
  X_train_hat <- predict(autoencoder_models$decoder, Z_train)
  print(dim(Z_train))
  print(head(Z_train))
  expect_equal(dim(Z_train),c(nrow(X_train_windowed),time_window,latent_dim))
  expect_equal(dim(X_train_windowed),dim(X_train_hat))
})



