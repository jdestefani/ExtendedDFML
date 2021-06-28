X <- EuStockMarkets
time_window <- 5
n_samples <- 100
X_windowed <- matrix2Tensor3D(X,time_window,time_window)

test_that("[Embedding] - Test time window and shift combinations", {
  for(time_window in 2:5){ # For time window = 1 there is no embedding needed
    for(shift in 1:time_window){ # The shift must be smaller than time window otherwise values will be lost
      original_matrix <- as.matrix(X[1:n_samples,])
      reconstructed_matrix <- tensor3D2matrix(matrix2Tensor3D(as.matrix(X[1:n_samples,]),time_window,shift),shift)
      print(c(time_window,shift))
      print(dim(original_matrix))
      print(dim(reconstructed_matrix))
    }
  }
})

test_that("[Embedding] - Test time window and shift combinations", {
  for(time_window in 2:5){ # For time window = 1 there is no embedding needed
      original_matrix <- as.matrix(X[1:n_samples,])
      reconstructed_matrix <- tensor3D2matrix(matrix2Tensor3D(as.matrix(X[1:n_samples,]),time_window,time_window),time_window)
      print(c(time_window,time_window))
      print(dim(original_matrix))
      print(dim(reconstructed_matrix))
      print(head(original_matrix))
      print(head(reconstructed_matrix))
      print(tail(original_matrix))
      print(tail(reconstructed_matrix))
    }
})
