#' autoencoder_keras
#' 
#' Wrapper function for all the different keras autoencoder implementations
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps - Numeric
#' @param method - Autoencoder type (from AUTOENCODER_METHODS list) - String
#' @param latent_dim - Number of latent dimensions - Numeric
#' @param time_window - Size of time window (in time steps) - Numeric
#' @param epochs - Number of epochs required for training - Numeric
#' @param batch_size - Batch size required for training - Numeric
#' @param optimizer_params - Optimizer parameters for keras fit function
#'                           \itemize{
#'                           \item{\code{loss}: }{Loss function used for optimization (among those defined by Keras) - String}
#'                            \item{\code{optimizer}: }{Optimizer function used for optimization (among those defined by Keras) - String}
#'                           }  
#' 
#' @param embedding_params - Paramaters to control the embedding process
#'                           \itemize{
#'                           \item{\code{padNA}: }{Value used to indicate whether incomplete tensors should be dropped or filled with NAs - Boolean}
#'                            \item{\code{shift}: }{Number of time step that be skipped in order to determine the start of the following tensor - Numeric}
#'                           }
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         \item{\code{time_window}: }{Time window used for the embedding (to fit the autoencoder) - Numeric scalar}
#'         \item{\code{train_history}: }{Object containing statistics on the training history - Keras object}
#'         }
#'
autoencoder_keras <- function(X,
                              method = AUTOENCODER_METHODS,
                              latent_dim=3,
                              time_window=5,
                              epochs=200,
                              batch_size=32,
                              optimizer_params=list(loss = "mean_squared_error",
                                                    optimizer = "adam"),
                              embedding_params=list(padNA=F,
                                                    shift=time_window)){

  method <- match.arg(method)
  if(is.null(optimizer_params)){
    stop("[ERROR] - Missing optimizer parameters")
  }
  if(missing(latent_dim)){
    stop("[ERROR] - Missing latent dimension parameter")
  }
  if(missing(epochs)){
    stop("[ERROR] - Missing epochs parameter")
  }
  if(missing(time_window) & method %in% REQUIRE_EMBEDDING_METHODS){
    stop("[ERROR] - An embedding method requires the time window")
  }

  # Recurrent based methods train on non overlapping windows, according to:
  # https://datascience.stackexchange.com/questions/27628/sliding-window-leads-to-overfitting-in-lstm
  switch(method,
         base={model <- BaseAutoencoder(X,latent_dim)},
         base_regularized={ model <- BaseRegularizedAutoencoder(X,latent_dim)},
         deep={ model <- DeepAutoencoder(X,latent_dim = latent_dim)},
         convolutional_1D={X <- matrix2Tensor3D(X,time_window = time_window)
                           model <- Convolutional1DAutoencoder(X,filters = c(32,32,16))},
         convolutional_2D={X <- matrix2Tensor3D(X,time_window = time_window)
                           model <- Convolutional2DAutoencoder(X,filters = c(32,32))
                           X <- array(X,dim=c(dim(X),1))},
         lstm={X <- matrix2Tensor3D(X,
                                    time_window = time_window,
                                    shift = embedding_params$shift,
                                    padNA = embedding_params$padNA)
              model <- LSTMAutoencoder(X,latent_dim)},
         lstm_dense={X <- matrix2Tensor3D(X,
                                          time_window = time_window,
                                          shift = embedding_params$shift,
                                          padNA = embedding_params$padNA)
              model <- LSTMDenseAutoencoder(X,latent_dim)},
         deep_lstm={X <- matrix2Tensor3D(X,
                                         time_window = time_window,
                                         shift = embedding_params$shift,
                                         padNA = embedding_params$padNA)
                   model <- DeepLSTMAutoencoder(X,latent_dim=latent_dim)},
         gru={X <- matrix2Tensor3D(X,
                                   time_window = time_window,
                                   shift = embedding_params$shift,
                                   padNA = embedding_params$padNA)
              model <- GRUAutoencoder(X,latent_dim)},
         deep_gru={X <- matrix2Tensor3D(X,
                                        time_window = time_window,
                                        shift = embedding_params$shift,
                                        padNA = embedding_params$padNA)
                   model <- DeepGRUAutoencoder(X,latent_dim=latent_dim)},
         lstm_convolutional_1D={X <- matrix2Tensor3D(X,
                                                     time_window = time_window,
                                                     shift = embedding_params$shift,
                                                     padNA = embedding_params$padNA)
                                model <- LSTMConvolutional1DAutoencoder(X,latent_dim = latent_dim)}
         )

  summary(model$autoencoder)

  model$autoencoder %>% compile(
    loss = optimizer_params$loss,
    optimizer = optimizer_params$optimizer,
    metrics = list(mae=metric_mean_absolute_error,
                   mape=metric_mean_absolute_percentage_error,
                   msle=metric_mean_squared_logarithmic_error)
  )

  early_stopping <- callback_early_stopping(patience = 5)

  callbacks_list <- list(early_stopping)
  
  train_history <- model$autoencoder %>% fit(
    x=X,
    y=X,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = list(X, X),
    #callbacks = callbacks_list,
    view_metrics = FALSE)

  return(list(autoencoder=model$autoencoder,encoder=model$encoder,decoder=model$decoder,time_window=time_window,train_history=train_history))
}

#' incremental_autoencoder_keras
#' 
#' Wrapper function for an incremental fitting of an autoencoder model
#'
#' @param X_update - nxN matrix containing the N time series as columns, each one of length n time steps used to perform the model update
#' @param method - Autoencoder type (from AUTOENCODER_METHODS list)
#' @param model - Pre-trained autoencoder model
#' @param time_window - Size of time window (in time steps)
#' @param epochs - Number of epochs required for training
#' @param batch_size - Batch size required for training
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         \item{\code{time_window}: }{Time window used for the embedding (to fit the autoencoder) - Numeric scalar}
#'         \item{\code{train_history}: }{Object containing statistics on the training history - Keras object}
#'         }
#'
incremental_autoencoder_keras_update <- function(X_update,
                                          method,
                                          model,
                                          time_window=5,
                                          epochs=200,
                                          batch_size=32){

  if(missing(epochs)){
    stop("[ERROR] - Missing epochs parameter")
  }
  if(missing(model)){
    stop("[ERROR] - A pre-trained model is required for incremental learning")
  }
  if(missing(time_window) & method %in% REQUIRE_EMBEDDING_METHODS){
    stop("[ERROR] - An embedding method requires the time window")
  }

  summary(model$autoencoder)

  # Reshape data according to the type of autoencoder (embed in 3D if required)
  if(method %in% REQUIRE_EMBEDDING_METHODS){
    X_update <- matrix2Tensor3D(X_update,time_window = time_window)
  }

  early_stopping <- callback_early_stopping(patience = 5)

  # From Learning.AI - Time series course
  #learning_rate_scheduler <- callback_learning_rate_scheduler(function(epoch){10^-8 * 10^(epoch/20)})

  if(batch_size == 1){
    for(i in 1:dim(X_update)[1]){
        model$autoencoder %>% fit(
        x=X_update,
        y=X_update,
        epochs = 1,
        batch_size = batch_size,
        callbacks = list(early_stopping),
        view_metrics = FALSE)
        #callbacks = list(checkpoint, early_stopping,learning_rate_scheduler))
    }
  }
  else{
    model$autoencoder %>% fit(
      x=X_update,
      y=X_update,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = list(early_stopping),
      view_metrics = FALSE)
    #callbacks = list(checkpoint, early_stopping,learning_rate_scheduler))
  }

  return(list(autoencoder=model$autoencoder,encoder=model$encoder,decoder=model$decoder,time_window=time_window))
}

#' BaseAutoencoder
#' 
#' Auxiliary function defining a basic autoencoder model (1 dense hidden layer)
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of latent dimensions (equivalent to the size of hidden layer) - numeric scalar
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
BaseAutoencoder <- function(X,latent_dim){
  input_tensor <- layer_input(shape = c(ncol(X)))
  encoded <- input_tensor %>% layer_dense(units = latent_dim, activation = "relu")
  decoded <- encoded %>% layer_dense(units = c(ncol(X)), activation = "sigmoid")

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded)

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(latent_dim))

  # retrieve the last layer of the autoencoder model
  decoder_layer <- utils::tail(autoencoder$layers,1)[[1]]

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = encoded_input %>% decoder_layer)
  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' BaseRegularizedAutoencoder
#' 
#' Auxiliary function defining a basic autoencoder model (1 dense hidden layer) with regularization
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of latent dimensions (equivalent to the size of hidden layer) - numeric scalar
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
BaseRegularizedAutoencoder <- function(X,latent_dim){
  input_tensor <- layer_input(shape = c(ncol(X)))
  encoded <- input_tensor %>% layer_dense(units = latent_dim, activation = "relu", activity_regularizer=regularizer_l1(10e-5))
  decoded <- encoded %>% layer_dense(units = c(ncol(X)), activation = "sigmoid")

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded)

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(latent_dim))

  # retrieve the last layer of the autoencoder model
  decoder_layer <- utils::tail(autoencoder$layers,1)[[1]]

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = encoded_input %>% decoder_layer)
  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' DeepAutoencoder
#' 
#' Auxiliary function defining a deep autoencoder model (3 dense hidden layer)
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of latent dimensions (equivalent to the sizes of hidden layers) - numeric vector
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
DeepAutoencoder <- function(X,latent_dim=c(10,5,2)){
  input_tensor <- layer_input(shape = c(ncol(X)))
  encoded <- input_tensor %>%
              layer_dense(units = latent_dim[1], activation = "relu") %>%
              layer_dense(units = latent_dim[2], activation = "relu") %>%
              layer_dense(units = latent_dim[3], activation = "relu")

  decoded <- encoded %>%
             layer_dense(units = latent_dim[2], activation = "relu") %>%
             layer_dense(units = latent_dim[1], activation = "relu") %>%
             layer_dense(units = c(ncol(X)), activation = "sigmoid")

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded)

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(latent_dim[3]))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,3)
  decoder_layers <- encoded_input %>%
                    (decoder_layers_list[[1]]) %>%
                    (decoder_layers_list[[2]]) %>%
                    (decoder_layers_list[[3]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' ConvolutionalAutoencoder
#' 
#' Auxiliary function defining a 1D convolutional autoencoder model (1 1DConv + 2x 1DConv+MaxPool)
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param filters - Size of the the filters in the different layers - numeric vector
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
Convolutional1DAutoencoder <- function(X,filters=c(32,32,16)){
  # https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
  # https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))

  encoded <- input_tensor %>%
    layer_conv_1d(filters=filters[1], kernel_size=3, activation='relu', padding='same') %>%
    layer_conv_1d(filters=filters[2], kernel_size=3, activation='relu', padding='same') %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters=filters[3], kernel_size=3, activation='relu', padding='same') %>%
    layer_max_pooling_1d(pool_size = 2)

  decoded <- encoded %>%
    layer_conv_1d(filters=filters[3], kernel_size=3, activation='relu', padding='same') %>%
    layer_upsampling_1d(size=2) %>%
    layer_conv_1d(filters=filters[2], kernel_size=3, activation='relu', padding='same') %>%
    layer_conv_1d(filters=filters[1], kernel_size=3, activation='relu', padding='same') %>%
    layer_upsampling_1d(size=2) %>%
    layer_conv_1d(filters=input_dim, kernel_size=3, activation='relu', padding='same')

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded)

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(1,filters[3]))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,6)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]]) %>%
    (decoder_layers_list[[3]]) %>%
    (decoder_layers_list[[4]]) %>%
    (decoder_layers_list[[5]]) %>%
    (decoder_layers_list[[6]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' Convolutional2DAutoencoder
#' 
#' Auxiliary function defining a 2D convolutional autoencoder model (2 x 2DConv+MaxPool)
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param filters - Size of the the filters in the different layers - numeric vector
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
Convolutional2DAutoencoder <- function(X,filters=c(32,32)){

  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim,1))

  encoded <- input_tensor %>%
    layer_conv_2d(filters=filters[1], kernel_size=3, activation='relu', padding='same') %>%
    layer_max_pooling_2d(pool_size=c(2, 2), padding='same') %>%
    layer_conv_2d(filters=filters[2], kernel_size=3, activation='relu', padding='same') %>%
    layer_max_pooling_2d(pool_size=c(2, 2), padding='same')


  decoded <- encoded %>%
    layer_conv_2d(filters=filters[2], kernel_size=3, activation='relu', padding='same') %>%
    layer_upsampling_2d(size=c(2, 2)) %>%
    layer_conv_2d(filters=filters[1], kernel_size=3, activation='relu', padding='same') %>%
    layer_upsampling_2d(size=c(2, 2)) %>%
    layer_conv_2d(filters=1, kernel_size=3, activation='relu', padding='same')

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded)

  #As well as the decoder model:
  # create a placeholder for an encoded input:
  # - Each Conv2D -> Increase 4th dimension (#Filter number)
  # - Each MaxPool -> Divise 2nd and 3rd dimension by the pool_size
  encoded_input <- layer_input(shape = c(round(timesteps/4),round(input_dim/4),filters[2]))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,5)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]]) %>%
    (decoder_layers_list[[3]]) %>%
    (decoder_layers_list[[4]]) %>%
    (decoder_layers_list[[5]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}


#' LSTMAutoencoder
#' 
#' Auxiliary function defining a LSTM autoencoder model with one hidden layer
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of LSTM cells in the hidden layer - numeric scalar
#' @param time_window - Size of time window (in time steps) for embedding - numeric scalar
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'         
LSTMAutoencoder <- function(X,latent_dim=3,time_window=5){

  # From https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))
  encoded <- input_tensor %>%
    layer_lstm(latent_dim, activation='relu', return_sequences = T , return_state = T)

  decoded <- encoded[[1]] %>%
    layer_lstm(latent_dim, activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=input_dim))

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded[[1]])

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(timesteps,latent_dim))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,2)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

LSTMDenseAutoencoder <- function(X,latent_dim=3,time_window=5){

  # From https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))
  encoded <- input_tensor %>%
    layer_lstm(latent_dim, activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=latent_dim))

  decoded <- encoded %>%
    layer_lstm(latent_dim, activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=input_dim))

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded)

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(timesteps,latent_dim))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,2)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' DeepLSTMAutoencoder
#' 
#' Auxiliary function defining a LSTM autoencoder model with two hidden layers
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of LSTM cells in the hidden layers - numeric vector
#' @param time_window - Size of time window (in time steps) for embedding - numeric scalar
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
#'
DeepLSTMAutoencoder <- function(X,latent_dim=c(10,5),time_window=5){

  # From https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))
  encoded <- input_tensor %>%
    layer_lstm(latent_dim[1], activation='relu', return_sequences = T) %>%
    layer_lstm(latent_dim[2], activation='relu', return_sequences = T, return_state = T)

  decoded <- encoded[[1]] %>%
    layer_lstm(latent_dim[2], activation='relu', return_sequences = T) %>%
    layer_lstm(latent_dim[1], activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=input_dim))

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded[[1]])

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(timesteps,latent_dim[2]))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,3)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]]) %>%
    (decoder_layers_list[[3]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' GRUAutoencoder
#' 
#' Auxiliary function defining a GRU autoencoder model with one hidden layer
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of GRU cells in the hidden layer - numeric scalar
#' @param time_window - Size of time window (in time steps) for embedding - numeric scalar
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
GRUAutoencoder <- function(X,latent_dim=3,time_window=5){

  # From https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))
  encoded <- input_tensor %>%
    layer_gru(latent_dim, activation='relu', return_sequences = T, return_state = T)

  decoded <- encoded[[1]] %>%
    layer_gru(latent_dim, activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=input_dim))

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded[[1]])

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(timesteps,latent_dim))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,2)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' DeepGRUAutoencoder
#' 
#' Auxiliary function defining a GRU autoencoder model with two hidden layers
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of GRU cells in the hidden layers - numeric vector
#' @param time_window - Size of time window (in time steps) for embedding - numeric scalar
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
DeepGRUAutoencoder <- function(X,latent_dim=c(10,5),time_window=5){

  # From https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))
  encoded <- input_tensor %>%
    layer_gru(latent_dim[1], activation='relu', return_sequences = T) %>%
    layer_gru(latent_dim[2], activation='relu', return_sequences = T, return_state = T)

  decoded <- encoded[[1]] %>%
    layer_gru(latent_dim[2], activation='relu', return_sequences = T) %>%
    layer_gru(latent_dim[1], activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=input_dim))

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded[[1]])

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(timesteps,latent_dim[2]))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,3)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]]) %>%
    (decoder_layers_list[[3]])

  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' LSTMConvolutional1DAutoencoder
#' 
#' Auxiliary function defining a convolutional + LSTM autoencoder model with one convolutional and two hidden layers
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param latent_dim - Number of filters (1st value) / LSTM cells (2nd/3rd values) in the hidden layers - numeric vector
#' 
#' @import keras
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{autoencoder}: }{Fitted autoencoder model (full model) - Keras object}
#'         \item{\code{encoder}: }{Encoder part of the autoencoder model (Original space -> Latent Dimensions) - Keras object}
#'         \item{\code{decoder}: }{Decoder part of the autoencoder model (Latent Dimensions -> Original space) - Keras object}
#'         }
#'
LSTMConvolutional1DAutoencoder <- function(X,latent_dim=c(32,32,16)){
  # https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
  # https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
  # https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%201.ipynb#scrollTo=4uh-97bpLZCA

  timesteps <- dim(X)[2]
  input_dim <- dim(X)[3]

  input_tensor <- layer_input(shape = c(timesteps,input_dim))

  encoded <- input_tensor %>%
    layer_conv_1d(filters=latent_dim[1],
                  kernel_size=3,
                  strides=1,
                  activation='relu',
                  padding='same') %>%
    layer_lstm(latent_dim[2], activation='relu', return_sequences = T) %>%
    layer_lstm(latent_dim[3], activation='relu', return_sequences = T, return_state = T)

  decoded <- encoded[[1]] %>%
    layer_lstm(latent_dim[3], activation='relu', return_sequences = T) %>%
    layer_lstm(latent_dim[2], activation='relu', return_sequences = T) %>%
    time_distributed(layer_dense(units=latent_dim[1])) %>%
    layer_conv_1d(filters=input_dim,
                  kernel_size=3,
                  strides=1,
                  activation='relu',
                  padding='same')

  # this model maps an input to its encoded representation
  autoencoder <- keras_model(inputs = input_tensor, outputs = decoded)
  encoder <- keras_model(inputs = input_tensor, outputs = encoded[[1]])

  #As well as the decoder model:
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input <- layer_input(shape = c(timesteps,latent_dim[3]))

  # retrieve the last layer of the autoencoder model
  decoder_layers_list <- utils::tail(autoencoder$layers,4)
  decoder_layers <- encoded_input %>%
    (decoder_layers_list[[1]]) %>%
    (decoder_layers_list[[2]]) %>%
    (decoder_layers_list[[3]]) %>%
    (decoder_layers_list[[4]])


  # create the decoder model
  decoder <- keras_model(inputs = encoded_input, outputs = decoder_layers)

  return(list(autoencoder=autoencoder,encoder=encoder,decoder=decoder))
}

#' plotLoss
#' 
#' Auxiliary function to plot the evolution of the loss function
#'
#' @param model - Keras model containing the history of the loss function to plot
#'
#' @import keras
plotLoss <- function(model){
  epochs <- 1:length(model$history$history$val_loss)
  plot(epochs,model$history$history$val_loss,xlab="Epochs",ylab="Loss")
}

#' learningRateDecay
#'
#' Auxiliary function to be passed to Keras callbacks in order to perform learning rate decay
#'
#' @param epoch - Epoch parameter
learningRateDecay <- function(epoch){10^-8 * 10^(epoch/20)}
