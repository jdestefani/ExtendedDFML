#' matrix2Tensor3D
#' 
#' Auxiliary function to embed a 2D matrix into a 3D tensor.
#'
#' @param X - Input matrix
#' @param time_window - Integer specifying the size of the time window (aka model order, 2nd dimension in tensor).
#' @param shift - Integer specifying how far the sliding window should be shifted in order to create the tensor (N.B. shift < time_window implies overlapping data)
#' @param padNA - Boolean specifying whether NAs should be used to pad tensors in case the input matrix size is not multiple of the time_window.
#'
#' @import abind
#' @export
#' @return 3D Tensor (Samples,Time Window,Features) containing the embedded form of X (the second dimension is fixed to time_window)
#' 
#' @examples 
#' X <- EuStockMarkets
#' time_window <- 5
#' X_windowed <- matrix2Tensor3D(X,time_window,time_window)
matrix2Tensor3D <- function(X,time_window,shift=1,padNA=FALSE){
    
    if(nrow(X) == 1){
        stop("[ERROR] - Embedding not possible with a row vector")
    }
    
    # Determine splits of the original matrix, by starting from the end, and going back to the beginning.
    # To determine the next split we move back of "shift" elements.
    # If the first split contains more values than the time window, the additional values will be not part of any
    splits <- rev(seq(nrow(X),time_window,-shift))
    missing_values <- utils::head(splits,1)-time_window
    
    # 3D Array of the required matrices -> (Time Window, Features, Samples)
    matrix_array<- sapply(splits, function(i){X[(i-time_window+1):i,]},simplify = "array")
    
    if(padNA){
        # If some values are lost due to the choice of the shift,
        # recover these values and pad them with NA to ensure consistency
        if(missing_values > 0){
            filler <- matrix(rep(NA,ncol(X)*(time_window-missing_values)),ncol=ncol(X))
            matrix_array <- abind(rbind(filler,X[1:missing_values,]),matrix_array,along = 3)
        }
    }
    else{
        warning(paste((time_window-missing_values+1),"values have been removed to ensure consistency of tensor"))
    }
    
    # Convert list to 3D vector -> (Samples,Time Window,Features)
    return(aperm(matrix_array,c(3,1,2)))
}

#' tensor3D2matrix
#' 
#' Auxiliary function to flatten 3D tensor into a 2D matrix, discarding overlapping data.
#'
#' @param X - Input 3D Tensor (Samples,Time Window,Features) containing the embedded form of X
#' @param shift - Integer specifying the shift value used to construct the tensor
#' 
#' @export
#' @return 2D Matrix (Samples,Features) obtained by "unrolling" the tensor
#' 
#' @examples 
#' X <- EuStockMarkets
#' time_window <- 5
#' X_windowed <- matrix2Tensor3D(X,time_window,time_window)
#' X_reconstructed <- tensor3D2matrix(X_windowed,time_window)
tensor3D2matrix <- function(X,shift=1){
    n_columns <- dim(X)[3]
    time_window <- dim(X)[2]
    overlap <- time_window - shift
    
    if(overlap > 0){
        # Remove overlapping values from time windows
        matrix_array <- sapply(1:dim(X)[1], function(i){utils::tail(X[i,,],-overlap)},simplify = "array")
        
        # Flatten 3D Array into 2D Matrix -> (Samples*Time Window, Features)
        matrix2d <- matrix(aperm(matrix_array, c(1, 3, 2)), ncol=n_columns)
        matrix2d <- rbind(utils::head(X[1,,],overlap),matrix2d)
    }
    else{
        matrix2d <- matrix(aperm(X, c(2, 1, 3)), ncol=n_columns)
    }
    
    # Convert list to 2D matrix -> (Samples,Features)
    return(matrix2d)
}
