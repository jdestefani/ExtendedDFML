#' vectorNormalize
#' Normalize the vector passed as parameter
#'
#' @param u - Vector to normalize
#'
#' @return Normalized vector
#'
vectorNormalize<-function(u){
  return(as.numeric(u%*%u))

}

#' vectorProject
#' Compute the vector projection of the vector u passed as parameters onto the vector v
#'
#' @param u - Vector to project
#' @param v - Vector to project onto
#'
#' @return Projection of u onto v
vectorProject<-function(u,v){

  return(u*as.numeric(u%*%v)/as.numeric(u%*%u))

}

#' cor.prob
#' 
#' Auxiliary function to perform statistical testing of the correlation levels in a correlation matrix
#'
#' @param X - n x n correlation matrix - numeric
#' @param dfr - Degrees of freedom - numeric
#'
#' @return - \code{R} - Matrix containing the probabilities of statistical significance of the different correlations
cor.prob <- function(X, dfr = nrow(X) - 2) {
  n<-NCOL(X)
  nt=n*(n-1)/2
  R <- stats::cor(X)
  above <- row(R) < col(R)
  r2 <- R[above]^2
  Fstat <- r2 * dfr / (1 - r2)
  R[above] <- min(1,(1 - stats::pf(Fstat, 1, dfr))*nt)
  R
}

#' incrementalPCA
#' 
#' Wrapper function for an iterative PCA method (onlinePCA::incRpca)
#'
#' @param X - n x N matrix containing the N time series as columns, each one of length n time steps
#' @param xbar - 1 x N matrix containing the means of the N input time series
#' @param base_pca_values - Matrix containing the PCA eigenvalues to use for the initialization
#' @param base_pca_vectors - Matrix containing the PCA eigenvectors to use for the initialization
#' @param pca_type - Incremental PCA method to employ - String among those defined in \code{INCREMENTAL_PCA_METHODS}
#' @param forgetting_factor - Forgetting factor to employ in the incremental PCA - Numeric
#' @param components - Desired number of PCA components - Numeric
#' @param start_value - Index of the first value in the dataset to be used for incremental PCA - Numeric
#' @param end_value - Index of the last value in the dataset to be used for incremental PCA - Numeric
#'
#' @return List containing:
#'         \itemize{
#'         \code{pca} - List containing:
#'           \itemize{
#'          \item{\code{values}: }{Matrix containing the eigenvalues associated to PCA}
#'          \item{\code{vectors}: }{Matrix containing the eigenvectors associated to PCA as columns}
#'          }
#'         \item{\code{x_bar}: }{1 x N matrix containing the means of the N input time series}
#'         }
#' 
incrementalPCA <- function(X,
                           xbar= NULL,
                           base_pca_values,
                           base_pca_vectors,
                           pca_type=INCREMENTAL_PCA_METHODS,
                           forgetting_factor=NA,
                           components=2,
                           start_value,
                           end_value){

  pca_type <- match.arg(pca_type)

  if(pca_type != "incrpca" & is.null(xbar)){
    stop("A mean vector (xbar) must be provided in order to use this online PCA method")
  }

  pca <- list(values=base_pca_values,vectors=base_pca_vectors)

  if(is.null(xbar)){
    if(pca_type %in% c("incrpca_block")){
      switch(pca_type,
             incrpca_block={pca <- onlinePCA::incRpca.block(X[(start_value:end_value),], (end_value-start_value), array(pca$values), pca$vectors, start_value,q=components,byrow = TRUE)})
    }
    else{
      for (i in start_value:end_value){
        switch(pca_type,
          incrpca={f <- if(is.na(forgetting_factor)) (1/(i-1)) else forgetting_factor
                   pca <- onlinePCA::incRpca(pca$values, pca$vectors, X[i,], i-1, f=f, q = components)})
      }
    }
  }
  else{
    if(pca_type %in% c("incrpca_block")){
      switch(pca_type,
             incrpca_block={pca <- onlinePCA::incRpca.block(X[(start_value:end_value),], B=(end_value-start_value), lambda=array(pca$values), U=pca$vectors, n0=start_value, q=components,center=xbar,byrow = TRUE)})
      xbar <- updateMean(xbar, X[(start_value:end_value),], start_value)
    }
    else{
        for (i in start_value:end_value){
          iterative_pca_results <- iterativePCASingleStep(X[i,],i,pca_type,xbar,pca,forgetting_factor,components)
          xbar <- iterative_pca_results$xbar
          pca <- iterative_pca_results$pca
        }
    }
  }

  return(list(pca=pca,xbar=xbar))
}

#' iterativePCASingleStep
#' 
#' Wrapper function implementing a single update in the iterative PCA procedure
#'
#' @param X_i - 1 x N matrix containing the sample of the original dataset to use for the update - Numeric 
#' @param i - Index of the sample - numeric
#' @param pca_type - Incremental PCA method to employ - String among those defined in \code{INCREMENTAL_PCA_METHODS}
#' @param xbar - 1 x N matrix containing the means of the N input time series
#' @param pca - List containing:
#'           \itemize{
#'          \item{\code{values}: }{Matrix containing the eigenvalues associated to PCA}
#'          \item{\code{vectors}: }{Matrix containing the eigenvectors associated to PCA as columns}
#'          }
#' @param forgetting_factor - Forgetting factor to employ in the incremental PCA - Numeric
#' @param components - Desired number of PCA components - Numeric
#' 
#' @return List containing:
#'         \itemize{
#'         \code{pca} - List containing:
#'           \itemize{
#'          \item{\code{values}: }{Matrix containing the eigenvalues associated to PCA}
#'          \item{\code{vectors}: }{Matrix containing the eigenvectors associated to PCA as columns}
#'          }
#'         \item{\code{x_bar}: }{1 x N matrix containing the means of the N input time series}
#'         }
iterativePCASingleStep <- function(X_i,i,pca_type,xbar,pca,forgetting_factor=NA,components){
  xbar <- onlinePCA::updateMean(xbar, X_i, i-1)
  switch(pca_type,
         incrpca={f <- if(is.na(forgetting_factor)) (1/(i-1)) else forgetting_factor
         pca <- onlinePCA::incRpca(pca$values, pca$vectors, X_i, i-1, f = f , q = components,center = xbar)},
         ccipca={l <- if(is.na(forgetting_factor)) 2 else forgetting_factor
         pca <- onlinePCA::ccipca(pca$values, pca$vectors, X_i, i-1, q = components, l = l, center = xbar)}, # 0 <= l < n, typical l \in [2,4]
         ghapca={gamma <- if(is.na(forgetting_factor)) (2/i) else forgetting_factor
         pca <- onlinePCA::ghapca(pca$values, pca$vectors, X_i, gamma = gamma, components, xbar)}, # gamma is a kind of forgetting factor -> c/n
         perturbationrpca={
           f <- if(is.na(forgetting_factor)) (1/(i-1)) else forgetting_factor
           pca <- onlinePCA::perturbationRpca(pca$values, pca$vectors, X_i, i-1, f=f, center=xbar)}, # 0 < f < 1
         #secularpca={pca <- onlinePCA::secularRpca(pca$values, pca$vectors, X[i,], i, center=xbar)},
         sgapca={gamma <- if(is.na(forgetting_factor)) (2/i) else forgetting_factor
         pca <- onlinePCA::sgapca(pca$values, pca$vectors, X_i, gamma=gamma, components, xbar)}, # gamma is a kind of forgetting factor -> c/n
         snlpca={gamma <- if(is.na(forgetting_factor)) (1/i) else forgetting_factor
         pca <- onlinePCA::snlpca(pca$values, pca$vectors, X_i, gamma=gamma, components, xbar)} # gamma is a kind of forgetting factor -> c/n
  )
  return(list(pca=pca,xbar=xbar))
}

#' incrementalpca_wrapper
#' 
#' Wrapper function for the iterative PCA method (onlinePCA::incRpca).
#'
#' @param X - Input dataset - Numeric matrix
#' @param components - Desired number of PCA components - Numeric 
#' @param init_sample_percentage - Percentage of the input dataset to be used as initialisation for iterative PCA - numeric
#' @param stop_iterative - Boolean flag to select whether the iterative part would be performed or not - boolean
#' @param centered - Boolean flag to selected whether a centering of the dataset would be performed or not - boolean
#'
#' @return \code{pca} - List containing:
#'         \itemize{
#'         \item{\code{values}: }{Matrix containing the eigenvalues associated to PCA}
#'         \item{\code{vectors}: }{Matrix containing the eigenvectors associated to PCA as columns}
#'         }
incrementalpca_wrapper <- function(X,components=10,init_sample_percentage=0.5, stop_iterative=FALSE, centered=TRUE){
  n <- nrow(X)
  init_sample_size <- round(init_sample_percentage*n)
  if(centered){
    ## Incremental PCA (IPCA, centered)
    pca <- stats::prcomp(X[1:init_sample_size,]) # initialization
    xbar <- pca$center
    pca <- list(values=pca$sdev[1:components]^2, vectors=pca$rotation[,1:components])
    if(!stop_iterative){
      pca <- incrementalPCA(X, xbar = xbar, base_pca_values = pca$values, base_pca_vectors = pca$vectors, components = components, start_value = (init_sample_size+1),end_value = n)
    }
  }
  else{
    ## Incremental PCA (IPCA, uncentered)
    pca <- stats::prcomp(X[1:init_sample_size,],center=FALSE) # initialization
    pca <- list(values = pca$sdev[1:components]^2, vectors = pca$rotation[,1:components])
    if(!stop_iterative){
      pca <- incrementalPCA(X, xbar= NULL, base_pca_values = pca$values, base_pca_vectors = pca$vectors, components = components,start_value = (init_sample_size+1),end_value = n)
    }
  }
  return(pca)
}
