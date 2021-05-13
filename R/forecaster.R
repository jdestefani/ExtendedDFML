#' forecaster
#'
#' Function implementing the forecasting module in the DFML architecture
#'
#' @param Z - nxk matrix containing the k time series as columns, each one of length n time steps
#' @param family - Forecasting family of method to employ - String among those defined in FORECAST_FAMILY
#' @param parameters - Parameters to be passed to the forecaster function - List
#'                     
#'                     The method name is passed through the \code{parameters$method} and should be one among those defined in \code{MULTISTEPAHEAD_METHODS}, \code{M4_METHODS}, \code{GRADIENT_BOOSTING_METHODS} 
#'                     
#'                     For the different methods family, at least the embedding order/model order \code{m} is required.
#'                     
#'                     For additional parameters:
#'                     \itemize{
#'                       \item{\code{MULTISTEPAHEAD_METHODS}: }{See \pkg{gbcode::multipleStepAhead} documentation for the role of the different parameters}
#'                       \item{\code{M4_METHODS}: }{No specific parameters required}
#'                       \item{\code{GRADIENT_BOOSTING_METHODS}: }{ 
#'                              \itemize{
#'                                 \item{\code{multistep_method:}{Multistep method to be chosen between \code{direct} and \code{recursive}}}
#'                                 \item{\code{forecasting_method:}{Forecasting method to be chosen between \code{lightgbm} and \code{xgboost}}}
#'                               }
#'                              See \pkg{lightgbm} and \pkg{xgboost} documentation for the role of the different parameters}
#'                     }
#'                     
#' @param h - Forecasting horizon - numeric scalar
#' 
#' @import gbcode
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{Z_hat}: }{k x h matrix containing the h-step ahead forecast for the k input series}
#'         \item{\code{time_forecast}: }{Computational time required to run the forecasting model - numeric scalar}
#'         }
#' @examples 
#' #See tests/testthat directory on https://github.com/jdestefani/ExtendedDFML
forecaster <- function(Z,
                       family=FORECAST_FAMILY,
                       parameters=NULL,
                       h){
    family <- match.arg(family)
    switch(family,
           multistepAhead={
               if(is.null(parameters$m)){
                   stop("[multistepAhead] m (Embedding order) parameter missing from the parameter list")
               }
               if(is.null(parameters$Kmin)){
                   stop("[multistepAhead] Kmin (Minimum number of neighbors) parameter missing from the parameter list")
               }
               if(is.null(parameters$C)){
                   stop("[multistepAhead] C (C*k = Maximimum number of neighbors) parameter missing from the parameter list")
               }
               if(is.null(parameters$FF)){
                   stop("[multistepAhead] FF (Forgetting factor) parameter missing from the parameter list")
               }
               parameters$method <- match.arg(parameters$method,MULTISTEPAHEAD_METHODS)
               ptm <- proc.time()
               if(parameters$method %in% c("mimo.pls","mimo.acf","mimo.acf.lin")){ # Those methods required a 1xn array as input
                   Z_hat <- apply(Z,2,function(ts){
                       gbcode::multiplestepAhead(array(ts,dim = c(1,length(ts))),n=parameters$m, H=h,method=parameters$method,Kmin=parameters$Kmin,C=parameters$C)
                   })
                   time_forecast <- proc.time() - ptm
               }
               else{ # The other methods work fine with a numeric
                   Z_hat <- apply(Z,2,gbcode::multiplestepAhead,n=parameters$m, H=h,method=parameters$method,Kmin=parameters$Kmin,C=parameters$C)
                   time_forecast <- proc.time() - ptm
               }
           },
           M4Methods={
               parameters$method <- match.arg(parameters$method,M4_METHODS)
               ptm <- proc.time()
               switch (parameters$method,
                       ES={forecast_list <- apply(Z, 2, MultivariateStatisticalBenchmarksTS::ESBenchmark ,h=h)},
                       holt_winters={forecast_list <- apply(Z, 2, MultivariateStatisticalBenchmarksTS::HoltWintersBenchmark ,h=h)},
                       holt_winters_damped={forecast_list <- apply(Z, 2, MultivariateStatisticalBenchmarksTS::HoltWintersDampedBenchmark ,h=h)},
                       theta={forecast_list <- apply(Z, 2, MultivariateStatisticalBenchmarksTS::thetaBenchmark ,h=h)},
                       combined={forecast_list <- apply(Z, 2, MultivariateStatisticalBenchmarksTS::combinedBenchmark ,h=h)}
               )
               time_forecast <- proc.time() - ptm
               Z_hat <- as.matrix(Reduce(cbind,lapply(forecast_list,function(x){x$forecasts})))
               colnames(Z_hat) <- outer("Z",seq(1:ncol(Z_hat)),paste0)},
           VAR={
               if(is.null(parameters$m)){
                   stop("[VAR] m (model order) parameter missing from the parameter list")
               }
               
               ptm <- proc.time()
               
               results <- VAR_recursive(Z,model_order = parameters$m,h)
               
               time_forecast <- proc.time() - ptm
               
               Z_hat <- results$forecast_matrix
           },
           gradientBoosting={
               if(is.null(parameters$m)){
                   stop("[gradientBoosting] m (Embedding order) parameter missing from the parameter list")
               }
               if(is.null(parameters$multistep_method)){
                   stop("[gradientBoosting] multistep_method parameter missing from the parameter list")
               }
               if(is.null(parameters$forecasting_method)){
                   stop("[gradientBoosting] forecasting_method parameter missing from the parameter list")
               }
               ptm <- proc.time()
               Z_hat <- apply(Z, 2, function(ts){
                   gradientBoostingForecaster(ts=matrix(ts,ncol = 1),
                                              embedding=parameters$m,
                                              horizon=h,
                                              multistep_method=parameters$multistep_method,
                                              forecasting_method=parameters$forecasting_method,
                                              forecasting_params=parameters$forecasting_params)})
               
               time_forecast <- proc.time() - ptm
               colnames(Z_hat) <- outer("Z",seq(1:ncol(Z_hat)),paste0)
           }
    )
    
    return(list(Z_hat=Z_hat,time_forecast=time_forecast[3]))
}




#' VAR_recursive
#' 
#' Implementation of a recursive multivariate VAR Model
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param model_order - Model order (m in VAR(m)) for the VAR model - numeric scalar
#' @param h - Forecasting horizon
#'
#' @import vars
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{forecasts}: }{N x h matrix containing the h-step ahead forecast for the N input series}
#'         \item{\code{coefficients}: }{(m+1) x N matrix containing the model coefficients (m coefficient + constant)}
#'         }
#'
VAR_recursive <- function(X,model_order,h) {
    VAR_model <- vars::VAR(X,p=model_order)
    coefficient_matrix <- mapply(coefficients,VAR_model$varresult)
    coefficient_matrix[is.na(coefficient_matrix)] <- 0
    lag_matrix <- as.matrix(VAR_model$datamat[,-c(1:dim(X)[2])])
    value_matrix <- as.matrix(VAR_model$datamat[,c(1:dim(X)[2])])
    
    # Recursive forecast
    for (i in 1:h) {
        # Multiply coefficients for lagged values to obtain new values
        value_matrix <- rbind(value_matrix,utils::tail(lag_matrix,1) %*% coefficient_matrix)
        # Add newly computed values as lags for the following estimation
        lag_matrix <- rbind(lag_matrix,c(as.vector(t(utils::tail(value_matrix,model_order)[model_order:1,])),1))
    }
    
    return(list(forecast_matrix=utils::tail(value_matrix,h),coefficient_matrix=coefficient_matrix))
}
