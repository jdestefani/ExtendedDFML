#' gradientBoostingForecaster
#' 
#' @import gbcode
#'
#' @param ts - Input time series as a vector
#' @param embedding - Embedding order
#' @param horizon - Forecasting horizon
#' @param delay - Delay for forecasting
#' @param multistep_method - Multistep method (to be chosen among "recursive" and "direct")
#' @param forecasting_method - Gradient boosting forecasting method (to be chosen among "lightgbm" and "xgboost")
#' @param forecasting_params - Parameters to be passed to the forecaster function - List
#'                              \itemize{
#'                                 \item{\code{multistep_method:}{Multistep method to be chosen between \code{direct} and \code{recursive}}}
#'                                 \item{\code{forecasting_method:}{Forecasting method to be chosen between \code{lightgbm} and \code{xgboost}}}
#'                               }
#'                               
#'                              See \pkg{lightgbm} and \pkg{xgboost} documentation for the role of the different parameters
#'                     
#'
#' @return h-step forecast of ts using the chosen multistep_method and forecasting method, with the given embedding order
gradientBoostingForecaster <- function(ts,
                                       embedding,
                                       horizon,
                                       delay=0,
                                       multistep_method=c("recursive","direct"),
                                       forecasting_method=c("lightgbm","xgboost"),
                                       forecasting_params=NULL){

  if(typeof(ts) != "matrix" && dim(ts)[2] != 1){
    stop("Input ts must be a matrix of 1-column")
  }

  multistep_method <- match.arg(multistep_method)
  forecasting_method <- match.arg(forecasting_method)

  ts_embed <- gbcode::MakeEmbedded(ts=ts,
                            n=c(embedding),
                            delay=c(0),
                            hor=c(horizon))

  # Remove the last horizon-1 values causing NA
  ts_embed$inp <- utils::head(ts_embed$inp,-(horizon-1))
  ts_embed$out <- utils::head(ts_embed$out,-(horizon-1))

  switch(forecasting_method,
         lightgbm={
                    switch(multistep_method,
                          recursive={ts_forecast <- lightGBMRecursive(ts_embed,horizon,forecasting_params)},
                          direct={ts_forecast <- lightGBMDirect(ts_embed,horizon,forecasting_params)}
                    )},
         xgboost={switch(multistep_method,
                         recursive={ts_forecast <- xgboostRecursive(ts_embed,horizon,forecasting_params)},
                         direct={ts_forecast <- xgboostDirect(ts_embed,horizon,forecasting_params)}
         )}
  )

  return(ts_forecast)
}

#' lightGBMDirect
#' 
#' Wrapper function for LightGBM forecasting technique with direct multistep ahead technique
#'
#' @import lightgbm
#'
#' @param ts_embed - Embedded time series, in the inp,out format of gbcode::makeEmbedded
#' @param horizon - Forecasting horizon
#' @param params - Additional (optional) parameters for LightGBM
#'
#' @return h-step forecast of ts_embed using lightGBM and direct multistep-ahead forecasting technique

lightGBMDirect <- function(ts_embed,
                           horizon,
                           params=NULL){

  if(is.null(params[["n_threads"]])){
    n_threads <- parallel::detectCores() - 1
  }
  else{
    n_threads <- params$n_threads
  }

  if(is.null(params[["n_rounds"]])){
    n_rounds <- 50
  }
  else{
    n_rounds <- params$n_rounds
  }

  if(is.null(params[["verbose"]])){
    verbose <- 0
  }
  else{
    verbose <- params$verbose
  }

  Y_hat_direct <- c()

  for (h in 1:horizon) {
    lgb.embed <- lgb.Dataset(data=as.matrix(ts_embed$inp), label=ts_embed$out[,h])
    model_direct <- lightgbm(boosting_type = 'gbdt',
                             objective = "regression",
                             metric = 'mae',
                             num_threads = n_threads,
                             lgb.embed,
                             nrounds = n_rounds,
                             verbose = verbose)
    Y_hat_direct <- c(Y_hat_direct,lightgbm::predict(model_direct, dat =  matrix(ts_embed$inp[nrow(ts_embed$inp),],nrow=1)))
  }
  return(Y_hat_direct)
}

#' lightGBMRecursive
#'  
#' Wrapper function for LightGBM forecasting technique with recursive multistep ahead technique
#'
#' @import lightgbm
#'
#' @param ts_embed - Embedded time series, in the inp,out format of gbcode::makeEmbedded
#' @param horizon - Forecasting horizon
#' @param params - Additional (optional) parameters for LightGBM
#' 
#' @return h-step forecast of ts_embed using lightGBM and recursive multistep-ahead forecasting technique

lightGBMRecursive <- function(ts_embed,
                           horizon,
                           params=NULL){

  if(is.null(params[["n_threads"]])){
    n_threads <- parallel::detectCores() - 1
  }
  else{
    n_threads <- params$n_threads
  }

  if(is.null(params[["n_rounds"]])){
    n_rounds <- 50
  }
  else{
    n_rounds <- params$n_rounds
  }

  if(is.null(params[["verbose"]])){
    verbose <- 0
  }
  else{
    verbose <- params$verbose
  }

  Y_hat_recursive <- c()
  lgb.embed <- lgb.Dataset(data=as.matrix(ts_embed$inp), label=ts_embed$out[,1])
  model_recursive <- lightgbm(boosting_type = 'gbdt',
                              objective = "regression",
                              metric = 'mae',
                              num_threads = n_threads,
                              lgb.embed,
                              nrounds = n_rounds,
                              verbose = verbose)
  X_input <- matrix(ts_embed$inp[nrow(ts_embed$inp),],nrow=1)
  for (h in 1:horizon) {
    Y_hat_current <- lightgbm::predict(model_recursive, dat = X_input)
    Y_hat_recursive <- c(Y_hat_recursive,Y_hat_current)

    X_input <-  matrix(c(X_input[-1],Y_hat_current),nrow=1)
  }

  return(Y_hat_recursive)
}

#' xgboostDirect
#' 
#' Wrapper function for XGBoost forecasting technique with direct multistep ahead technique
#'
#' @import xgboost
#'
#' @param ts_embed - Embedded time series, in the inp,out format of gbcode::makeEmbedded
#' @param horizon - Forecasting horizon
#' @param params - Additional (optional) parameters for XGBoost
#'
#' @return h-step forecast of ts_embed using xgboost and direct multistep-ahead forecasting technique
xgboostDirect <- function(ts_embed,
                           horizon,
                           params=NULL){

  if(is.null(params[["n_threads"]])){
    n_threads <- parallel::detectCores() - 1
  }
  else{
    n_threads <- params$n_threads
  }

  if(is.null(params[["n_rounds"]])){
    n_rounds <- 50
  }
  else{
    n_rounds <- params$n_rounds
  }

  if(is.null(params[["verbose"]])){
    verbose <- 0
  }
  else{
    verbose <- params$verbose
  }

  Y_hat_direct <- c()

  for (h in 1:horizon) {
    xgb.embed <- xgb.DMatrix(data=as.matrix(ts_embed$inp), label=ts_embed$out[,h])
    model_direct <- xgboost(data = xgb.embed,
                            max.depth = 2,
                            eta = 1,
                            nthread = n_threads,
                            nrounds = n_rounds,
                            objective = "reg:squarederror",
                            verbose = verbose)
    Y_hat_direct <- c(Y_hat_direct,xgboost::predict(model_direct,matrix(ts_embed$inp[nrow(ts_embed$inp),],nrow=1)))
  }
  return(Y_hat_direct)
}

#' xgboostRecursive
#' 
#' Wrapper function for XGboost forecasting technique with recursive multistep ahead technique
#'
#' @import xgboost
#'
#' @param ts_embed - Embedded time series, in the inp,out format of gbcode::makeEmbedded
#' @param horizon - Forecasting horizon
#' @param params - Additional (optional) parameters for XGBoost
#'
#' @return h-step forecast of ts_embed using xgboost and recursive multistep-ahead forecasting technique
xgboostRecursive <- function(ts_embed,
                              horizon,
                              params=NULL){

  if(is.null(params[["n_threads"]])){
    n_threads <- parallel::detectCores() - 1
  }
  else{
    n_threads <- params$n_threads
  }

  if(is.null(params[["n_rounds"]])){
    n_rounds <- 50
  }
  else{
    n_rounds <- params$n_rounds
  }

  if(is.null(params[["verbose"]])){
    verbose <- 0
  }
  else{
    verbose <- params$verbose
  }

  Y_hat_recursive <- c()
  xgb.embed <- xgb.DMatrix(data=as.matrix(ts_embed$inp), label=ts_embed$out[,1])
  model_recursive <- xgboost(data = xgb.embed,
                             max.depth = 2,
                             eta = 1,
                             nthread = n_threads,
                             nrounds = n_rounds,
                             objective = "reg:squarederror",
                             verbose = 0)

  X_input <- matrix(ts_embed$inp[nrow(ts_embed$inp),],nrow=1)
  for (h in 1:horizon) {
    Y_hat_current <- xgboost::predict(model_recursive,X_input)
    Y_hat_recursive <- c(Y_hat_recursive,Y_hat_current)
    X_input <-  matrix(c(X_input[-1],Y_hat_current),nrow=1)
  }

  return(Y_hat_recursive)
}
