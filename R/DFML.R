#' DFML
#'
#' Core function implementing the EDFML technique.
#' Takes the input data X, and passes it through the dimensionality reduction step (via the dimensionalityReduction fucntion) to obtaing the dynamic factors Z.
#' Factors are then forecast (via the forecaster function), in order to obtain the factor forecasts (Z_hat).
#' Finally, factor forecast are transformed to forecast in the original space via an inverse dimensionality reduction (dimensionality increase).
#'
#' @param X - nxN matrix containing the N time series as columns, each one of length n time steps
#' @param dimensionality_method - Dimensionality reduction method to employ - String among those defined in DIMENSIONALITY_METHOD
#' @param forecast_family - Forecasting family of method to employ - String among those defined in FORECAST_FAMILY
#' @param dimensionality_parameters - Parameters to be passed to the dimensionalityReduction function - List
#' @param forecast_parameters - Parameters to be passed to the forecaster function - List
#' @param components - Number of desired factors - numeric scalar
#' @param h - Forecasting horizon - numeric scalar
#'
#' @return List containing:
#'         \itemize{
#'         \item{\code{X_hat}: }{hxN matrix containing the forecasts of N time series as columns, each one of length h time steps}
#'         \item{\code{Model_dim}: }{Dimensionality reduction model as produced by dimensionality_reduction function}
#'         \item{\code{Time_dim}: }{Computational time required to run the dimensionality reduction model - numeric scalar}
#'         \item{\code{Time_forecast}: }{Computational time required to run the forecasting model - numeric scalar}
#'         }
#' @export
#'
#' @examples
#' #See tests/testthat directory on https://github.com/jdestefani/ExtendedDFML
DFML <- function(X,
                 dimensionality_method=DIMENSIONALITY_METHODS,
                 forecast_family=FORECAST_FAMILY,
                 dimensionality_parameters,
                 forecast_parameters,
                 components,
                 h)
{
  dimentionality_method <- match.arg(dimensionality_method)
  forecast_family <- match.arg(forecast_family)

  dim_red_results <- dimensionalityReduction(X,components,dimensionality_method,dimensionality_parameters)
  forecast_results <- forecaster(dim_red_results$Z,forecast_family,forecast_parameters,h)
  dim_inc_results <- dimensionalityIncrease(forecast_results$Z_hat,dimensionality_method,dim_red_results$model,dim_red_results$Z,dimensionality_parameters)
  return(list(X_hat=dim_inc_results$X_hat,
              Model_dim=dim_red_results$model,
              Time_dim=dim_red_results$time_dim,
              Time_forecast=forecast_results$time_forecast)
         )
}

