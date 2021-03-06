% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gradientBoosting.R
\name{xgboostDirect}
\alias{xgboostDirect}
\title{xgboostDirect}
\usage{
xgboostDirect(ts_embed, horizon, params = NULL)
}
\arguments{
\item{ts_embed}{\itemize{
\item Embedded time series, in the inp,out format of gbcode::makeEmbedded
}}

\item{horizon}{\itemize{
\item Forecasting horizon
}}

\item{params}{\itemize{
\item Additional (optional) parameters for XGBoost
}}
}
\value{
h-step forecast of ts_embed using xgboost and direct multistep-ahead forecasting technique
}
\description{
Wrapper function for XGBoost forecasting technique with direct multistep ahead technique
}
