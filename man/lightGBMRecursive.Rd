% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gradientBoosting.R
\name{lightGBMRecursive}
\alias{lightGBMRecursive}
\title{lightGBMRecursive

Wrapper function for LightGBM forecasting technique with recursive multistep ahead technique}
\usage{
lightGBMRecursive(ts_embed, horizon, params = NULL)
}
\arguments{
\item{ts_embed}{\itemize{
\item Embedded time series, in the inp,out format of gbcode::makeEmbedded
}}

\item{horizon}{\itemize{
\item Forecasting horizon
}}

\item{params}{\itemize{
\item Additional (optional) parameters for LightGBM
}}
}
\value{
h-step forecast of ts_embed using lightGBM and recursive multistep-ahead forecasting technique
}
\description{
lightGBMRecursive

Wrapper function for LightGBM forecasting technique with recursive multistep ahead technique
}
