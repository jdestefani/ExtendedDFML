% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/forecaster.R
\name{VAR_recursive}
\alias{VAR_recursive}
\title{VAR_recursive}
\usage{
VAR_recursive(X, model_order, h)
}
\arguments{
\item{X}{\itemize{
\item nxN matrix containing the N time series as columns, each one of length n time steps
}}

\item{model_order}{\itemize{
\item Model order (m in VAR(m)) for the VAR model - numeric scalar
}}

\item{h}{\itemize{
\item Forecasting horizon
}}
}
\value{
List containing:
\itemize{
\item{\code{forecasts}: }{N x h matrix containing the h-step ahead forecast for the N input series}
\item{\code{coefficients}: }{(m+1) x N matrix containing the model coefficients (m coefficient + constant)}
}
}
\description{
Implementation of a recursive multivariate VAR Model
}
