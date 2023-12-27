# Nonlinear regression using the delta method to estimate confidence intervals and prediction intervals in Python

by Greg Pelletier (gjpelletier@gmail.com)

Introducing a new Python function using the delta method to estimate confidence intervals for predicted values, and prediction intervals for new data, using nonlinear regression. This new delta method function extends the capabilities of the python package lmfit to apply the delta method for confidence intervals and prediction intervals. 

The lmfit package (https://lmfit.github.io/lmfit-py/) is used to find the best-fit parameter values, and the variance-covariance matrix of the model parameters. Our new delta method function is applied next to find the confidence intervals and prediction intervals.

Two examples are provided: 

- **delta_method_sigmoid4**: In this example we use a 4-parameter logistic function with a sigmoid shape to fit an observed data set provided in the R base package datasets, and consisting of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA. This is the data set used in the example the MAP566 online lecture on nonlinear regression.

- **delta_method_asympt3**: In this example we use an asymptotic 3-parameter exponential function to fit an observed data set for calcification rates of hard clams from Ries et al (2009) (https://doi.org/10.1130/G30210A.1)

The user may build any expression for the nonlinear relationship between observed x and y for the nonlinear regression using the ExpressionModel function of lmfit.

To estimate the confidence intervals and prediction intervals, we use a new python function that applies the delta method. The delta method is described in detail in Section 5.1 of this MAP566 online lecture by Julien Chiquet from Institut Polytechnique de Paris:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html

