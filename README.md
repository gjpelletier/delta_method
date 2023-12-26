# delta_method

Nonlinear regression using lmfit combined with using the delta method to estimate confidence intervals and prediction intervals

by Greg Pelletier (gjpelletier@gmail.com) 12/25/2023

This python script uses the python package called lmfit for nonlinear regression. We also introduce a new function using the delta method, to extend beyond the capabilities of lmfit, to estimate confidence intervals for predicted values, and prediction intervals for new data, using the nonlinear regression fit.

The lmfit package is used to find the best-fit values and the variance-covariance matrix of the model parameters. The user may specify any expression for the nonlinear regression model. 

The lmfit package is described at the following link:

https://lmfit.github.io//lmfit-py/index.html

In this example we use a 3-parameter exponential function to fit an observed data set for calcification rates of hard clams from Ries et al (2009) (https://doi.org/10.1130/G30210A.1)

To estimate the confidence intervals and prediction intervals, we use a new python function that applies the delta method. The delta method is described in detail in Section 5.1 of this online lecture:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html

