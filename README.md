# Nonlinear regression using the delta method to estimate confidence intervals and prediction intervals in Python

by Greg Pelletier (gjpelletier@gmail.com)

These scripts use the python package called lmfit for nonlinear regression. We also introduce a new function using the delta method, to extend beyond the capabilities of lmfit, to estimate confidence intervals for predicted values, and prediction intervals for new data, using the nonlinear regression fit.

The lmfit package is used to find the best-fit values and the variance-covariance matrix of the model parameters. The user may specify any expression for the nonlinear regression model. 

The lmfit package is described at the following link:

https://lmfit.github.io//lmfit-py/index.html

Two examples are provided: 

- **delta_method_exp3**: In this example we use a 3-parameter exponential function to fit an observed data set for calcification rates of hard clams from Ries et al (2009) (https://doi.org/10.1130/G30210A.1)

- **delta_method_sigmoid4**: In this example we use a 4-parameter logistic function with a sigmoid shape to fit an observed data set provided in the R base package datasets, and consisting of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA. This is the data set used in the example the MAP566 online lecture on nonlinear regression

To estimate the confidence intervals and prediction intervals, we use a new python function that applies the delta method. The delta method is described in detail in Section 5.1 of this MAP566 online lecture by Julien Chiquet from Institut Polytechnique de Paris:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html

