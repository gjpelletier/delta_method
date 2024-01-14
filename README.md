# Nonlinear regression using the delta-method or parametric bootstrap to estimate confidence intervals and prediction intervals in Python and Jupyter Notebooks

by Greg Pelletier (gjpelletier@gmail.com)

Introducing new Python functions using either the delta-method or parametric bootstrap to estimate confidence intervals for predicted values, and prediction intervals for new data, using nonlinear regression. These new functions extend the capabilities of the python packages scipy or lmfit to apply either the delta-method or parametric bootstrap for confidence intervals and prediction intervals. 

The first step is to use either scipy or lmfit to find the optimum parameter values and the variance-covariance matrix of the model parameters. The user may specify any expression for the nonlinear regression model.

The second step is to estimate the confidence intervals and prediction intervals using new python functions that apply either the delta-method or parametric bootstrap. 

The following example is provided: 

- **delta_method_sigmoid4**: In this example we use a 4-parameter logistic function with a sigmoid shape to fit an observed data set provided in the R base package datasets, and consisting of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA. This is the data set used in the example the MAP566 online lecture on nonlinear regression (https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#confidence-intervals-and-prediction-intervals). This example also shows how to use a **parametric bootstrap** as an alternative to the **delta-method**, and how to calculate confidence intervals for model parameters.

The user may build any expression for the nonlinear relationship between observed x and y for the nonlinear regression using either scipy.optimize.curve_fit or the ExpressionModel function of lmfit.

To estimate the confidence intervals and prediction intervals, we use new python functions that apply either the delta-method or parametric bootstrap as described in Section 5 of this MAP566 online lecture by Julien Chiquet from Institut Polytechnique de Paris:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#confidence-intervals-and-prediction-intervals

