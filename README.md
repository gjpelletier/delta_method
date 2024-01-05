# Nonlinear regression using the delta-method to estimate confidence intervals and prediction intervals in Python and Jupyter Notebooks

by Greg Pelletier (gjpelletier@gmail.com)

Introducing a new Python function using the delta-method to estimate confidence intervals for predicted values, and prediction intervals for new data, using nonlinear regression. This new function extends the capabilities of the python packages scipy or lmfit to apply the delta-method for confidence intervals and prediction intervals. 

The first step is to use either scipy or lmfit to find the optimum parameter values and the variance-covariance matrix of the model parameters. The user may specify any expression for the nonlinear regression model.

The second step is to estimate the confidence intervals and prediction intervals using a new python function that applies the delta-method. 

Three examples are provided: 

- **delta_method_sigmoid4**: In this example we use a 4-parameter logistic function with a sigmoid shape to fit an observed data set provided in the R base package datasets, and consisting of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA. This is the data set used in the example the MAP566 online lecture on nonlinear regression (https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#the-delta-method). We also show how to use a parametric bootstrap as an alternative to the delta-method following the example in Section 5.2 of Julien Chiquet's online lecture.

- **delta_method_asympt3**: In this example we use an asymptotic 3-parameter exponential function to fit an observed data set for calcification rates of hard clams from Ries et al (2009) (https://doi.org/10.1130/G30210A.1)

- **delta_method_monod2**: In this example we use a 2-parameter Monod function (Michaelis-Menten) to fit an enzymology data set (https://rforbiochemists.blogspot.com/2015/05/plotting-and-fitting-enzymology-data.html)

The user may build any expression for the nonlinear relationship between observed x and y for the nonlinear regression using either scipy.optimize.curve_fit or the ExpressionModel function of lmfit.

To estimate the confidence intervals and prediction intervals, we use a new python function that applies the delta-method. The delta-method is described in detail in Section 5.1 of this MAP566 online lecture by Julien Chiquet from Institut Polytechnique de Paris:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#the-delta-method

