# Nonlinear regression using scipy combined with using the delta-method or parametric bootstrap to estimate confidence intervals and prediction intervals

by Greg Pelletier (gjpelletier@gmail.com)

We introduce the following two new new functions, to extend beyond the capabilities of scipy's opt.curve_fit, to estimate confidence intervals for predicted values, and prediction intervals for nonlinear regression:

- **delta_method**
- **parametric_bootstrap**

The first step before using either of these two new functions is to use scipy's opt.curve_fit to find the best-fit values and the covariance matrix of the model parameters.

The second step is to estimate the confidence intervals and prediction intervals using new delta_metod or parametric_bootstrap functions using the methods described in this online lecture:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#confidence-intervals-and-prediction-intervals

# Installation for Google Colab, Jupyter Notebooks, and Python

First install weightedcorrs as follows with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/delta_method.git
```

Next import the delta_method and parametric_bootstrap function as follows in your notebook or python code:<br>
```
from weightedcorrs import weightedcorrs
```
# Example for Google Colab, Jupyter Notebook, and Python

An example showing how to use the new delta_method and parametric_bootstrap functions is provided in this Jupyter Notebook:

https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.ipynb
