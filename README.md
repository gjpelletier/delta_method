# Nonlinear regression using scipy combined with using the delta-method or parametric bootstrap to estimate confidence intervals and prediction intervals

**An example showing how to use new functions called "delta_method" and "parametric_bootstrap"**

by Greg Pelletier (gjpelletier@gmail.com)

This script uses scipy to find the optimum parameters and the variance-covariance of the parameters for nonlinear regression. We also show how to use the following two new new functions to extend beyond the capabilities of scipy, to estimate confidence intervals for predicted values, and prediction intervals for new data:

- **delta_method**
- **parametric_bootstrap**

The first step is to use scipy to find the best-fit values and the variance-covariance matrix of the model parameters.

The second step is to estimate the confidence intervals and prediction intervals using new delta_metod and parametric_bootstrap functionsas described in this online lecture:

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
