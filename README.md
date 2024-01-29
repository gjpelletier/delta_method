# Confidence intervals for nonlinear regression using the delta-method or parametric bootstrap in Jupyter Notebooks, Python, or MATLAB

by Greg Pelletier (gjpelletier@gmail.com)

We introduce the following two new new functions to estimate confidence intervals and prediction intervals for nonlinear regression:

- **delta_method**
- **parametric_bootstrap**

The first step before using either of these two new functions is to use scipy's opt.curve_fit or lmfit to find the best-fit values and the covariance matrix of the model parameters.

The second step is to estimate the confidence intervals and prediction intervals using new delta_metod or parametric_bootstrap functions using the methods described in this online lecture:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#confidence-intervals-and-prediction-intervals

# Installation for MATLAB

Download delta_method.m and parametric_boostrap.m from this github repository (https://github.com/gjpelletier/delta_method) or MATLAB File Exchange and add them to your working directory or session search path.<br>

# Installation for Google Colab, Jupyter Notebooks, and Python

First install delta_method.py as follows with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/delta_method.git
```

Next import the delta_method and parametric_bootstrap functions as follows in your notebook or python code:<br>
```
from delta_method import delta_method, parametric_bootstrap
```

As an alternative, you can also download delta_method.py from this github repository (https://github.com/gjpelletier/delta_method) and add it to your own project.<br>

# Example for Google Colab, Jupyter Notebook, and Python

An example showing how to use the new delta_method and parametric_bootstrap functions is provided in this matlab script:

https://github.com/gjpelletier/delta_method/blob/main/delta_method_example_example.m

# Example for Google Colab, Jupyter Notebook, and Python

An example showing how to use the new delta_method and parametric_bootstrap functions is provided in this Jupyter Notebook:

https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.ipynb
