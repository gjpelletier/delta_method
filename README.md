# Confidence and prediction intervals for nonlinear regression using the delta-method or parametric bootstrap in MATLAB, Python, Jupyter Notebook, and Google Colab

by Greg Pelletier (gjpelletier@gmail.com)

We introduce the following two new new functions to estimate confidence intervals and prediction intervals for nonlinear regression:

- **delta_method**
- **parametric_bootstrap**

The first step before using either of these two new functions is to find the optimum parameter values and the parameter covariance matrix. This step can be done using MATLAB's nlinfit, or Python's scipy opt.curve_fit or lmfit.

The second step is to estimate the confidence intervals and prediction intervals using our new delta_method or parametric_bootstrap functions. We also show how to use the parametric_bootstrap function as an alternative to linear approximations to estimate confidence intervals of the nonlinear regression model parameters.

# Installation for MATLAB

Download the delta_method.m and parametric_boostrap.m files from this github repository (https://github.com/gjpelletier/delta_method) or MATLAB File Exchange and copy them to your working directory or session search path folder.<br>

# Installation for Python, Jupyter Notebook, and Google Colab

First install the new functions as follows with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/delta_method.git --upgrade
```

Next import the delta_method and parametric_bootstrap functions as follows in your notebook or python code:<br>
```
from delta_method import delta_method, parametric_bootstrap
```

As an alternative, you can also download delta_method.py from this github repository (https://github.com/gjpelletier/delta_method) and add both functions to your own project.<br>

# Syntax

SYNTAX:

-	d = delta_method(pcov,popt,x_new,f,x,y,alpha)   
- b = parametric_bootstrap(popt,x_new,f,x,y,alpha,trials)

INPUTS:

- popt = optimum best-fit parameters of the regression function
- pcov = variance-covariance matrix of the model parameters
- x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
- f = user-defined regression function to predict y-values given inputs of x-values and parameters. The Python version requires x as the first argument, and parameters as separate arguments after x. The MATLAB version requires parameters together in a vector as the first argument before x. See the example scripts for reference.
- x = observed x
- y = observed y
- alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
- trials = number of trials for the bootstrap Monte Carlo

OUTPUTS:

d and b are output structures (MATLAB) or dictionaries (Python) that contain the following output variables

- lwr_conf: lower confidence interval for each value in x_new
- upr_conf: upper confidence interval for each value in x_new
- lwr_pred: lower prediction interval for each value in x_new
- upr_pred: upper prediction interval for each value in x_new

In addition, the parametric_boostrap output includes the following:

- popt_lwr_conf: lower confidence interval for each parameter
- popt_upr_conf: upper confidence interval for each parameter
- popt_b: bootstrap trials of optimum best-fit parameter values (trials x nparam)
- f_hat_b: bootstrap trials of new 'predicted' y values at each x_new (trials x n_new)
- y_hat_b: bootstrap trials of new 'observed' y values at each x_new (trials x n_new)
- qnorm: 2-tailed normal distribution score at alpha
- rq: ratio of t-score to normal-score for unbiasing

Both functions also include output of the following regression statistics:

- SST: Sum of Squares Total
- SSR: Sum of Squares Regression
- SSE: Sum of Squares Error
- MSR: Mean Square Regression
- MSE: Mean Square Error of the residuals
- syx: standard error of the regression
- nobs: number of observations
- nparam: number of parameters
- df: degrees of freedom = nobs-nparam
- qt: 2-tailed t-statistic at alpha
- Fstat: F-statistic = MSR/MSE
- dfn: degrees of freedom for the numerator of the F-test = nparam-1
- dfd: degrees of freedom for the denominator of the F-test = nobs-nparam
- pvalue: signficance level of the regression from the probability of the F-test
- rsquared: r-squared = SSR/SST
- adj_rsquared: adjusted squared

# Example for MATLAB

An example showing how to use the new delta_method and parametric_bootstrap functions is provided in this matlab script:

https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.m

# Example for Python, Jupyter Notebook, and Google Colab

An example showing how to use the new delta_method and parametric_bootstrap functions is provided in this Jupyter Notebook:

https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.ipynb

# Acknowledgement

The methods used in the delta_method and parametric_bootstrap functions and examples follow the methods described in an online lecture by Dr. Julien Chiquet (École Polytechnique/Université Paris-Saclay/AgroParisTech/INRAE) available at the following link:<br>
https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
