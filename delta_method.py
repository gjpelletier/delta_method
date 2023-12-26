# -*- coding: utf-8 -*-
"""delta_method.py

Example of nonlinear regression using lmfit 
combined with using the delta method 
to estimate confidence intervals and prediction intervals

by Greg Pelletier (gjpelletier@gmail.com) 12/25/2023

This colab notebook uses the python package called lmfit for nonlinear regression. 
We also introduce a new function using the delta method, to extend beyond the capabilities of lmfit, 
to estimate confidence intervals for predicted values, and prediction intervals for new data, 
using the nonlinear regression fit.

The lmfit package is used to find the best-fit values and the variance-covariance matrix 
of the model parameters. The user may specify any expression for the nonlinear regression model. 
The lmfit package is described at the following link:

https://lmfit.github.io//lmfit-py/index.html

In this example we use a 3-parameter exponential function to fit an observed data set 
for calcification rates of hard clams from Ries et al (2009) (https://doi.org/10.1130/G30210A.1)

To estimate the confidence intervals and prediction intervals, we use a new python function 
that applies the delta method. The delta method is described in detail in Section 5.1 of this online lecture:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html

___

First we need to install lmfit if it is not already installed:
"""

# uncomment the following line if you do not already have lmfit installed:
# pip install lmfit

"""Next we import the necessary python packages:"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from numpy import exp, linspace, sin
import lmfit as fit
from lmfit.models import ExpressionModel

# Uncomment the following line if you are using ipython: 
# %matplotlib auto        


"""Next we will use example observations from Ries et al (2009) for hard clams, 
where x= TA:DIC, and y= calcification rate (mmol/g/hr)"""

x = np.array([1.14705882, 1.14705882, 1.14705882, 1.14705882, 1.14705882,
       1.14705882, 1.14705882, 1.10570071, 1.10570071, 1.10570071,
       1.10570071, 1.10570071, 1.06117782, 1.06117782, 1.06117782,
       1.06117782, 1.06117782, 1.06117782, 1.06117782, 0.99613713,
       0.99613713, 0.99613713, 0.99613713, 0.99613713, 0.99613713])
y = np.array([ 1.08235921e-04,  3.33033603e-05,  1.33213441e-04,  6.66067206e-05,
        4.16292004e-05,  5.41179605e-05,  2.49775202e-05,  5.82808805e-05,
        4.99550405e-05,  3.74662803e-05,  7.49325607e-05,  2.91404403e-05,
        5.82808805e-05, -1.66516802e-05,  2.91404403e-05,  6.24438006e-05,
        4.99550405e-05,  3.74662803e-05,  4.16292004e-06, -9.57471609e-05,
       -1.37376361e-04, -1.24887601e-04, -6.66067206e-05, -7.90954807e-05,
       -7.07696407e-05])

"""We will use the ExpressionModel function of lmfit to specify the 
3-parameter exponential function that we want to fit to our data:"""

mod = ExpressionModel('b1 + b2 * exp(b3 * x)')

"""The next step is to define the initial values are ranges for the three parameters (b1, b2, b3) 
of the model function. We will assume that the intial values for b2 and b3 are zero, 
and the intial value for b1 is the 10th percentile of the observed y values. 
In other words, the intial guess for the model result is a constant value 
equal to the 10th percentile of y at all observed x values.

The 10th percentile of the observed y is calculated as follows:
"""

y10th = np.percentile(y,10)

"""Now we can build set the initial parameter values as follows:

"""

pars = fit.Parameters()
pars['b1'] = fit.Parameter(name='b1', value=y10th, min=-np.inf, max=np.inf)
pars['b2'] = fit.Parameter(name='b2', value=0, min=-np.inf, max=np.inf)
pars['b3'] = fit.Parameter(name='b3', value=0, min=-np.inf, max=np.inf)

"""Now we are ready to run lmfit to find the best-fit parameters and variance-covariance matrix of the model. 
The output results will be stored in "out"
"""

out = mod.fit(y, pars, x=x)

"""We can print out a summary of the model fit results as follows:"""

print(out.fit_report())

"""The best-fit values of the model parameters can now be extracted 
into a numpy array that we named "param" from the lmfit results as follows:"""

param=np.array(out.params)
param

"""The variance-covariance matrix of the model parameters can be extracted 
into a numpy array that we name "COVB" as follows:"""

COVB = out.covar
COVB

"""Now we have almost everything we need in order to estimate the confidence intervals  of predictions
and prediction intervals of new new observations. Before doing that, we need to define a function 
that will apply the delta method as follows:"""

def delta_method(COVB,param,x_new,f,x,y,alpha):
    # - - -
    # Function to calculate the confidence interval and prediction interval
    # for any user-defined regression function using the delta-method
    # as described in Sec 5.1 of the following online statistics lecture:
    # https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
    #
    # Greg Pelletier (gjpelletier@gmail.com)
    # - - -
    # INPUT
    # COVB = variance-covariance matrix of the model parameters (e.g. out.covar from lmfit)
    # param = best-fit parameters of the regression function
    # x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
    # f = user-defined regression lambda function to predict y given inputs if param and x values (e.g. observed x or x_new)
    # 	For example, if using the 3-parameter nonlinear regression exponential threshold function, then
    # 	f = lambda param,xval : param[0] + param[1] * exp(param[2] * xval)
    # x = observed x
    # y = observed y
    # alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    # - - -
    # OUTPUT
    # y_new = predicted y values at x_new
    # lwr_conf = lower confidence interval for each value in x_new
    # upr_conf = upper confidence interval for each value in x_new
    # lwr_pred = lower prediction interval for each value in x_new
    # upr_pred = upper prediction interval for each value in x_new
    # pvalue = signficance level of the regression from the probability of the F-test
    import numpy as np
    from scipy import stats
    # - - -
    # calculate predicted y_new at each x_new
    y_new = f(param,x_new)
    # calculate derivative gradients at x_new (change in f(x_new) per change in each param using +1ppb change in each param)
    grad_new = np.empty(shape=(np.size(x_new),np.size(param)))
    db = 1e-9       # use 1ppb change for each param to calculate the derivative gradient
    for i in range(np.size(param)):
        # gradient of each param i at x_new
        param2 = np.copy(param)
        # gradient forward
        param2[i] = (1+db) * param[i]
        y_new2 = f(param2, x_new)
        dy = y_new2 - y_new
        dparam = param2[i] - param[i]
        grad_up = dy / dparam
        # gradient backward
        param2[i] = (1-db) * param[i]
        y_new2 = f(param2, x_new)
        dy = y_new2 - y_new
        dparam = param2[i] - param[i]
        grad_dn = dy / dparam
        # average gradient forward and backward
        grad_new[:,i] = (grad_up + grad_dn) / 2
    # calculate variance in y_new due to each parameter and for all parameters combined
    G_new = np.matmul(grad_new,COVB) * grad_new         # variance in y_new due to each param at each x_new
    GS_new = np.sum(G_new,axis=1)                       # total variance from all param values at each x_new
    # - - -
    # # lwr_conf and upr_conf are confidence intervals of the best-fit curve
    nobs = np.size(x)
    nparam = np.size(param)
    df = nobs - nparam
    qt = stats.t.ppf(1-alpha/2, df)
    delta_f = np.sqrt(GS_new) * qt
    lwr_conf = y_new - delta_f
    upr_conf = y_new + delta_f
    # - - -
    # # lwr_pred and upr_pred are prediction intervals of new observations
    yhat = f(param,x)
    SSE = np.sum((y-yhat) ** 2)                 # sum of squares (residual error)
    MSE = SSE / df                              # mean square (residual error)
    syx = np.sqrt(MSE)                          # std error of the estimate
    delta_y = np.sqrt(GS_new + MSE) * qt
    lwr_pred = y_new - delta_y
    upr_pred = y_new + delta_y
    # - - -
    # optional additional outputs of regression statistics
    SST = np.sum(y **2) - np.sum(y) **2 / nobs  # sum of squares (total)
    SSR = SST - SSE                             # sum of squares (regression model)
    MSR = SSR / (np.size(param)-1)              # mean square (regression model)
    Fstat = MSR / MSE           # F statistic
    dfn = np.size(param) - 1    # df numerator = degrees of freedom for model = number of model parameters - 1
    dfd = df                    # df denomenator = degrees of freedom of the residual = df = nobs - nparam
    pvalue = 1-stats.f.cdf(Fstat, dfn, dfd)      # p-value of F test statistic
    rsquared = SSR / SST                                                        # ordinary rsquared
    adj_rsquared = 1-(1-rsquared)*(np.size(x)-1)/(np.size(x)-np.size(param)-1)  # adjusted rsquared
    # - - -
    return y_new, lwr_conf, upr_conf, lwr_pred, upr_pred, pvalue

"""Before using the delta_method function, we first need to make a few more inputs for it as follows:"""

# model lambda function to use for any param or xval
f = lambda param,xval : param[0] + param[1] * exp(param[2] * xval)

# evenly spaced new observations that extend from below min(x) to above max(x)
x_new = linspace(.95, 1.3, 100)

# probability level for the prediction limits (e.g. alpha=0.05 for 95% prediction limits and 95% confidence limits)
alpha=0.05

"""Now we are ready to use the delta_method function to find the predicted y_new values, 
and the 95% prediction limits (lwr_pred and upr_pred), 
and 95% confidence limits (lwr_conf and upr_conf), 
at alpha=0.05 for new observations (x_new)"""

y_new, lwr_conf, upr_conf, lwr_pred, upr_pred, pvalue = delta_method(COVB,param,x_new,f,x,y,alpha)

"""Now we can plot the results of lmfit, combined with the results of the delta_method function 
to estimate the prediction intervals and confidence intervals."""

# make string values of fit stats and eqn for the plot labels
pstr = '%.2e' %pvalue
rsquared = out.rsquared
rsqstr = '%.4f' %rsquared
b1str = '%.4e' %param[0]
b2str = '%.4e' %param[1]
b3str = '%.4e' %param[2]
eqnstr = 'y = '+b2str+' exp('+b3str+' * x) + '+b1str

# generate the plot
plt.figure()
plt.plot(x, y, '.', label='observations')
# observations
plt.plot(x_new, y_new, '-', label='best fit')
# 95% prediction limits using the delta method
plt.fill_between(x_new, lwr_pred, upr_pred,color="#d3d3d3", label='95% prediction interval (delta method)')
# 95% confidence limits using the delta method
plt.fill_between(x_new, lwr_conf, upr_conf,color="#ABABAB", label='95% confidence interval (delta method)')
plt.legend(loc='lower right')
plt.suptitle('Delta-method for prediction and confidence intervals')
plt.title('using hard clam data from Ries et al (2009)')
plt.xlabel('TA:DIC')
plt.ylabel('calcification rate (mmol/g/hr)')
plt.text(1.0, -.0004, eqnstr, fontsize = 9)
plt.text(1.0, -.00045, 'r^2 = '+rsqstr+', p = '+pstr, fontsize = 9)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

