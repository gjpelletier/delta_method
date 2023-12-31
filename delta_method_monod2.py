# -*- coding: utf-8 -*-
"""delta_method_monod2_v27.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q-doC1ijwFKG6MteV6pJmQJyf2bhRsZ7

# Nonlinear regression using scipy or lmfit combined with using the delta-method to estimate confidence intervals and prediction intervals

**An example using a 2-parameter Monod function**

by Greg Pelletier (gjpelletier@gmail.com)

This script uses the python package scipy or lmfit to find the optimum parameters and the variance-covariance of the parameters for nonlinear regression. We also include a new function using the delta-method, to extend beyond the capabilities of scipy or lmfit, to estimate confidence intervals for predicted values, and prediction intervals for new data, using the nonlinear regression fit.

The first step is to use scipy or lmfit to find the best-fit values and the variance-covariance matrix of the model parameters. The user may specify any expression for the nonlinear regression model.

The second step is to estimate the confidence intervals and prediction intervals using a new python function that applies the delta-method. The delta-method is described in detail in Section 5.1 of this online lecture:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#the-delta-method

In this example we use a 2-parameter Monod function (Michaelis-Menten) to fit an enzymology data set (https://rforbiochemists.blogspot.com/2015/05/plotting-and-fitting-enzymology-data.html)

___

First we need to install lmfit if it is not already installed:
"""

pip install lmfit

"""Next we import the necessary python packages:"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from numpy import exp, linspace, sin
import lmfit as fit
from lmfit.models import ExpressionModel
import scipy.optimize as opt

"""Next we will data from an enzyme study, where x= substrate (mmol), and y= velocity (nmmol/sec)"""

x = np.array([0,1,2,5,8,12,30,50])
y = np.array([0,11.1,25.4,44.8,54.5,58.2,72,60.1])

"""_____
**Step 1: Using lmfit to find the optimum parameters and the variance-covarance matrix of the parameters**

Next we will use lmfit to find the optimum parameters (popt) and the variance-covariance matrix of the model parameters (pcov). Later we will show how to use scipy instead of lmfit for this step. The advantage of using lmfit is that it provides a concise report of many useful regression statistics.

We will use the ExpressionModel function of lmfit to specify the 2-parameter Monod function that we want to fit to our data. ExpressionModel allows the user to define any function for the model as described at this link: https://lmfit.github.io/lmfit-py/builtin_models.html#user-defined-models
"""

mod = ExpressionModel('(Vm * x)/(Km + x)')

"""The next step is to define the initial values for the parameters Vm and Km of the model function"""

Vm_init = 60
Km_init = 5

"""Now we can build the set of the initial parameter values as follows:

"""

pars = fit.Parameters()
pars['Vm'] = fit.Parameter(name='Vm', value=Vm_init, min=-np.inf, max=np.inf)
pars['Km'] = fit.Parameter(name='Km', value=Km_init, min=-np.inf, max=np.inf)

"""Now we are ready to run lmfit to find the best-fit parameters and variance covariance matrix of the model. The output results will be stored in "out"
"""

out = mod.fit(y, pars, x=x)

"""We can print out a summary of the model fit results as follows:"""

print(out.fit_report())

"""The optimum best-fit values of the model parameters (popt) can now be extracted from the lmfit results as follows:"""

popt=np.array(out.params)
popt

"""The variance-covariance matrix of the model parameters (pcov) can be extracted as follows:"""

pcov = out.covar
pcov

"""_____
**Step 2: Using the delta-method to find the confidence intervals and prediction intervals**

Now we have almost everything we need in order to estimate the confidence intervals and prediction intervals of the model using the delta-method. Before doing that, we need to define a function that will apply the delta-method as follows:
"""

def delta_method(pcov,popt,x_new,f,x,y,alpha):
    # - - -
    # Function to calculate the confidence interval and prediction interval
    # for any user-defined regression function using the delta-method
    # as described in Sec 5.1 of the following online statistics lecture:
    # https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
    #
    # Greg Pelletier (gjpelletier@gmail.com)
    # - - -
    # INPUT
    # pcov = variance-covariance matrix of the model parameters (e.g. from scipy or lmfit)
    # popt = optimum best-fit parameters of the regression function (e.g. from scipy or lmfit)
    # x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
    # f = user-defined regression lambda function to predict y given inputs of parameters and x values (e.g. observed x or x_new)
    # 	For example, if using the 3-parameter nonlinear regression exponential threshold function, then
    # 	f = lambda param,xval : param[0] + param[1] * exp(param[2] * xval)
    # x = observed x
    # y = observed y
    # alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    # - - -
    # OUTPUT
    # dict = dictionary of output varlables with the following keys:
    #        'popt': optimum best-fit parameter values used as input
    #        'pcov': variance-covariance matrix used as input
    #        'fstr': string of the input lambda function of the regression model
    #        'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    #        'x': observed x values used as input
    #        'y': observed y values used as input
    #        'yhat': predicted y at observed x values
    #        'x_new': new x-values used as input to evaluate unew predicted y_new values
    #        'y_new': new predicted y_new values at new x_new values
    #        'lwr_conf': lower confidence interval for each value in x_new
    #        'upr_conf': upper confidence interval for each value in x_new
    #        'lwr_pred': lower prediction interval for each value in x_new
    #        'upr_pred': upper prediction interval for each value in x_new
    #        'grad_new': derivative gradients at x_new (change in f(x_new) per change in each popt)
    #        'G_new': variance due to each parameter at x_new
    #        'GS_new': variance due to all parameters combined at x_new
    #        'SST': Sum of Squares Total
    #        'SSR': Sum of Squares Regression
    #        'SSE': Sum of Squares Error
    #        'MSR': Mean Square Regression
    #        'MSE': Mean Square Error of the residuals
    #        'syx': standard error of the estimate
    #        'nobs': number of observations
    #        'nparam': number of parameters
    #        'df': degrees of freedom = nobs-nparam
    #        'qt': 2-tailed t-statistic at alpha
    #        'Fstat': F-statistic = MSR/MSE
    #        'dfn': degrees of freedom for the numerator of the F-test = nparam-1
    #        'dfd': degrees of freedom for the denominator of the F-test = nobs-nparam
    #        'pvalue': signficance level of the regression from the probability of the F-test
    #        'rsquared': r-squared = SSR/SST
    #        'adj_rsquared': adjusted squared
    import numpy as np
    from scipy import stats
    import inspect
    # - - -
    # calculate predicted y_new at each x_new
    y_new = f(popt,x_new)
    # calculate derivative gradients at x_new (change in f(x_new) per change in each popt)
    grad_new = np.empty(shape=(np.size(x_new),np.size(popt)))
    h = 1e-8       # h= small change for each popt to balance truncation error and rounding error of the gradient
    for i in range(np.size(popt)):
        # make a copy of popt
        popt2 = np.copy(popt)
        # gradient forward
        popt2[i] = (1+h) * popt[i]
        y_new2 = f(popt2, x_new)
        dy = y_new2 - y_new
        dpopt = popt2[i] - popt[i]
        grad_up = dy / dpopt
        # gradient backward
        popt2[i] = (1-h) * popt[i]
        y_new2 = f(popt2, x_new)
        dy = y_new2 - y_new
        dpopt = popt2[i] - popt[i]
        grad_dn = dy / dpopt
        # centered gradient is the average gradient forward and backward
        grad_new[:,i] = (grad_up + grad_dn) / 2
    # calculate variance in y_new due to each parameter and for all parameters combined
    G_new = np.matmul(grad_new,pcov) * grad_new         # variance in y_new due to each popt at each x_new
    GS_new = np.sum(G_new,axis=1)                       # total variance from all popt values at each x_new
    # - - -
    # # lwr_conf and upr_conf are confidence intervals of the best-fit curve
    nobs = np.size(x)
    nparam = np.size(popt)
    df = nobs - nparam
    qt = stats.t.ppf(1-alpha/2, df)
    delta_f = np.sqrt(GS_new) * qt
    lwr_conf = y_new - delta_f
    upr_conf = y_new + delta_f
    # - - -
    # # lwr_pred and upr_pred are prediction intervals of new observations
    yhat = f(popt,x)
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
    MSR = SSR / (np.size(popt)-1)              # mean square (regression model)
    Fstat = MSR / MSE           # F statistic
    dfn = np.size(popt) - 1    # df numerator = degrees of freedom for model = number of model parameters - 1
    dfd = df                    # df denomenator = degrees of freedom of the residual = df = nobs - nparam
    pvalue = 1-stats.f.cdf(Fstat, dfn, dfd)      # p-value of F test statistic
    rsquared = SSR / SST                                                        # ordinary rsquared
    adj_rsquared = 1-(1-rsquared)*(np.size(x)-1)/(np.size(x)-np.size(popt)-1)  # adjusted rsquared
    # - - -
    # make a string of the lambda function f to save in the output dictionary
    fstr = str(inspect.getsourcelines(f)[0])
    # make the dictionary of output variables from the delta-method
    dict = {
            'popt': popt,
            'pcov': pcov,
            'fstr': fstr,
            'alpha': alpha,
            'x': x,
            'y': y,
            'yhat': yhat,
            'x_new': x_new,
            'y_new': y_new,
            'lwr_conf': lwr_conf,
            'upr_conf': upr_conf,
            'lwr_pred': lwr_pred,
            'upr_pred': upr_pred,
            'grad_new': grad_new,
            'G_new': G_new,
            'GS_new': GS_new,
            'SST': SST,
            'SSR': SSR,
            'SSE': SSE,
            'MSR': MSR,
            'MSE': MSE,
            'syx': syx,
            'nobs': nobs,
            'nparam': nparam,
            'df': df,
            'qt': qt,
            'Fstat': Fstat,
            'dfn': dfn,
            'dfd': dfd,
            'pvalue': pvalue,
            'rsquared': rsquared,
            'adj_rsquared': adj_rsquared
            }

    return dict

"""Before using the delta_method function, we first need to make a few more inputs for it as follows (NOTE: make sure that the lambda function f below matches the ExpressionModel function defined earlier):"""

# model lambda function to use for any parameters (param) or x-values (xval)
f = lambda param,xval : (param[0] * xval) / (param[1] + xval)

# evenly spaced new observations that extend from below min(x) to above max(x)
x_new = linspace(0, 50, 100)

# probability level for the prediction limits (e.g. alpha=0.05 for 95% prediction limits and 95% confidence limits)
alpha=0.05

"""Now we are ready to use the delta_method function to get its outut dictionary, which includes the predicted y_new values, and the 95% prediction limits (lwr_pred and upr_pred), and 95% confidence limits (lwr_conf and upr_conf), at alpha=0.05 for new observations (x_new), as well as other regression statistics using the optimum parameters (popt) and covariance matrix (pcov)"""

d = delta_method(pcov,popt,x_new,f,x,y,alpha)

"""Now we can plot the results of the delta_method function to estimate the prediction intervals and confidence intervals."""

# extract the output values we need for the plot from the delta-method output dictionary
y_new = d['y_new']
lwr_conf = d['lwr_conf']
upr_conf = d['upr_conf']
lwr_pred = d['lwr_pred']
upr_pred = d['upr_pred']
rsquared = d['rsquared']
pvalue = d['pvalue']

# make string values of fit stats and eqn for the plot labels
pstr = '%.2e' %pvalue
rsqstr = '%.4f' %rsquared
b1str = '%.2f' %popt[0]
b2str = '%.3f' %popt[1]
eqnstr = 'y = (' + b1str + ' * x) + (' + b2str + ' + x)'

# generate the plot
plt.figure()
plt.plot(x, y, '.', label='observations')
# observations
plt.plot(x_new, y_new, '-', label='best fit (lmfit)')
# 95% prediction interval (PI) using the delta method
plt.fill_between(x_new, lwr_pred, upr_pred,color="#d3d3d3", label='95% PI (delta method)')
# 95% confidence interval (CI) using the delta method
plt.fill_between(x_new, lwr_conf, upr_conf,color="#ABABAB", label='95% CI (delta method)')
plt.legend(loc='lower right')
plt.title('Delta-method using lmfit for parameters and covariance')
plt.xlabel('x= substrate (mM)')
plt.ylabel('y= enzyme velocity (nmol/sec)')
plt.text(15, 23, eqnstr, fontsize = 10)
plt.text(15, 18, 'r^2 = '+rsqstr+', p = '+pstr, fontsize = 10)
plt.ticklabel_format(style='plain', axis='y')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

"""We can also print out more regression statistics calculated using the delta_method function:"""

# extract the output values we need for the regression stats
pvalue = d['pvalue']
rsquared = d['rsquared']
adj_rsquared = d['adj_rsquared']
qt = d['qt']
MSE = d['MSE']
syx = d['syx']
df = d['df']

# print the regression stats
print('p-value of the regression F test = ' + '%.2e' %pvalue)
print('rsquared = ' + '%.4f' %rsquared)
print('adjusted rsquared = ' + '%.4f' %adj_rsquared)
print('2-tailed t-statistic at alpha ' + '%.2f' %alpha + ' used for confidence/prediction intervals = ' + '%.3f' %qt)
print('Mean Square Error (MSE) of the residuals = ' + '%.3e' %MSE)
print('standard error of the regression = ' + '%.3e' %syx)
print('degrees of freedom (number of observations - number of parameters) = ' + '%.0f' %df)

"""_____
**Using scipy to find the optimum parameters and the variance-covarance matrix of the parameters, combined with using the delta-method to find the confidence intervals and prediction intervals**

Next we will show how to use scipy instead of lmfit to estimate the optimum parameters (popt_) and the variance-covariance matrix (pcov_), combined with using the delta-method to estimate the prediction interval and confidence interval.
"""

# First we need to define the model function, f(x, …)for scipy.
# It must take the independent variable as the first argument,
# and the parameters to fit as separate remaining arguments.
# Make sure that this function matches the previously defined functions
# "mod" used for Expression model and "f" used for the delta method.
def f_(x, Vm, Km):
    return (Vm * x)/(Km + x)

# Next we will make a numpy array of the guesses for the initial paramter values to use with scipy
# Note that the guesses need to be approximately the same order of magnitude as the optimum values
Vm_init_ = 60
Km_init_ = 5
p_init_ = np.array([Vm_init_, Km_init_])

# Now we are ready to find best fit paramter array (popt_) and variance-covariance matrix (pcov_) from scipy
# Note that we could also specify min and max bounds for each parameter with the bounds option
popt_, pcov_ = opt.curve_fit(f_, x, y, p0=p_init_, bounds=(-np.inf,np.inf))

# Now we are ready to use the delta_method function to find the prediction interval and confidence interval
# using the scipy estimates of popt_ and pcov_ with the same x_new, f, and alpha that we defined previously
d_ = delta_method(pcov_,popt_,x_new,f,x,y,alpha)

# extract the output values from the delta-method output dictionary
y_new_ = d_['y_new']
lwr_conf_ = d_['lwr_conf']
upr_conf_ = d_['upr_conf']
lwr_pred_ = d_['lwr_pred']
upr_pred_ = d_['upr_pred']
rsquared_ = d_['rsquared']
pvalue_ = d_['pvalue']

# make string values of fit stats and eqn for the plot labels
pstr_ = '%.2e' %pvalue_
rsqstr_ = '%.4f' %rsquared_
b1str_ = '%.2f' %popt_[0]
b2str_ = '%.3f' %popt_[1]
eqnstr = 'y = (' + b1str_ + ' * x) + (' + b2str_ + ' + x)'

# plot the results
plt.figure()
plt.plot(x, y, '.', label='observations')
# plt.plot(x_new, y_new, '--k', label='best-fit (lmfit)',linewidth=4)    # lmfit solution of best fit
plt.plot(x_new, y_new_, '-', label='best-fit (scipy)')     # scipy solution of best fit
# 95% prediction limits
plt.fill_between(x_new, d_['lwr_pred'], d_['upr_pred'],color="#d3d3d3", label='95% PI (delta method)')
# 95% confidence limits
plt.fill_between(x_new, d_['lwr_conf'], d_['upr_conf'],color="#ABABAB", label='95% CI (delta method)')
plt.legend(loc='lower right')
plt.title('Delta-method using scipy for parameters and covariance')
plt.xlabel('x= substrate (mM)')
plt.ylabel('y= enzyme velocity (nmol/sec)')
plt.text(15, 23, eqnstr, fontsize = 10)
plt.text(15, 18, 'r^2 = '+rsqstr_ +', p = '+ pstr_, fontsize = 10)
plt.ticklabel_format(style='plain', axis='y')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))