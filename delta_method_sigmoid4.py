# -*- coding: utf-8 -*-
"""delta_method_sigmoid4_v28b12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ofincp0WnpjBE-G3oQ2C24PCuQkdy94z

# Nonlinear regression using scipy or lmfit combined with using the delta-method or parametric bootstrap to estimate confidence intervals and prediction intervals

**An example using a 4-parameter logistic function with a sigmoid shape**

by Greg Pelletier (gjpelletier@gmail.com)

This script uses the python package scipy or lmfit to find the optimum parameters and the variance-covariance of the parameters for nonlinear regression. We also include new functions using either the delta-method or parametric bootstrap, to extend beyond the capabilities of scipy or lmfit, to estimate confidence intervals for predicted values, and prediction intervals for new data, using the nonlinear regression fit.

The first step is to use scipy or lmfit to find the best-fit values and the variance-covariance matrix of the model parameters. The user may specify any expression for the nonlinear regression model.

The second step is to estimate the confidence intervals and prediction intervals using new python functions that apply either the delta-method or parametric bootstrap as described in this online lecture:

https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#confidence-intervals-and-prediction-intervals

In this example we use a 4-parameter logistic function with a sigmoid shape to fit an observed data set. The data set that we use provided by the R base package datasets, and consist of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA.

___

First we need to install lmfit if it is not already installed:
"""

pip install lmfit

"""Next we import the necessary python packages:"""

# Commented out IPython magic to ensure Python compatibility.
# change the next line to auto istead of inline if using ipython
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from numpy import exp, linspace, sin
import lmfit as fit
from lmfit.models import ExpressionModel
import scipy.optimize as opt

"""Next we will use an example provided by the R base package datasets, and consist of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA. In this example, x= duration of the eruption (minutes), and y= waiting time between eruptions (minutes)."""

x = np.array([3.6,1.8,3.333,2.283,4.533,2.883,4.7,3.6,1.95,4.35,1.833,3.917,
              4.2,1.75,4.7,2.167,1.75,4.8,1.6,4.25,1.8,1.75,3.45,3.067,4.533,3.6,1.967,
              4.083,3.85,4.433,4.3,4.467,3.367,4.033,3.833,2.017,1.867,4.833,1.833,4.783,
              4.35,1.883,4.567,1.75,4.533,3.317,3.833,2.1,4.633,2,4.8,4.716,1.833,4.833,
              1.733,4.883,3.717,1.667,4.567,4.317,2.233,4.5,1.75,4.8,1.817,4.4,4.167,4.7,
              2.067,4.7,4.033,1.967,4.5,4,1.983,5.067,2.017,4.567,3.883,3.6,4.133,4.333,
              4.1,2.633,4.067,4.933,3.95,4.517,2.167,4,2.2,4.333,1.867,4.817,1.833,4.3,
              4.667,3.75,1.867,4.9,2.483,4.367,2.1,4.5,4.05,1.867,4.7,1.783,4.85,3.683,
              4.733,2.3,4.9,4.417,1.7,4.633,2.317,4.6,1.817,4.417,2.617,4.067,4.25,1.967,
              4.6,3.767,1.917,4.5,2.267,4.65,1.867,4.167,2.8,4.333,1.833,4.383,1.883,4.933,
              2.033,3.733,4.233,2.233,4.533,4.817,4.333,1.983,4.633,2.017,5.1,1.8,5.033,
              4,2.4,4.6,3.567,4,4.5,4.083,1.8,3.967,2.2,4.15,2,3.833,3.5,4.583,2.367,5,
              1.933,4.617,1.917,2.083,4.583,3.333,4.167,4.333,4.5,2.417,4,4.167,1.883,
              4.583,4.25,3.767,2.033,4.433,4.083,1.833,4.417,2.183,4.8,1.833,4.8,4.1,
              3.966,4.233,3.5,4.366,2.25,4.667,2.1,4.35,4.133,1.867,4.6,1.783,4.367,
              3.85,1.933,4.5,2.383,4.7,1.867,3.833,3.417,4.233,2.4,4.8,2,4.15,1.867,
              4.267,1.75,4.483,4,4.117,4.083,4.267,3.917,4.55,4.083,2.417,4.183,2.217,
              4.45,1.883,1.85,4.283,3.95,2.333,4.15,2.35,4.933,2.9,4.583,3.833,2.083,
              4.367,2.133,4.35,2.2,4.45,3.567,4.5,4.15,3.817,3.917,4.45,2,4.283,4.767,
              4.533,1.85,4.25,1.983,2.25,4.75,4.117,2.15,4.417,1.817,4.467])
y = np.array([79,54,74,62,85,55,88,85,51,85,54,84,78,47,83,52,62,84,52,79,51,47,78,69,
              74,83,55,76,78,79,73,77,66,80,74,52,48,80,59,90,80,58,84,58,73,83,64,53,
              82,59,75,90,54,80,54,83,71,64,77,81,59,84,48,82,60,92,78,78,65,73,82,56,
              79,71,62,76,60,78,76,83,75,82,70,65,73,88,76,80,48,86,60,90,50,78,63,72,
              84,75,51,82,62,88,49,83,81,47,84,52,86,81,75,59,89,79,59,81,50,85,59,87,
              53,69,77,56,88,81,45,82,55,90,45,83,56,89,46,82,51,86,53,79,81,60,82,77,
              76,59,80,49,96,53,77,77,65,81,71,70,81,93,53,89,45,86,58,78,66,76,63,88,
              52,93,49,57,77,68,81,81,73,50,85,74,55,77,83,83,51,78,84,46,83,55,81,57,
              76,84,77,81,87,77,51,78,60,82,91,53,78,46,77,84,49,83,71,80,49,75,64,76,
              53,94,55,76,50,82,54,75,78,79,78,78,70,79,70,54,86,50,90,54,54,77,79,64,
              75,47,86,63,85,82,57,82,67,74,54,83,73,73,88,80,71,83,56,79,78,84,58,83,
              43,60,75,81,46,90,46,74])

"""_____
**Using lmfit to find the optimum parameters and the variance-covarance matrix of the parameters**

Next we will use lmfit to find the optimum parameters (popt) and the variance-covariance matrix of the model parameters (pcov). Later we will show how to use scipy instead of lmfit for this step. The advantage of using lmfit is that it provides a concise report of many useful regression statistics.

We will use the ExpressionModel function of lmfit to specify the 4-parameter logistic function with a sigmoid shape that we want to fit to our data. ExpressionModel allows the user to define any function for the model as described at this link: https://lmfit.github.io/lmfit-py/builtin_models.html#user-defined-models
"""

mod = ExpressionModel('(A-S) / ( 1 + exp(-gamma * (x - tau)) ) + S')

"""The next step is to define the initial values for the parameters of the model function. We will assume that the intial values for A=90, gamma=2, tau=2, S=50 as follows:"""

A_init = 90
gamma_init = 2
tau_init = 2
S_init = 50

"""Now we can build the set of the initial parameter values for input to lmfit as follows:"""

pars = fit.Parameters()
pars['A'] = fit.Parameter(name='A', value=A_init, min=-np.inf, max=np.inf)
pars['gamma'] = fit.Parameter(name='gamma', value=gamma_init, min=-np.inf, max=np.inf)
pars['tau'] = fit.Parameter(name='tau', value=tau_init, min=-np.inf, max=np.inf)
pars['S'] = fit.Parameter(name='S', value=S_init, min=-np.inf, max=np.inf)

"""Now we are ready to run lmfit to find the best-fit parameters and variance covariance matrix of the model. The output results will be stored in "out"
"""

out = mod.fit(y, pars, x=x)

"""We can print out a summary of the model fit results from lmfit as follows:"""

print(out.fit_report())

"""The optimum values of the model parameters (popt) from lmfit can now be extracted from the lmfit results as follows:"""

popt=np.array(out.params)
popt

"""The variance-covariance matrix of the model parameters (pcov) from lmfit can be extracted as follows:"""

pcov = out.covar
pcov

"""_____
**Using the delta-method to find the confidence intervals and prediction intervals**

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

"""Before using the delta_method function, we first need to make a few more inputs for it as follows (NOTE: make sure that the lambda function f below matches the ExpressionModel function defined above):"""

# model lambda function to use for any array of parameters (param) or x-values (xval)
f = lambda param,xval : (param[0]-param[3])/(1+exp(-param[1]*(xval-param[2])))+param[3]

# evenly spaced new observations that extend from below min(x) to above max(x)
x_new = linspace(1, 6, 100)

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
syx = d['syx']

# make string values of fit stats and eqn for the plot labels
pstr = '%.2e' %pvalue
rsqstr = '%.4f' %rsquared
b1str = '%.1f' %popt[0]
b2str = '%.2f' %popt[1]
b3str = '%.2f' %popt[2]
b4str = '%.1f' %popt[3]
syxstr = '%.4f' %syx
eqnstr = 'y = (' + b1str + '-' + b4str + ')/(1+exp(-' + b2str + '*(x-' + b3str + '))+' + b4str

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
plt.xlabel('x= eruption length (min)')
plt.ylabel('y= waiting time to next eruption (min)')
plt.text(1.0, 94, eqnstr, fontsize = 8)
plt.text(1.0, 91, 'r^2 = '+rsqstr+', p = '+pstr, fontsize = 8)
plt.text(1.0, 88, 'std err of the regression = '+ syxstr, fontsize = 8)
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

# first we will define the f, x_new, and alpha as we did before in the delta-method example
f = lambda param,xval : (param[0]-param[3])/(1+exp(-param[1]*(xval-param[2])))+param[3]
x_new = linspace(1, 6, 100)
alpha=0.05

# next we also need to define the model function, f(x, …)for scipy.
# It must take the independent variable as the first argument,
# and the parameters to fit as separate remaining arguments.
# Make sure that this function matches the previously defined functions
# "mod" used for Expression model and "f" used for the delta method.
def f_scipy(x, A, gamma, tau, S):
    return (A-S) / ( 1 + exp(-gamma * (x - tau)) ) + S

# Next we will make a numpy array of the initial parameter values to use with scipy.
# Note that the guesses need to be approximately the same order of magnitude as the optimum values
A_init = 90
gamma_init = 2
tau_init = 2
S_init = 50
p_init = np.array([A_init, gamma_init, tau_init, S_init])

# Now we are ready to find the optimum best fit paramter array (popt) and variance-covariance matrix (pcov) from scipy
# Note that we could also specify min and max bounds for each parameter with the bounds option
popt, pcov = opt.curve_fit(f_scipy, x, y, p0=p_init, bounds=(-np.inf,np.inf))

# Now we are ready to use the delta_method function to find the prediction interval and confidence interval
# using the scipy estimates of popt and pcov with the same x_new, f, and alpha that we defined previously
d = delta_method(pcov,popt,x_new,f,x,y,alpha)

# extract the output values from the delta-method output dictionary
y_new = d['y_new']
lwr_conf = d['lwr_conf']
upr_conf = d['upr_conf']
lwr_pred = d['lwr_pred']
upr_pred = d['upr_pred']
rsquared = d['rsquared']
pvalue = d['pvalue']
syx = d['syx']

# make string values of fit stats and eqn for the plot labels
pstr = '%.2e' %pvalue
rsqstr = '%.4f' %rsquared
b1str = '%.1f' %popt[0]
b2str = '%.2f' %popt[1]
b3str = '%.2f' %popt[2]
b4str = '%.1f' %popt[3]
syxstr = '%.4f' %syx
eqnstr = 'y = (' + b1str + '-' + b4str + ')/(1+exp(-' + b2str + '*(x-' + b3str + '))+' + b4str

# plot the results
plt.figure()
plt.plot(x, y, '.', label='observations')
# plt.plot(x_new, y_new, '--k', label='best-fit (lmfit)',linewidth=4)    # lmfit solution of best fit
plt.plot(x_new, y_new, '-', label='best-fit (scipy)')     # scipy solution of best fit
# 95% prediction limits
plt.fill_between(x_new, d['lwr_pred'], d['upr_pred'],color="#d3d3d3", label='95% PI (delta method)')
# 95% confidence limits
plt.fill_between(x_new, d['lwr_conf'], d['upr_conf'],color="#ABABAB", label='95% CI (delta method)')
plt.legend(loc='lower right')
plt.title('Delta-method using scipy for parameters and covariance')
plt.xlabel('x= eruption length (min)')
plt.ylabel('y= waiting time to next eruption (min)')
plt.text(1.0, 94, eqnstr, fontsize = 8)
plt.text(1.0, 91, 'r^2 = '+ rsqstr +', p = '+ pstr, fontsize = 8)
plt.text(1.0, 88, 'std err of the regression = '+ syxstr, fontsize = 8)
plt.ticklabel_format(style='plain', axis='y')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

"""_____
**Parametric bootstrap instead of the delta-method**

Parametric bootstrapping is an alternative to the delta-method. Next we will apply a parametric bootstrap to estimate the confidence intervals and prediction intervals of the nonlinear regression. We will use scipy to estimate the optimum parameters, and we will follow the example in Section 5.2 of Julien Chiquet's online lecture (https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#parametric-bootstrap-2)

First we will define a function that will perform the parameteric bootstrap
"""

def parametric_bootstrap(popt,x_new,f,f_scipy,x,y,alpha,trials):
    # - - -
    # Function to calculate the confidence interval and prediction interval
    # for any user-defined regression function using a parametric bootstrap
    # as described in Sec 5.2 of the following online statistics lecture:
    # https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
    #
    # Greg Pelletier (gjpelletier@gmail.com)
    # - - -
    # INPUT
    # popt = optimum best-fit parameters of the regression function (e.g. from scipy or lmfit)
    # x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
    # f = user-defined regression lambda function to predict y given inputs of parameters and x values (e.g. observed x or x_new)
    # 	For example, if using the 3-parameter nonlinear regression exponential threshold function, then
    # 	f = lambda param,xval : param[0] + param[1] * exp(param[2] * xval)
    # f_scipy = model function for scipy.opt_corve_fit with x as first argument and parameters as separate arguments
    # 	For example, if using the 3-parameter nonlinear regression exponential threshold function, then:
    #   def f_scipy(x, A, gamma, tau, S):
    #       return (A-S) / ( 1 + exp(-gamma * (x - tau)) ) + S
    # x = observed x
    # y = observed y
    # alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    # trials = number of trials for the bootstrap Monte Carlo
    # - - -
    # OUTPUT
    # dict = dictionary of output varlables with the following keys:
    #        'popt': optimum best-fit parameter values used as input
    #        'popt_lwr_conf': lower confidence interval for each parameter
    #        'popt_upr_conf': upper confidence interval for each parameter
    #        'popt_b': bootstrap trials of optimum best-fit parameter values (trials x nparam)
    #        'f_hat_b': bootstrap trials of new 'predicted' y values at each x_new (trials x n_new)
    #        'y_hat_b': bootstrap trials of new 'observed' y values at each x_new (trials x n_new)
    #        'fstr': string of the input lambda function of the regression model
    #        'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    #        'trials': number of trials for the bootstrap Monte Carlo
    #        'x': observed x values used as input
    #        'y': observed y values used as input
    #        'yhat': reference predicted y at observed x values using input popt
    #        'x_new': new x-values used as input to evaluate new predicted y_new values
    #        'y_new': reference new predicted y_new values at new x_new values using input popt
    #        'lwr_conf': lower confidence interval for each value in x_new
    #        'upr_conf': upper confidence interval for each value in x_new
    #        'lwr_pred': lower prediction interval for each value in x_new
    #        'upr_pred': upper prediction interval for each value in x_new
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
    #        'qnorm': 2-tailed normal distribution score at alpha
    #        'rq': ratio of t-score to normal-score for unbiasing
    #        'Fstat': F-statistic = MSR/MSE
    #        'dfn': degrees of freedom for the numerator of the F-test = nparam-1
    #        'dfd': degrees of freedom for the denominator of the F-test = nobs-nparam
    #        'pvalue': signficance level of the regression from the probability of the F-test
    #        'rsquared': r-squared = SSR/SST
    #        'adj_rsquared': adjusted squared
    import numpy as np
    from scipy import stats
    import scipy.optimize as opt
    import inspect
    # - - -
    # calculate predicted y_new at each x_new using optimum parameters
    y_new = f(popt,x_new)
    # - - -
    # some things we need for the bootstrap
    nobs = np.size(x)
    nparam = np.size(popt)
    n_new = np.size(x_new)
    df = nobs - nparam
    qt = stats.t.ppf(1-alpha/2, df)
    qnorm = stats.norm.ppf(1-alpha/2)
    rq = qt/qnorm # ratio of t-score to normal-score for unbiasing
    yhat = f(popt,x)
    SSE = np.sum((y-yhat) ** 2)                 # sum of squares (residual error)
    MSE = SSE / df                              # mean square (residual error)
    syx = np.sqrt(MSE)                          # std error of the estimate
    beta_hat = popt               # reference optimum parameters
    y_hat_ref = f(beta_hat, x)    # reference predicted y_hat at x
    f_new = f(beta_hat,x_new)     # reference predicted y_new at x_new
    # - - -
    # Monte Carlo simulation
    res_f_hat = np.zeros((trials,n_new))
    res_y_hat = np.zeros((trials,n_new))
    res_popt_b = np.zeros((trials,nparam))
    for i in range(trials):
        y_b = y_hat_ref + syx * stats.norm.rvs(size=nobs)
        popt_b, pcov_b = opt.curve_fit(f_scipy, x, y_b, p0=popt, bounds=(-np.inf,np.inf))
        f_b = f(popt_b, x_new)
        res_popt_b[i,:] = popt_b
        res_f_hat[i,:] = f_b
        res_y_hat[i,:] = f_b + stats.norm.rvs(loc=0,scale=syx,size=1)
    # - - -
    # Monte Carlo results summary
    # mc_x = x_new
    # mc_f = f_new
    # un-biased quantiles for confidence intervals and prediction intervals
    mc_lwr_conf = f_new + rq * (np.quantile(res_f_hat, alpha/2, axis=0) - f_new)
    mc_upr_conf = f_new + rq * (np.quantile(res_f_hat, 1-alpha/2, axis=0) - f_new)
    mc_lwr_pred = f_new + rq * (np.quantile(res_y_hat, alpha/2, axis=0) - f_new)
    mc_upr_pred = f_new + rq * (np.quantile(res_y_hat, 1-alpha/2, axis=0) - f_new)
    # un-biased quantiles for confidence intervals for parameters
    mc_popt_lwr_conf = beta_hat + rq * (np.quantile(res_popt_b, alpha/2, axis=0) - beta_hat)
    mc_popt_upr_conf = beta_hat + rq * (np.quantile(res_popt_b, 1-alpha/2, axis=0) - beta_hat)
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
    # make the dictionary of output variables
    dict = {
            'popt': popt,
            'popt_lwr_conf': mc_popt_lwr_conf,
            'popt_upr_conf': mc_popt_upr_conf,
            'popt_b': res_popt_b,
            'f_hat_b': res_f_hat,
            'y_hat_b': res_y_hat,
            'fstr': fstr,
            'alpha': alpha,
            'trials': trials,
            'x': x,
            'y': y,
            'yhat': yhat,
            'x_new': x_new,
            'y_new': y_new,
            'lwr_conf': mc_lwr_conf,
            'upr_conf': mc_upr_conf,
            'lwr_pred': mc_lwr_pred,
            'upr_pred': mc_upr_pred,
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
            'qnorm': qnorm,
            'rq': rq,
            'Fstat': Fstat,
            'dfn': dfn,
            'dfd': dfd,
            'pvalue': pvalue,
            'rsquared': rsquared,
            'adj_rsquared': adj_rsquared
            }

    return dict

"""Next we will use the new function to perform the parametric bootstrap"""

# first we will define the f, x_new, and alpha as we did before in the delta-method example
f = lambda param,xval : (param[0]-param[3])/(1+exp(-param[1]*(xval-param[2])))+param[3]
x_new = linspace(1, 6, 100)
alpha=0.05

# next we also need to define the model function, f(x, …)for scipy.
# It must take the independent variable as the first argument,
# and the parameters to fit as separate remaining arguments.
# Make sure that this function matches the previously defined functions
def f_scipy(x, A, gamma, tau, S):
    return (A-S) / ( 1 + exp(-gamma * (x - tau)) ) + S

# next we will make a numpy array of the initial parameter values to use with scipy.
# Note that the guesses need to be approximately the same order of magnitude as the optimum values
A_init = 90
gamma_init = 2
tau_init = 2
S_init = 50
p_init = np.array([A_init, gamma_init, tau_init, S_init])

# next we find the optimum best fit paramter array (popt) and variance-covariance matrix (pcov) from scipy
# Note that we could also specify min and max bounds for each parameter with the bounds option
popt, pcov = opt.curve_fit(f_scipy, x, y, p0=p_init, bounds=(-np.inf,np.inf))

# now we are ready to use the parameteric_bootstrap function to find the prediction interval and confidence interval
# using the scipy estimates of popt with the inputs that we defined previously
trials = 10000   # number of trials for bootstrap Monte Carlo
b = parametric_bootstrap(popt,x_new,f,f_scipy,x,y,alpha,trials)

# extract the output values from the output dictionary
y_new = b['y_new']
mc_lwr_conf = b['lwr_conf']
mc_upr_conf = b['upr_conf']
mc_lwr_pred = b['lwr_pred']
mc_upr_pred = b['upr_pred']
rsquared = b['rsquared']
pvalue = b['pvalue']
syx = b['syx']

# make string values of fit stats and eqn for the plot labels
pstr = '%.2e' %pvalue
rsqstr = '%.4f' %rsquared
b1str = '%.1f' %popt[0]
b2str = '%.2f' %popt[1]
b3str = '%.2f' %popt[2]
b4str = '%.1f' %popt[3]
syxstr = '%.4f' %syx
eqnstr = 'y = (' + b1str + '-' + b4str + ')/(1+exp(-' + b2str + '*(x-' + b3str + '))+' + b4str

# plot the results
plt.figure()
plt.plot(x, y, '.', label='observations')
# plt.plot(x_new, y_new, '--k', label='best-fit (lmfit)',linewidth=4)    # lmfit solution of best fit
plt.plot(x_new, y_new, '-', label='best-fit (scipy)')     # scipy solution of best fit
# 95% prediction limits
plt.fill_between(x_new, mc_lwr_pred, mc_upr_pred,color="#d3d3d3", label='95% PI (bootstrap)')
# 95% confidence limits
plt.fill_between(x_new, mc_lwr_conf, mc_upr_conf,color="#ABABAB", label='95% CI (bootstrap)')
plt.legend(loc='lower right')
plt.title('Bootstrap using scipy for optimum parameters')
plt.xlabel('x= eruption length (min)')
plt.ylabel('y= waiting time to next eruption (min)')
plt.text(1.0, 94, eqnstr, fontsize = 8)
plt.text(1.0, 91, 'r^2 = '+ rsqstr +', p = '+ pstr, fontsize = 8)
plt.text(1.0, 88, 'std err of the regression = '+ syxstr, fontsize = 8)
plt.ticklabel_format(style='plain', axis='y')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

"""_____
**Confidence intervals for model parameters**

We will use various methods to estimate the confidence intervals for model parameters, as described in https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html#confidence-intervals-for-the-model-parameters

Method 1. Parametric bootstrap
"""

print("Confidence intervals of parameters using parametric bootstrap")
print("")
print("Parameter\t beta_hat \t 95% confidence intervals")
print("A:       \t %.4f     \t (%.4f - %.4f)" % (popt[0], b['popt_lwr_conf'][0], b['popt_upr_conf'][0]))
print("gamma:   \t %.4f     \t (%.4f - %.4f)" % (popt[1], b['popt_lwr_conf'][1], b['popt_upr_conf'][1]))
print("tau:     \t %.4f     \t (%.4f - %.4f)" % (popt[2], b['popt_lwr_conf'][2], b['popt_upr_conf'][2]))
print("S:       \t %.4f     \t (%.4f - %.4f)" % (popt[3], b['popt_lwr_conf'][3], b['popt_upr_conf'][3]))

"""Method 2. Linearization approach"""

import scipy.stats as stats

# optimum values of each parameter from scipy
beta_hat = popt

# standard errors of each parameter are the sqrt of the diagonal of the variance-covariance matrix
se_lin = np.sqrt(np.diag(pcov))

# t-statistic and 2-sided probability Pr(>|t|)
t_stat = beta_hat/se_lin                  # t statistic
df = d["df"]                              # degrees of freedom
pval = stats.t.sf(np.abs(t_stat), df)*2   # two-sided p-value = Prob(abs(t)>tt)
qt = d["qt"]                              # t(1-alpha/2,df)

# 2-tailed confidence interals at alpha (we are using alpha=0.05 for 95% confidence intervals)
lwr_ci = popt - qt * se_lin        # lower confidence interval
upr_ci = popt + qt * se_lin        # upper confidence interval

# print out a summary
print("Confidence intervals of parameters using the linearization approach")
print("")
print("Parameter \t beta_hat \t se_lin \t Pr(>|t|) \t 95% confidence intervals")
print("A:        \t %.4f     \t %.4f   \t %.1e     \t (%.4f - %.4f)" % (popt[0], se_lin[0], pval[0], lwr_ci[0], upr_ci[0]))
print("gamma:    \t %.4f     \t %.4f   \t %.1e     \t (%.4f - %.4f)" % (popt[1], se_lin[1], pval[1], lwr_ci[1], upr_ci[1]))
print("tau:      \t %.4f     \t %.4f   \t %.1e     \t (%.4f - %.4f)" % (popt[2], se_lin[2], pval[2], lwr_ci[2], upr_ci[2]))
print("S:        \t %.4f     \t %.4f   \t %.1e     \t (%.4f - %.4f)" % (popt[3], se_lin[3], pval[3], lwr_ci[3], upr_ci[3]))