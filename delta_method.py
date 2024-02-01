# -*- coding: utf-8 -*-

__version__ = "1.0.31"

def delta_method(pcov,popt,x_new,f,x,y,alpha):

    """
    Function to calculate the confidence interval and prediction interval for any user-defined regression function using the delta-method as described in Sec 5.1 of the following online statistics lecture:
    https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
    by Greg Pelletier (gjpelletier@gmail.com)
    SYNTAX
    result = delta_method(pcov,popt,x_new,f,x,y,alpha)
    INPUT
    - pcov = variance-covariance matrix of the model parameters (e.g. from scipy or lmfit)
    - popt = optimum best-fit parameters of the regression function (e.g. from scipy or lmfit)
    - x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
    - f = model function for scipy.opt_curve_fit with x as first argument and parameters as separate arguments. For example, if using the 4-parameter sigmoid function, then
      def f(x, A, gamma, tau, S):
          return (A-S) / ( 1 + exp(-gamma * (x - tau)) ) + S
    - x = observed x
    - y = observed y
    - alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    OUTPUT
    - result = dictionary of output varlables with the following keys:
    - 'popt': optimum best-fit parameter values used as input
    - 'pcov': variance-covariance matrix used as input
    - 'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    - 'x': observed x values used as input
    - 'y': observed y values used as input
    - 'yhat': predicted y at observed x values
    - 'x_new': new x-values used as input to evaluate unew predicted y_new values
    - 'y_new': new predicted y_new values at new x_new values
    - 'lwr_conf': lower confidence interval for each value in x_new
    - 'upr_conf': upper confidence interval for each value in x_new
    - 'lwr_pred': lower prediction interval for each value in x_new
    - 'upr_pred': upper prediction interval for each value in x_new
    - 'grad_new': derivative gradients at x_new (change in f(x_new) per change in each popt)
    - 'G_new': variance due to each parameter at x_new
    - 'GS_new': variance due to all parameters combined at x_new
    - 'SST': Sum of Squares Total
    - 'SSR': Sum of Squares Regression
    - 'SSE': Sum of Squares Error
    - 'MSR': Mean Square Regression
    - 'MSE': Mean Square Error of the residuals
    - 'syx': standard error of the estimate
    - 'nobs': number of observations
    - 'nparam': number of parameters
    - 'df': degrees of freedom = nobs-nparam
    - 'qt': 2-tailed t-statistic at alpha
    - 'Fstat': F-statistic = MSR/MSE
    - 'dfn': degrees of freedom for the numerator of the F-test = nparam-1
    - 'dfd': degrees of freedom for the denominator of the F-test = nobs-nparam
    - 'pvalue': signficance level of the regression from the probability of the F-test
    - 'rsquared': r-squared = SSR/SST
    - 'adj_rsquared': adjusted squared
    """

    import numpy as np
    from scipy import stats
    import sys

    ctrl = np.isreal(x).all() and (not np.isnan(x).any()) and (not np.isinf(x).any()) and x.ndim==1
    if not ctrl:
      print('Check x: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl = np.isreal(y).all() and (not np.isnan(y).any()) and (not np.isinf(y).any()) and y.ndim==1
    if not ctrl:
      print('Check y: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl = np.isreal(x_new).all() and (not np.isnan(x_new).any()) and (not np.isinf(x_new).any()) and x_new.ndim==1
    if not ctrl:
      print('Check x_new: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl = np.isreal(popt).all() and (not np.isnan(popt).any()) and (not np.isinf(popt).any()) and popt.ndim==1
    if not ctrl:
      print('Check popt: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl =  np.size(x)==np.size(y)
    if not ctrl:
      print('Check x and y: x and y need to be the same size!','\n')
      sys.exit()
    ctrl = np.shape(pcov)[0]==np.shape(pcov)[1] and np.shape(pcov)[0]==np.size(popt)
    if not ctrl:
      print('Check pcov and popt: pcov must be a square covariance matrix for popt with dimensions length(popt) x length(popt)!!','\n')
      sys.exit()

    # - - -
    # calculate predicted y_new at each x_new
    y_new = f(x_new,*popt)
    # calculate derivative gradients at x_new (change in f(x_new) per change in each popt)
    grad_new = np.empty(shape=(np.size(x_new),np.size(popt)))
    h = 1e-8       # h= small change for each popt to balance truncation error and rounding error of the gradient
    for i in range(np.size(popt)):
        # make a copy of popt
        popt2 = np.copy(popt)
        # gradient forward
        popt2[i] = (1+h) * popt[i]
        y_new2 = f(x_new,*popt2)
        dy = y_new2 - y_new
        dpopt = popt2[i] - popt[i]
        grad_up = dy / dpopt
        # gradient backward
        popt2[i] = (1-h) * popt[i]
        y_new2 = f(x_new,*popt2)
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
    yhat = f(x,*popt)
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
    # make the dictionary of output variables from the delta-method
    result = {
            'popt': popt,
            'pcov': pcov,
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

    return result
    
def parametric_bootstrap(popt,x_new,f,x,y,alpha,trials):

    """
    Function to calculate the confidence interval and prediction interval for any user-defined regression function using a parametric bootstrap as described in Sec 5.2 of the following online statistics lecture:
    https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
    by Greg Pelletier (gjpelletier@gmail.com)
    SYNTAX
    result = parametric_bootstrap(popt,x_new,f,x,y,alpha,trials)
    INPUT
    - popt = optimum best-fit parameters of the regression function (e.g. from scipy or lmfit)
    - x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
    - f = model function for scipy.opt_curve_fit with x as first argument and parameters as separate arguments. For example, if using the 4-parameter sigmoid function, then:
      def f(x, A, gamma, tau, S):
          return (A-S) / ( 1 + exp(-gamma * (x - tau)) ) + S
    - x = observed x
    - y = observed y
    - alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    - trials = number of trials for the bootstrap Monte Carlo
    OUTPUT
    - result = dictionary of output varlables with the following keys:
    - 'popt': optimum best-fit parameter values used as input
    - 'popt_lwr_conf': lower confidence interval for each parameter
    - 'popt_upr_conf': upper confidence interval for each parameter
    - 'popt_b': bootstrap trials of optimum best-fit parameter values (trials x nparam)
    - 'f_hat_b': bootstrap trials of new 'predicted' y values at each x_new (trials x n_new)
    - 'y_hat_b': bootstrap trials of new 'observed' y values at each x_new (trials x n_new)
    - 'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
    - 'trials': number of trials for the bootstrap Monte Carlo
    - 'x': observed x values used as input
    - 'y': observed y values used as input
    - 'yhat': reference predicted y at observed x values using input popt
    - 'x_new': new x-values used as input to evaluate new predicted y_new values
    - 'y_new': reference new predicted y_new values at new x_new values using input popt
    - 'lwr_conf': lower confidence interval for each value in x_new
    - 'upr_conf': upper confidence interval for each value in x_new
    - 'lwr_pred': lower prediction interval for each value in x_new
    - 'upr_pred': upper prediction interval for each value in x_new
    - 'SST': Sum of Squares Total
    - 'SSR': Sum of Squares Regression
    - 'SSE': Sum of Squares Error
    - 'MSR': Mean Square Regression
    - 'MSE': Mean Square Error of the residuals
    - 'syx': standard error of the estimate
    - 'nobs': number of observations
    - 'nparam': number of parameters
    - 'df': degrees of freedom = nobs-nparam
    - 'qt': 2-tailed t-statistic at alpha
    - 'qnorm': 2-tailed normal distribution score at alpha
    - 'rq': ratio of t-score to normal-score for unbiasing
    - 'Fstat': F-statistic = MSR/MSE
    - 'dfn': degrees of freedom for the numerator of the F-test = nparam-1
    - 'dfd': degrees of freedom for the denominator of the F-test = nobs-nparam
    - 'pvalue': signficance level of the regression from the probability of the F-test
    - 'rsquared': r-squared = SSR/SST
    - 'adj_rsquared': adjusted squared
    """

    import numpy as np
    from scipy import stats
    import scipy.optimize as opt
    import sys

    ctrl = np.isreal(x).all() and (not np.isnan(x).any()) and (not np.isinf(x).any()) and x.ndim==1
    if not ctrl:
      print('Check x: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl = np.isreal(y).all() and (not np.isnan(y).any()) and (not np.isinf(y).any()) and y.ndim==1
    if not ctrl:
      print('Check y: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl = np.isreal(x_new).all() and (not np.isnan(x_new).any()) and (not np.isinf(x_new).any()) and x_new.ndim==1
    if not ctrl:
      print('Check x_new: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl = np.isreal(popt).all() and (not np.isnan(popt).any()) and (not np.isinf(popt).any()) and popt.ndim==1
    if not ctrl:
      print('Check popt: it needs be a vector of real numbers with no infinite or nan values!','\n')
      sys.exit()
    ctrl =  np.size(x)==np.size(y)
    if not ctrl:
      print('Check x and y: x and y need to be the same size!','\n')
      sys.exit()
    
    # - - -
    # calculate predicted y_new at each x_new using optimum parameters
    y_new = f(x_new,*popt)
    # - - -
    # some things we need for the bootstrap
    nobs = np.size(x)
    nparam = np.size(popt)
    n_new = np.size(x_new)
    df = nobs - nparam
    qt = stats.t.ppf(1-alpha/2, df)
    qnorm = stats.norm.ppf(1-alpha/2)
    rq = qt/qnorm # ratio of t-score to normal-score for unbiasing
    yhat = f(x,*popt)
    SSE = np.sum((y-yhat) ** 2)                 # sum of squares (residual error)
    MSE = SSE / df                              # mean square (residual error)
    syx = np.sqrt(MSE)                          # std error of the estimate
    beta_hat = popt               # reference optimum parameters
    y_hat_ref = f(x,*beta_hat)    # reference predicted y_hat at x
    f_new = f(x_new,*beta_hat)     # reference predicted y_new at x_new
    # - - -
    # Monte Carlo simulation
    res_f_hat = np.zeros((trials,n_new))
    res_y_hat = np.zeros((trials,n_new))
    res_popt_b = np.zeros((trials,nparam))
    for i in range(trials):
        y_b = y_hat_ref + syx * stats.norm.rvs(size=nobs)
        popt_b, pcov_b = opt.curve_fit(f, x, y_b, p0=popt, bounds=(-np.inf,np.inf))
        f_b = f(x_new,*popt_b)
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
    # make the dictionary of output variables
    result = {
            'popt': popt,
            'popt_lwr_conf': mc_popt_lwr_conf,
            'popt_upr_conf': mc_popt_upr_conf,
            'popt_b': res_popt_b,
            'f_hat_b': res_f_hat,
            'y_hat_b': res_y_hat,
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

    return result
    
