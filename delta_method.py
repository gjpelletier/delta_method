# -*- coding: utf-8 -*-

__version__ = "1.0.43"

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
    - 'rmse': root mean squared error
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
    rmse = np.sqrt(SSE / nobs)                  # root mean squared error
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
            'rmse': rmse,
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
    - 'rmse': root mean squared error
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
    rmse = np.sqrt(SSE / nobs)                  # root mean squared error
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
            'rmse': rmse,
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
    
def kdeplot(
    x, y,
    ax=None,
    threshold=0.001,
    scale_kde=True,
    fill=True,
    color=None,
    cmap='turbo',
    cbar=True,
    cbar_fontsize=10,
    cbar_fmt='%.2f',
    grid_size=200,
    levels=None,
    num_levels=None,
    linewidths=1,
    linestyles='solid',
    clabel=False,
    clabel_fontsize=8,
    clabel_fmt='%.2f',
    **kwargs
):
    """
    Add a scaled KDE plot as contourf to a matplotlib figure
    by Greg Pelletier (gjpelletier@gmail.com)

    Parameters:
    - x, y: 1D arrays of data points
    - ax: matplotlib Axes object (optional). If None, uses current axes.
    - threshold: float, values below this threshold (relative to max KDE) are masked (default 0.001)
    - scale_kde: bool, whether to scale KDE values to [0, 1] (default True)
    - fill: bool, whether to use contourf (True) or contour (False)
    - color: colors of the levels, i.e. the lines for contour and the areas for contourf (default None))
    - cmap: str, colormap name (default 'turbo')
    - cbar: bool, whether to show a colorbar for the plot (default True if cmap is used)
    - cbar_fontsize: font size to use for colorbar label
    - cbar_fmt: string format of colorbar tick labels (default '%.2f')
    - grid_size: int, resolution of meshgrid (default 200)
    - levels: int, list, or array-like, number and positions of the contour lines / regions
    - num_levels: int, number of discrete color levels (default 11)
    - linewidths: float, contour line widths if fill=False (default 1)
    - linestyles: contour line style if fill=False  
        'solid' (default) 'dashed', 'dashdot', 'dotted'
    - clabel: bool, whether to add labels to contour lines, used if fill=False
    - clabel_fontsize: float, font size for contour line labels (default 8),
    - clabel_fmt: string format of contour line labels (default '%.2f')
    - kwargs: additional keyword arguments passed to plt.contourf
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import FormatStrFormatter

    # Convert inputs to 1D arrays and remove NaNs
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x and y must be broadcastable to the same shape. Got shapes {x.shape} and {y.shape}.")

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must contain at least one non-NaN value after filtering.")

    if levels==None:
        if num_levels==None:
            if threshold<0.05:
                num_levels=21
            elif threshold<0.1:
                num_levels=20
            else:
                num_levels=19
        elif isinstance(num_levels, int) & (num_levels<=1 or num_levels>256):
            num_levels = 20
        elif not isinstance(num_levels, int):
            num_levels = 20

    if ax is None:
        ax = plt.gca()

    # Create meshgrid
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    # Compute KDE
    kde = gaussian_kde(np.vstack([x, y]))
    z = kde(grid_coords).reshape(xx.shape)

    # Scale KDE to [0, 1] if requested
    if scale_kde:
        z = (z - z.min()) / (z.max() - z.min())

    # Apply threshold mask
    z_max = z.max()
    z_masked = np.where(z < threshold * z_max, np.nan, z)

    # Define discrete levels
    if levels==None:
        levels = np.linspace(threshold * z_max, z_max, num_levels)

    # Use either the colors or cmap
    if color==None:
        cmap = plt.get_cmap(cmap, num_levels)
    else:
        cmap = None

    if fill:
        contour = ax.contourf(xx, yy, z_masked, 
            levels=levels, colors=color, cmap=cmap, **kwargs)
    else:
        contour = ax.contour(xx, yy, z_masked, 
            levels=levels, colors=color, cmap=cmap, 
            linewidths=linewidths, linestyles=linestyles, **kwargs)
        if clabel:
            # Add labels to the contour lines
            plt.clabel(contour, inline=True, fontsize=clabel_fontsize, fmt=clabel_fmt)

    # add colorbar
    if cbar and cmap != None:
        if scale_kde:
            if levels==None:
                levels = np.linspace(threshold, 1.0, num_levels)
            if num_levels<22:
                cbar = plt.colorbar(contour, ax=ax, ticks=levels)
            else:
                cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Scaled KDE (0–1)', fontsize=cbar_fontsize)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_fmt))
        else:
            cbar = plt.colorbar(contour, ax=ax, label='KDE')
            cbar.set_label('KDE', fontsize=cbar_fontsize)

    return contour
