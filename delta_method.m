function result = delta_method(pcov,popt,x_new,f,x,y,alpha)

% version 1.0.30
% - - -
% Function to calculate the confidence interval and prediction interval
% for any user-defined regression function using the delta-method
% as described in Sec 5.1 of the following online statistics lecture:
% https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
%
% Greg Pelletier (gjpelletier@gmail.com)
% - - -
% INPUT
% pcov = variance-covariance matrix of the model parameters (e.g. from MATLAB nlinfit)
% popt = optimum best-fit parameters of the regression function (e.g. from MATLAB nlinfit)
% x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
% f = user-defined regression @ function to predict y given inputs of parameters and x values (e.g. observed x or x_new)
% 	For example, if using the 4-parameter sigmoid function, then
% 	f = @(param,xval) (param(1)-param(4)) ./ (1+exp(-param(2) .* (xval-param(3))))+param(4)
% x = observed x
% y = observed y
% alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
% - - -
% OUTPUT
% result = structure with the following output variables:
%        'popt': optimum best-fit parameter values used as input
%        'pcov': variance-covariance matrix used as input
%        'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
%        'x': observed x values used as input
%        'y': observed y values used as input
%        'yhat': predicted y at observed x values
%        'x_new': new x-values used as input to evaluate unew predicted y_new values
%        'y_new': new predicted y_new values at new x_new values
%        'lwr_conf': lower confidence interval for each value in x_new
%        'upr_conf': upper confidence interval for each value in x_new
%        'lwr_pred': lower prediction interval for each value in x_new
%        'upr_pred': upper prediction interval for each value in x_new
%        'grad_new': derivative gradients at x_new (change in f(x_new) per change in each popt)
%        'G_new': variance due to each parameter at x_new
%        'GS_new': variance due to all parameters combined at x_new
%        'SST': Sum of Squares Total
%        'SSR': Sum of Squares Regression
%        'SSE': Sum of Squares Error
%        'MSR': Mean Square Regression
%        'MSE': Mean Square Error of the residuals
%        'syx': standard error of the estimate
%        'nobs': number of observations
%        'nparam': number of parameters
%        'df': degrees of freedom = nobs-nparam
%        'qt': 2-tailed t-statistic at alpha
%        'Fstat': F-statistic = MSR/MSE
%        'dfn': degrees of freedom for the numerator of the F-test = nparam-1
%        'dfd': degrees of freedom for the denominator of the F-test = nobs-nparam
%        'pvalue': signficance level of the regression from the probability of the F-test
%        'rsquared': r-squared = SSR/SST
%        'adj_rsquared': adjusted squared

ctrl = isvector(x) & isreal(x) & ~any(isnan(x)) & ~any(isinf(x));
if ~ctrl
  error('Check x: it needs be a vector of real numbers with no infinite or nan values!')
end
ctrl = isvector(y) & isreal(y) & ~any(isnan(y)) & ~any(isinf(y));
if ~ctrl
  error('Check y: it needs be a vector of real numbers with no infinite or nan values!')
end
ctrl = length(x) == length(y);
if ~ctrl
  error('length(x) has to be equal to length(y)!')
end
ctrl = isvector(x_new) & isreal(x_new) & ~any(isnan(x_new)) & ~any(isinf(x_new));
if ~ctrl
  error('Check x_new: it needs be a vector of real numbers with no infinite or nan values!')
end
ctrl = isvector(popt) & isreal(popt) & ~any(isnan(popt)) & ~any(isinf(popt));
if ~ctrl
  error('Check popt: it needs be a vector of real numbers with no infinite or nan values!')
end
if size(x,1)>1
	x = x';
end
if size(y,1)>1
	y = y';
end
if size(x_new,1)>1
	x_new = x_new';
end
if size(popt,1)>1
	popt = popt';
end
ctrl = length(popt)==size(pcov,1) & length(popt)==size(pcov,2) & ndims(pcov)==2;
if ~ctrl
  error('pcov must be a square matrix with dimensions length(popt) x length(popt)!')
end


% calculate predicted y_new at each x_new
y_new = f(popt,x_new);
% calculate derivative gradients at x_new (change in f(x_new) per change in each popt)
grad_new = nan(length(x_new),length(popt));
h = 1e-8;       % h= small change for each popt to balance truncation error and rounding error of the gradient
for i=1:length(popt)
	% make a copy of popt
	popt2 = popt;
	% gradient forward
	popt2(i) = (1+h) * popt(i);
	y_new2 = f(popt2, x_new);
	dy = y_new2 - y_new;
	dpopt = popt2(i) - popt(i);
	grad_up = dy / dpopt;
	% gradient backward
	popt2(i) = (1-h) * popt(i);
	y_new2 = f(popt2, x_new);
	dy = y_new2 - y_new;
	dpopt = popt2(i) - popt(i);
	grad_dn = dy / dpopt;
	% centered gradient is the average gradient forward and backward
	grad_new(:,i) = (grad_up + grad_dn) ./ 2;
end
% calculate variance in y_new due to each parameter and for all parameters combined
G_new = (grad_new * pcov) .* grad_new;    	% variance in y_new due to each popt at each x_new
GS_new = sum(G_new,2);                 		% total variance from all popt values at each x_new
% - - -
% lwr_conf and upr_conf are confidence intervals of the best-fit curve
nobs = length(x);
nparam = length(popt);
df = nobs - nparam;
qt = tinv(1-alpha/2, df);
delta_f = sqrt(GS_new) .* qt;
lwr_conf = y_new' - delta_f;
upr_conf = y_new' + delta_f;
% lwr_pred and upr_pred are prediction intervals of new observations
yhat = f(popt,x);
SSE = sum((y-yhat) .^ 2);                 % sum of squares (residual error)
MSE = SSE ./ df;                          % mean square (residual error)
syx = sqrt(MSE);                          % std error of the estimate
delta_y = sqrt(GS_new + MSE) .* qt;
lwr_pred = y_new' - delta_y;
upr_pred = y_new' + delta_y;
% - - -
% optional additional outputs of regression statistics
SST = sum(y .^ 2) - sum(y) .^ 2 ./ nobs;  	% sum of squares (total)
SSR = SST - SSE;                            % sum of squares (regression model)
MSR = SSR / (length(popt)-1);              	% mean square (regression model)
Fstat = MSR / MSE;           				% F statistic
dfn = length(popt) - 1;    					% df numerator = degrees of freedom for model = number of model parameters - 1
dfd = df;                    				% df denomenator = degrees of freedom of the residual = df = nobs - nparam
% pvalue = 1-stats.f.cdf(Fstat, dfn, dfd);    % p-value of F test statistic
pvalue = 1 - fcdf(Fstat, dfn, dfd);			% p-value of F test statistic 
rsquared = SSR / SST;                                                        	% ordinary rsquared
adj_rsquared = 1-(1-rsquared) .* (length(x)-1) ./ (length(x)-length(popt)-1);  	% adjusted rsquared
% - - -
% put result into an output structure
result = [];
result.popt = popt;
result.pcov = pcov;
result.alpha = alpha;
result.x_new = x_new';
result.y_new = y_new';
result.lwr_conf = lwr_conf;
result.upr_conf = upr_conf;
result.lwr_pred = lwr_pred;
result.upr_pred = upr_pred;
result.grad_new = grad_new;
result.G_new = G_new;
result.GS_new = GS_new;
result.x = x';
result.y = x';
result.yhat = yhat';
result.nobs = nobs;
result.nparam = nparam;
result.SSE = SSE;
result.MSE = MSE;
result.syx = syx;
result.SST = SST;
result.SSR = SSR;
result.MSR = MSR;
result.Fstat = Fstat;
result.qt = qt;
result.df = df;
result.dfn = dfn;
result.dfd = dfd;
result.pvalue = pvalue;
result.rsquared = rsquared;
result.adj_rsquared = adj_rsquared;

