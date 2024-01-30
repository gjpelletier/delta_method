function result = parametric_bootstrap(popt,x_new,f,x,y,alpha,trials)

% version 1.0.29
% - - -
% Function to calculate the confidence interval and prediction interval
% for any user-defined regression function using the parametric bootstrap method
% as described in Sec 5.1 of the following online statistics lecture:
% https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
%
% Greg Pelletier (gjpelletier@gmail.com)
% - - -
% INPUT
% popt = optimum best-fit parameter values (e.g. from MATLAB nlinfit)
% x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
% f = user-defined regression @ function to predict y given inputs of parameters and x values (e.g. observed x or x_new)
% 	For example, if using the 4-parameter sigmoid function, then
% 	f = @(param,xval) (param(1)-param(4)) ./ (1+exp(-param(2) .* (xval-param(3))))+param(4)
% x = observed x
% y = observed y
% alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
% trials = number of iteration trials for Monte Carlo simulation
% - - -
% OUTPUT
% result = structure with the following output variables:
%        'popt': optimum best-fit parameter values used as input
%        'popt_lwr_conf': lower confidence interval for each parameter
%        'popt_upr_conf': upper confidence interval for each parameter
%        'popt_b': bootstrap trials of optimum best-fit parameter values (trials x nparam)
%        'f_hat_b': bootstrap trials of new 'predicted' y values at each x_new (trials x n_new)
%        'y_hat_b': bootstrap trials of new 'observed' y values at each x_new (trials x n_new)
%        'fstr': string of the input lambda function of the regression model
%        'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
%        'trials': number of trials for the bootstrap Monte Carlo
%        'x': observed x values used as input
%        'y': observed y values used as input
%        'yhat': reference predicted y at observed x values using input popt
%        'x_new': new x-values used as input to evaluate new predicted y_new values
%        'y_new': reference new predicted y_new values at new x_new values using input popt
%        'lwr_conf': lower confidence interval for each value in x_new
%        'upr_conf': upper confidence interval for each value in x_new
%        'lwr_pred': lower prediction interval for each value in x_new
%        'upr_pred': upper prediction interval for each value in x_new
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
%        'qnorm': 2-tailed normal distribution score at alpha
%        'rq': ratio of t-score to normal-score for unbiasing
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
if size(x,1)>1
	x = permute(x,[2 1]);
end
if size(y,1)>1
	y = permute(y,[2 1]);
end
if size(x_new,1)>1
	x_new = permute(x_new,[2 1]);
end

% calculate predicted y_new at each x_new
y_new = f(popt,x_new);
% some things we need for the bootstrap
nobs = length(x);
nparam = length(popt);
n_new = length(x_new);
df = nobs - nparam;
qt = tinv(1-alpha/2, df);
qnorm = norminv(1-alpha/2);
rq = qt/qnorm; 							% ratio of t-score to normal-score for unbiasing
yhat = f(popt,x);
SSE = sum((y-yhat) .^ 2);               % sum of squares (residual error)
MSE = SSE ./ df;                        % mean square (residual error)
syx = sqrt(MSE);                        % std error of the estimate
beta_hat = popt;               			% reference optimum parameters
y_hat_ref = f(beta_hat, x);    			% reference predicted y_hat at x
f_new = f(beta_hat,x_new);     			% reference predicted y_new at x_new
% - - -
% Monte Carlo simulation
res_f_hat = zeros(trials,n_new);
res_y_hat = zeros(trials,n_new);
res_popt_b = zeros(trials,nparam);
disp('Running bootstrap Monte Carlo iterations...');
for i=1:trials
	y_b = y_hat_ref + syx .* normrnd(0,1,[1,nobs]);
	popt_b = nlinfit(x,y_b,f,popt);
	f_b = f(popt_b, x_new);
	res_popt_b(i,:) = popt_b;
	res_f_hat(i,:) = f_b;
	res_y_hat(i,:) = f_b + normrnd(0,syx,[1,1]);
end
disp('Parametric bootstrap is completed');
% - - -
% Monte Carlo results summary
% mc_x = x_new
% mc_f = f_new
% un-biased quantiles for confidence intervals and prediction intervals
mc_lwr_conf = f_new + rq .* (quantile(res_f_hat, alpha/2, 1) - f_new);
mc_upr_conf = f_new + rq .* (quantile(res_f_hat, 1-alpha/2, 1) - f_new);
mc_lwr_pred = f_new + rq .* (quantile(res_y_hat, alpha/2, 1) - f_new);
mc_upr_pred = f_new + rq .* (quantile(res_y_hat, 1-alpha/2, 1) - f_new);
% un-biased quantiles for confidence intervals for parameters
mc_popt_lwr_conf = beta_hat + rq .* (quantile(res_popt_b, alpha/2, 1) - beta_hat);
mc_popt_upr_conf = beta_hat + rq .* (quantile(res_popt_b, 1-alpha/2, 1) - beta_hat);
% - - -
% optional additional outputs of regression statistics
SST = sum(y .^ 2) - sum(y) .^ 2 ./ nobs;  	% sum of squares (total)
SSR = SST - SSE;                            % sum of squares (regression model)
MSR = SSR / (length(popt)-1);              	% mean square (regression model)
Fstat = MSR / MSE;           				% F statistic
dfn = length(popt) - 1;    					% df numerator = degrees of freedom for model = number of model parameters - 1
dfd = df;                    				% df denomenator = degrees of freedom of the residual = df = nobs - nparam
pvalue = 1 - fcdf(Fstat, dfn, dfd);			% p-value of F test statistic 
rsquared = SSR / SST;                                                        	% ordinary rsquared
adj_rsquared = 1-(1-rsquared) .* (length(x)-1) ./ (length(x)-length(popt)-1);  	% adjusted rsquared
% - - -
% put result into an output structure
result = [];
result.popt = popt;
result.popt_lwr_conf = mc_popt_lwr_conf;
result.popt_upr_conf = mc_popt_upr_conf;
result.popt_b = res_popt_b;
result.f_hat_b = res_f_hat;
result.y_hat_b = res_y_hat;
result.trials = trials;
result.alpha = alpha;
result.x_new = x_new';
result.y_new = y_new';
result.lwr_conf = mc_lwr_conf';
result.upr_conf = mc_upr_conf';
result.lwr_pred = mc_lwr_pred';
result.upr_pred = mc_upr_pred';
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

