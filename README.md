## Confidence and prediction intervals for nonlinear regression using the delta-method or parametric bootstrap, bivariate KDE contour plots, and bivariate quantile plots for Python and Jupyter Notebook

by Greg Pelletier (gjpelletier@gmail.com)

We introduce the following new functions to estimate confidence intervals and prediction intervals for nonlinear regression, and plot contours of bivariate Kernel Density Estimates (KDE), and bivariate quantiles of data distributions:

- **delta_method**
- **parametric_bootstrap**
- **kde_contour**
- **quantile_contour**

The first step before using the **delta_method** or **parametric_bootstrap** functions is to find the optimum parameter values and the parameter covariance matrix. This step can be done using MATLAB's nlinfit, or Python's scipy opt.curve_fit or lmfit.

The second step is to estimate the confidence intervals and prediction intervals using our new delta_method or parametric_bootstrap functions. We also show how to use the parametric_bootstrap function as an alternative to linear approximations to estimate confidence intervals of the nonlinear regression model parameters.

The **kde_contour** and **quantile_contour** functions are alternatives to a scatterplot for visualizing the distribution of two variables with a very large number of samples. The **kde_contour** function produces a bivariate KDE contour plot to visualize the joint probability density function of two continuous variables. While a scatterplot shows the individual locations of data points, a bivariate KDE contour plot focuses on the density of these points, providing a continuous representation of the data's distribution rather than just discrete points.

The **quantile_contour** function computes quantile contours of bivariate data with Guassian kernel density estimates (KDE) using normalized cumulative integration to match specified density mass thresholds, enabling quantile-based interpretation of bivariate distributions. To identify regions of high data concentration in bivariate space, this function applies a grid-based KDE using a Gaussian kernel, as implemented in the Python function scipy.stats.guassian_kde. The KDE iss evaluated over a uniform grid spanning the data domain, and the resulting density values are sorted in descending order. The function then computes the cumulative density mass by integrating over the bivariate grid cells, scaling the cumulative sum to unity. For each target quantile (e.g., 0.1, 0.5, 0.9, 0.99, 0.999), we identify the KDE threshold corresponding to the smallest density value that enclosed the desired proportion of total mass. This threshold defines a contour level whose enclosed area contains the specified fraction of the data density, while the area outside the contour corresponds to the complementary fraction. This approach aligns with a quantile matching strategy, where each contour level acts as a probabilistic boundary: for example, the 0.9 contour encloses approximately 90% of the data density, leaving 10% outside. Such contours are particularly useful for interpreting spatial or bivariate distributions of oceanographic variables, identifying core regions of biological or physical activity, and delineating anomalous or peripheral zones. Similar contour integration methods have been employed in ocean sciences and geophysical studies, including analyses of sediment transport and biogeochemical distributions (e.g., Klaassen et al., 2025;  Pourzangbar  et al 2023). These contours provide a statistically grounded framework for comparing bivariate or spatial patterns across datasets and for quantifying uncertainty in observational coverage, or comparison of data distributions and bioassay treatment levels.

## Installation for Python and Jupyter Notebook

First install the new functions as follows with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/delta_method.git --upgrade
```

Next import the delta_method and parametric_bootstrap functions as follows in your notebook or python code:<br>
```
from delta_method import delta_method, parametric_bootstrap, kde_contour
```

## Installation for MATLAB

Download the delta_method.m and parametric_boostrap.m files from this github repository (https://github.com/gjpelletier/delta_method) or MATLAB File Exchange and copy them to your working directory or session search path folder.<br>

## Syntax

SYNTAX:

- d = delta_method(pcov,popt,x_new,f,x,y,alpha)   
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
- rmse: root mean squared error
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

## Examples

### Example 1: Using delta_method and parametric_bootstrap

A detailed example showing how to use the delta_method and parametric_bootstrap functions is provided in this Jupyter Notebook:

[https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.ipynb](https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.ipynb)

and in this matlab script:

[https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.m](https://github.com/gjpelletier/delta_method/blob/main/delta_method_example.m)

In this example we use a 4-parameter logistic function with a sigmoid shape to fit an observed data set. The data set that we use provided by the R base package datasets, and consist of the waiting time between eruptions and the duration of the eruption for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA.

The resulting confidence intervals and prediction intervals of the nonlinear regression using the delta_method in this example are shown in the following figure:

<img width="1920" height="1440" alt="example_waiting_time_vs_eruption_length" src="https://github.com/user-attachments/assets/aec2ae7c-d7e1-4244-9c0a-8d5b014e8ba9" />

### Example 2: Using kde_contour and quantile_contour with delta_method to overlay nonlinear regression confidence and prediction intervals

In this example we analyze the relationship between pH and aragonite saturation (Ωara) in seawater using data from the Multistressor Observations of Coastal Hypoxia and Acidification (MOCHA) Synthesis dataset (https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0277984). We use a subset of the MOCHA data collected along the US west coast during April-September in the upper 200 meters of the ocean within 40 km of the coast. 

The nonlinear regression best fit, 95% confidence intervals, and 95% prediction intervals are shown in the red lines using the **delta_method** function

The bivariate Kernel Density Estimates (KDE) are shown in shades of blue using the **kde_contour** function. The KDE values indicate the areas with the densest numbers of data points. 

The bivariate data quantiles are shown in the black contour lines using the **quantile_contour** function. For example, 50% of the data are enclosed within the 0.5 contour, 90% of the data are enclosed within the 0.9 contour, 99% of the data are within the 0.99 contour, etc. 

Comparing the quantile contours with the regression prediction intervals we see that 90-99% of the data points are within the 95% prediction intervals of the regression. The quantile contours are a big help for interpreting how well the regression fits the vast majority of the data.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, linspace
from delta_method import delta_method, kde_contour, quantile_contour

# -----
# ----- Read MOCHA seawater data from the csv available in this github repo ----
# -----
df = pd.read_csv("mocha_data.csv")
x=df['omega_ara']
y=df['pH']

# -----
# ----- Use delta-method for nonlinear regression ----
# -----
import scipy.optimize as opt
from delta_method import delta_method
# function for the nonlinear regression
def f(x, b0, b1, b2, b3):
     return b0 + b1*x + b2*x**2 + b3*x**3
# initial values for coefs
p_init = np.array([np.nanmean(y), 0, 0, 0])
# calc popt, pcov
popt, pcov = opt.curve_fit(f, x, y, p0=p_init, bounds=(-np.inf,np.inf))
# x_new and alpha
x_new = linspace(np.nanmin(x), np.nanmax(x), 100)
alpha=0.05
# run delta_method
d = delta_method(pcov,popt,x_new,f,x,y,alpha)
# Format regression equation
terms = []
for i, c in enumerate(popt[0:]):
    if i==0:
        terms.append(f"{c:.3f}")
    else:
        terms.append(f"{c:.3e}x^{i}")
equation = " + ".join(terms)
# eqn and stats for plot text box
textstr = (
    f"Equation: y = {equation}\n"
    f"R² = {d['rsquared']:.3f}, RMSE = {d['rmse']:.3f}, p={d['pvalue']:.2e}, N={len(x)}"
)

# -----
# ----- Use kde_contour and quantile_contour, and overlay with the regression results ----
# -----
fig, ax = plt.subplots(figsize=(10, 6))

# KDE values as filled contours in shades of blue with no lines
kde = kde_contour(x, y, ax=ax)

# Quantile contour lines in black with labels indicating the quantile enclosed within each line
quantiles = quantile_contour(x, y, ax=ax)

plt.plot(x_new, d['y_new'], color='red', label=f'Regression Best Fit')
plt.plot(x_new, d['lwr_pred'], '--', color='red', label=f'95% Prediction Interval')
plt.plot(x_new, d['upr_pred'], '--', color='red' )
plt.plot(x_new, d['lwr_conf'], ':', color='red', label=f'95% Confidence Interval')
plt.plot(x_new, d['upr_conf'], ':', color='red' )

plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

plt.legend(loc='lower right')
plt.title('Bivariate KDE and data quantiles of Ωara vs. pH, Apr-Sep, 0-200m depth, 0-40km from coast\n(e.g. 90% of the observations are enclosed within the 0.9 contour)')
plt.xlabel('x= Ωara', fontsize=12)
plt.ylabel('y= pH (total)', fontsize=12)
plt.xlim(np.nanmin(x), np.nanmax(x))
plt.ylim(np.nanmin(y), np.nanmax(y))
plt.grid(True)
plt.tight_layout()
plt.savefig("kdeplot_example.png", dpi=300)
plt.show()
```
<img width="3000" height="1800" alt="kdeplot_example" src="https://github.com/user-attachments/assets/5b1bd180-cd29-4d73-aab6-8f7110fc2e4c" />

### Example 3: Bivariate KDE and quantile contours

In this example we use a dataset from seaborn to demonstrate a bivariate Kernel Density Estimate (KDE) and quantile contours plot. A bivariate KDE plot visualizes the joint probability density function of two continuous variables. While a scatterplot shows the individual locations of data points, a bivariate KDE plot focuses on the density of these points, providing a continuous representation of the data's distribution rather than just discrete points. 

In this example, we demonstrate the following functions:

- **kde_contour** displays shades of blue indicating the bivariate KDE values
- **quantile_contour** displays black contour lines labeled with the data quantiles that are enclosed within (e.g. 90% of the data are encloed within the 0.9 contour line).
```
import seaborn as sns
import matplotlib.pyplot as plt
from delta_method import kde_contour, quantile_contour

plt.figure(figsize=(8, 6))

# load the iris data
iris = sns.load_dataset("iris")

# KDE contour plot as filled contours in shades of blue
kde_contour(
    x=iris['sepal_width'],
    y=iris['sepal_length'],
)

# scatter plot of sepal length vs width
plt.scatter(
    x=iris['sepal_width'],
    y=iris['sepal_length'],
    color='black',
    s=5,
    label='data'
)

# Quantile contour plot as black lines labeled with the data quantiles enclosed within each contour line
quantile_contour(
    x=iris['sepal_width'],
    y=iris['sepal_length'],
)

plt.legend(loc='upper left')
plt.title('Bivariate KDE and quantile contours of iris sepal length vs. width\n(e.g. 90% of the data are enclosed within the 0.9 contour line)')
plt.xlabel('sepal width')
plt.ylabel('sepal length')
plt.savefig("kdeplot_iris_example.png", dpi=300)
plt.show()
```
<img width="2400" height="1800" alt="kdeplot_iris_example" src="https://github.com/user-attachments/assets/4da8b553-2b32-428a-bab1-a76246755020" />

# Acknowledgement

The methods used in the delta_method and parametric_bootstrap functions and examples follow the methods described in an online lecture by Dr. Julien Chiquet (École Polytechnique/Université Paris-Saclay/AgroParisTech/INRAE) available at the following link:<br>
https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
