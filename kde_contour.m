function contourHandle = kde_contour(x, y, varargin)
% Add a scaled bivariate KDE plot to a figure as contourf or contour plot in MATLAB
% Using MATLABS mvksdensity kernel smoothing function estimate for multivariate data
% Requires MATLAB Statsitics and Machine Learning Toolbox
% by Greg Pelletier

% % Example 1: filled contour plot of scaled bivariate KDE:
% load fisheriris;
% sepal_length = meas(:,1);
% sepal_width = meas(:,2);
% figure;
% kde_contour(sepal_width, sepal_length);
% xlabel('sepal width');
% ylabel('sepal length');
% title('Scaled KDE contours of iris sepal length vs. width');
% saveas(gcf, 'kde_contour_example1.png');

% % Example 2: unfilled contour plot of scaled bivariate KDE:
% load fisheriris;
% sepal_length = meas(:,1);
% sepal_width = meas(:,2);
% figure;
% kde_contour(sepal_width, sepal_length, 'fill', false, 'levels', [.05 .1, .25, .50, .75, .9, .95], 'clabel', true, 'cbar', false, 'color', 'black');
% xlabel('sepal width');
% ylabel('sepal length');
% title('Scaled KDE contours of iris sepal length vs. width');
% saveas(gcf, 'kde_contour_example2.png');

% Parse inputs
p = inputParser;
addParameter(p, 'threshold', 0.001);
addParameter(p, 'scale_kde', true);
addParameter(p, 'fill', true);
addParameter(p, 'color', []);
addParameter(p, 'cmap', 'turbo');
addParameter(p, 'cbar', true);
addParameter(p, 'cbar_fontsize', 10);
addParameter(p, 'cbar_fmt', '%.2f');
addParameter(p, 'grid_size', 200);
addParameter(p, 'levels', []);
addParameter(p, 'num_levels', []);
addParameter(p, 'linewidths', 1);
addParameter(p, 'linestyles', '-');
addParameter(p, 'clabel', false);
addParameter(p, 'clabel_fontsize', 8);
addParameter(p, 'clabel_fmt', '%.2f');
parse(p, varargin{:});
params = p.Results;

% Remove NaNs
x = x(:);
y = y(:);
mask = ~isnan(x) & ~isnan(y);
x = x(mask);
y = y(mask);

if isempty(x) || isempty(y)
    error('Input arrays must contain at least one non-NaN value.');
end

% Determine number of levels
if isempty(params.levels)
    if isempty(params.num_levels)
        if params.threshold < 0.05
            params.num_levels = 21;
        elseif params.threshold < 0.1
            params.num_levels = 20;
        else
            params.num_levels = 19;
        end
    elseif params.num_levels <= 1 || params.num_levels > 256
        params.num_levels = 20;
    end
end

% Create meshgrid
x_min = min(x); x_max = max(x);
y_min = min(y); y_max = max(y);
[xi, yi] = meshgrid(linspace(x_min, x_max, params.grid_size), ...
                    linspace(y_min, y_max, params.grid_size));
grid_coords = [xi(:)'; yi(:)'];

% KDE estimation using mvksdensity
z = mvksdensity([x y], [xi(:) yi(:)], 'Bandwidth', [], 'Kernel', 'normal');
z = reshape(z, size(xi));

% Scale KDE
if params.scale_kde
    z = (z - min(z(:))) / (max(z(:)) - min(z(:)));
end

% Apply threshold
z_max = max(z(:));
z(z < params.threshold * z_max) = NaN;

% Define levels
if isempty(params.levels)
    if params.scale_kde
        levels = linspace(params.threshold, 1.0, params.num_levels);
    else
        levels = linspace(params.threshold * z_max, z_max, params.num_levels);
    end
else
    levels = params.levels;
end

% Plot
hold on;
if params.fill
    contourHandle = contourf(xi, yi, z, levels, 'LineStyle', 'none');
else

    [contourHandle, h] = contour(xi, yi, z, levels, ...
        'LineWidth', params.linewidths, 'LineStyle', params.linestyles);
    if params.clabel
        clabel(contourHandle, h, 'FontSize', params.clabel_fontsize, ...
            'FontWeight', 'normal', 'LabelSpacing', 300);
    end
	if ~isempty(params.color)
		h.LineColor = params.color;
	end
	
end

% Colormap and colorbar
if isempty(params.color)
    colormap(params.cmap);
end
if params.cbar && isempty(params.color)
    cb = colorbar;

    if params.scale_kde
		cb.Label.String = "Scaled KDE (0–1)";
	else
		cb.Label.String = "KDE";
	end

    cb.Label.FontSize = params.cbar_fontsize;
    cb.TickLabelInterpreter = 'none';
end

end