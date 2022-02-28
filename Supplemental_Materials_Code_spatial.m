% Supporting code for Sundell and Macdonald (submitted, EPSL) for bivariate KDE of global eHfT data from Puetz et al. (2021)

clear all 
close all
clc

[filename pathname] = uigetfile({'*'},'File Selector'); %load the supplemental file with zircon age eHfT data

if ispc == 1
	fullpathname = char(strcat(pathname, '\', filename));
end
if ismac == 1
	fullpathname = char(strcat(pathname, '/', filename));
end

% Read in data, format is name header and two columns of info, for our example we use age + Hf, but any 2D data will work
[lat_lon_All text, data] = xlsread(fullpathname);

% Range of 2D data
xmin = -180;
xmax = 180;
ymin = -90;
ymax = 90;

% kernel bandwidths
bandwidth_x = 10;
bandwidth_y = 5;

% how many pixels for the images, has to be in powers of 2, ne need to go over go over 2^12, results lookthe same
gridspc = 2^9;

% set min/max ranges for kde2d function
MIN_XY=[xmin,ymin];
MAX_XY=[xmax,ymax];

cmap = cmocean('balance',100);
%cmap(1:5,:) = 1; %clip zero vals and set as white space

[bandwidth1,density1,X1,Y1] = kde2d_set_kernel([lat_lon_All(:,2),lat_lon_All(:,1)], gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y);
density1 = density1./sum(sum(density1));

figure
hold on
surf(X1,Y1,density1);
colormap(cmap)
shading interp
view(2)
xlabel('Longitude','FontSize',20)
ylabel('Latitude','FontSize',20)
axis([xmin xmax ymin ymax])
set(gca,'FontSize',20)
title('Spatial eHfT Data Density','FontSize',20)
borders('countries','k','linewidth',3)

figure
hold on
borders('countries','k','linewidth',3)
scatter(lat_lon_All(:,2),lat_lon_All(:,1),'filled')

F = figure;
hold on
max_density = max(max(density1));
max_density_conf = max_density - max_density*.95;
contour3(X1,Y1,density1,[max_density_conf max_density_conf],'k', 'LineWidth', 4);
%max_density = max(max(density1));
%max_density_conf = max_density - max_density*.68;
%contour3(X1,Y1,density1,[max_density_conf max_density_conf],'k', 'LineWidth', 4);
view(2)
grid off
axis([xmin xmax ymin ymax])
%[file,path] = uiputfile('*.eps','Save file'); print(F,'-depsc','-painters',[path file]); epsclean([path file]); % save simplified contours

figure
hold on
surf(X1,Y1,density1);
colormap(cmap)
shading interp
view(2)
xlabel('Longitude','FontSize',20)
ylabel('Latitude','FontSize',20)
axis([xmin xmax ymin ymax])
set(gca,'FontSize',20)
title('Spatial eHfT Data Density','FontSize',20)

figure
hold on
surf(X1,Y1,density1);
colormap(cmap)
shading interp
max_density = max(max(density1));
max_density_conf = max_density - max_density*.95;
contour3(X1,Y1,density1,[max_density_conf max_density_conf],'k', 'LineWidth', 4);
max_density = max(max(density1));
max_density_conf = max_density - max_density*.68;
contour3(X1,Y1,density1,[max_density_conf max_density_conf],'k', 'LineWidth', 4);
view(2)
xlabel('Longitude','FontSize',20)
ylabel('Latitude','FontSize',20)
axis([xmin xmax ymin ymax])
set(gca,'FontSize',20)
title('Spatial eHfT Data Density','FontSize',20)
