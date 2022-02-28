% Supporting code for Sundell and Macdonald (submitted, EPSL) for bivariate KDE and two-dimensional
% similarity comparisons of global eHfT data from Puetz et al. (2021)

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

% Range of 2D data
xmin = 485;
xmax = 635;
ymin = -41;
ymax = 20;
extra = 45;

% kernel bandwidths
bandwidth_x = 15;
bandwidth_y = 1;

% how many pixels for the images, has to be in powers of 2, no need to go over go over 2^12, results look the same
gridspc = 2^9;

% Read in data, format is name header and two columns of info, for our example we use age + Hf, but any 2D data will work
[numbers text, data] = xlsread(fullpathname);
numbers = num2cell(numbers);

% Filter out any data that are not pairs of numbers
for i = 1:size(numbers,1)
	for j = 1:size(numbers,2)
		if cellfun('isempty', numbers(i,j)) == 0
			if cellfun(@isnan, numbers(i,j)) == 1
				numbers(i,j) = {[]};
			end
		end
	end
end

% pull the names from the headers
for i = 1:(size(data,2)+1)/2
	Name(i,1) = data(1,i*2-1);
end

data_tmp = numbers(1:end,:); %use temporary variable
N = size(data_tmp,2)/2; % figure out how many samples

% Filter out any data not in the range set above
for k = 1:N
	for i = 1:length(data_tmp(:,1))
		if cellfun('isempty', data_tmp(i,k*2-1)) == 0 && cellfun('isempty', data_tmp(i,k*2)) == 0
			if cell2num(data_tmp(i,k*2-1)) >= xmin && cell2num(data_tmp(i,k*2-1)) <= xmax && ...
					cell2num(data_tmp(i,k*2)) >= ymin && cell2num(data_tmp(i,k*2)) <= ymax
				data1(i,k*2-1:k*2) = cell2num(data_tmp(i,k*2-1:k*2));
			end
		end
	end
end

% set min/max ranges for kde2d function
MIN_XY=[xmin-extra,ymin];
MAX_XY=[xmax+extra,ymax];

%colormaps
% cmap = cmocean('ColormapName') returns a 256x3 colormap. ColormapName can be any of
% of the following:
%
%          SEQUENTIAL:                DIVERGING:
%          'thermal'                  'balance'
%          'haline'                   'delta'
%          'solar'                    'curl'
%          'ice'                      'diff'
%          'gray'                     'tarn'
%          'oxy'
%          'deep'                     CONSTANT LIGHTNESS:
%          'dense'                    'phase'
%          'algae'
%          'matter'                   OTHER:
%          'turbid'                   'topo'
%          'speed'
%          'amp'
%          'tempo'
%          'rain'

cmap = cmocean('balance',300);
cmap(1:3,:) = 1; %clip zero vals at 95% and set as white space

% Make and plot bivariate kdes for samples, save as 3D matrix block
for k = 1:N
	data2 = data1(:,k*2-1:k*2);
	data2 = data2(any(data2 ~= 0,2),:);
	[bandwidth1,density1(:,:,k),X1,Y1] = kde2d_set_kernel(data2, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y);
	density1(:,:,k) = density1(:,:,k)./sum(sum(density1(:,:,k)));
	
	figure
	hold on
	
	%plot density
	surf(X1,Y1,density1(:,:,k));
	
	%make contours
	max_density1 = max(max(density1(:,:,k)));
	perc = 0.99; % percent of bivariate KDE from peak
	max_density_conf(k,1) = max_density1*(1-perc); % contour at 68% from peak density 
	
	%{
	%plot contours
	contour3(X1,Y1,density1(:,:,k),[max_density_conf(k,1) max_density_conf(k,1)],'k', 'LineWidth', 3);

	%plot running averages +/- 2stdev
	X2 = (data2(:,1));
	Y2 = (data2(:,2));
	[X_sorted, X_order] = sort(X2);
	Y_sorted = Y2(X_order,:);
	M = movmean(Y_sorted,500);
	Std = movstd(Y_sorted,500);
	plot3(X_sorted,M,ones(length(M),1), 'linewidth',3, 'color', 'g')
	plot3(X_sorted,M+2*Std,ones(length(M),1), 'linewidth',1, 'color', 'g')
	plot3(X_sorted,M-2*Std,ones(length(M),1), 'linewidth',1, 'color', 'g')
	clear data2 M Std
	%}
	
	% format plots
	colormap(cmap)
	shading interp
	view(2)
	title(Name(k,1),'FontSize',40)
	xlabel('Age (Ma)','FontSize',20)
	ylabel('eHfT','FontSize',20)
	axis([xmin-extra xmax+extra ymin ymax])
	set(gca,'FontSize',20)
end


%for saving simplified contour figures to individual files, each fig has separate browser prompt, good for illustrator
for i = 1:N
	F1 = figure;
	contour3(X1,Y1,density1(:,:,i),[max_density_conf(i,1) max_density_conf(i,1)],'k', 'LineWidth', 3);
	grid off
	view(2)
	[file,path] = uiputfile('*.eps','Save file'); print(F1,'-depsc','-painters',[path file]); epsclean([path file]); % save simplified contours
	axis([xmin-extra xmax+extra ymin ymax])
end

%{
% Calculate and plot pairwise 2D Similarity
%for i = 1:(N*N-N)/2 % number of comparisons
count = 1;
for j = 1:N
	for k = 1:N
		if j > k
			for m = 1:size(density1,1)
				for n = 1:size(density1,2)
					Similarity_Maps(m,n,count) = sqrt(density1(m,n,j).*density1(m,n,k)); % Similarity map
				end
			end
			Similarity_Maps_S(j,k) = sum(sum(Similarity_Maps(:,:,count)));
			Similarity_Maps_peak(j,k) = max(max(Similarity_Maps(:,:,count)));
			name_comp(count,1) = strcat(Name(j,1), {' vs '}, Name(k,1));
			count = count + 1;
		end
	end
end
%}

%{
%find peak similarity for all comparisons
Similarity_Maps_peak_max = max(max(Similarity_Maps_peak));

%normalize to peak similarity
count = 1;
for j = 1:N
	for k = 1:N
		if j > k
			cmap_clip(count,1) = round(Similarity_Maps_peak(j,k)/Similarity_Maps_peak_max*length(cmap(:,1)));
			count = count + 1;
		end
	end
end


%plot all comparisons with normalized colormaps
for i = 1:(N*N-N)/2 % number of comparisons
	figure
	surf(X1,Y1,Similarity_Maps(:,:,i));
	colormap(cmap(1:cmap_clip(i,1),:))
	shading interp
	view(2)
	title(name_comp(i,1),'FontSize',40)
	xlabel('Age (Ma)','FontSize',20)
	ylabel('eHfT','FontSize',20)
	axis([xmin xmax ymin ymax])
	set(gca,'FontSize',20)
end
%}

%{
%Similarity calculation (number btw 0 and 1 where 1 is exact match)
for i = 1:N
	S(i,1) = sum(Similarity_Maps(:,:,i),'all');
end

%Sum all similarity maps into single map, plot similarity sum
S_Map = sum(Similarity_Maps,3);
figure
hold on
surf(X1,Y1,S_Map);
colormap(cmap)
shading interp
view(2)
title('Similarity Sum','FontSize',40)
xlabel('Age (Ma)','FontSize',20)
ylabel('eHfT','FontSize',20)
axis([xmin xmax ymin ymax])
set(gca,'FontSize',20)

%make contours
max_density_Smap = max(max(S_Map));
perc1 = 0.68; % percent of bivariate KDE from peak
perc2 = 0.95; % percent of bivariate KDE from peak
max_density_conf1(k,1) = max_density_Smap*(1-perc1); % contour at 68% from peak density
max_density_conf2(k,1) = max_density_Smap*(1-perc2); % contour at 95% from peak density

%plot contours
contour3(X1,Y1,S_Map,[max_density_conf1(k,1) max_density_conf1(k,1)],'k', 'LineWidth', 3);
contour3(X1,Y1,S_Map,[max_density_conf2(k,1) max_density_conf2(k,1)],'k', 'LineWidth', 3);


%for saving simplified contour figures to individual files, each fig has separate browser prompt, good for illustrator
F1 = figure;
hold on
contour3(X1,Y1,S_Map,[max_density_conf1(k,1) max_density_conf1(k,1)],'k', 'LineWidth', 3);
contour3(X1,Y1,S_Map,[max_density_conf2(k,1) max_density_conf2(k,1)],'k', 'LineWidth', 3);
grid off
view(2)
axis([xmin xmax ymin ymax])
%[file,path] = uiputfile('*.eps','Save file'); print(F1,'-depsc','-painters',[path file]); epsclean([path file]); % save simplified contours
%}

%{
% Binning routine follows (Fig. 3A-B in manuscript)
bins = 45;

edges = 0:(xmax-xmin)/bins:xmax;

for i = 1:N
	tmp = discretize(nonzeros(data1(:,i*2-1)),edges);
	Disc(i,1) = {tmp};
	clear tmp
end

% add up how many data points at each bin
for j = 1:N
	tmp = Disc{j,1};
	for i = 1:bins
		Disc_Bins(i,j) = sum(tmp==i);
	end
	clear tmp
end

Disc_Bins_Sum = sum(Disc_Bins,2);

Disc_Bins_Perc = Disc_Bins./repmat(Disc_Bins_Sum,1,N).*100;

disc = Disc_Bins_Perc/100;

for k = 1:N
	data2 = data1(:,k*2-1:k*2);
	data2 = data2(any(data2 ~= 0,2),:);
	[bandwidth1,density1(:,:,k),X1,Y1] = kde2d_set_kernel(data2, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y);
	density1(:,:,k) = density1(:,:,k)./sum(sum(density1(:,:,k)));
	clear data2
end

for i = 1:N
	Disc_Bins_Cumu(:,i) = sum(Disc_Bins_Perc(:,1:i),2);
end

Xh = 0:(xmax-xmin)/(bins-1):xmax;

figure
hold on
c = colormap(jet(N));
for i = N:-1:1
	patch([Xh fliplr(Xh)], [zeros(1,bins) fliplr(Disc_Bins_Cumu(:,i)')], c(i,:))
end
axis([xmin xmax 0 100])
title(strcat('Data binned by sample'),'FontSize',40)
legend(flipud(Name))
%}