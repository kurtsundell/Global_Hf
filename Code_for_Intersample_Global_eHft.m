% Supporting code for Sundell and Macdonald (submitted) Geology for 2D KDE
% similarity comparison of global age-Hf data, or any other 2D data 

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
xmin = 0;
xmax = 4400;
ymin = -41;
ymax = 20;

% kernel bandwidths
bandwidth_x = 40;
bandwidth_y = 1.5;

% how many pixels for the images, has to be in powers of 2, ne need to go over go over 2^12, results lookthe same
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
MIN_XY=[xmin,ymin];
MAX_XY=[xmax,ymax];

% make bivariate kdes for samples, save as 3D matrix block
for k = 1:N
	data2 = data1(:,k*2-1:k*2);
	data2 = data2(any(data2 ~= 0,2),:);
	[bandwidth1,density1(:,:,k),X1,Y1] = kde2d_set_kernel(data2, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y);
	density1(:,:,k) = density1(:,:,k)./sum(sum(density1(:,:,k)));
	clear data2
end

% jet colormap that clips 0 values
cmap =[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000;1,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]';

% Ask user if they want to plot all input sample bivariate KDEs?
while(1)
	choice = menu('Plot all samples?','Yes','No');
	if choice==1 
		for i = 1:N
			figure
			surf(X1,Y1,density1(:,:,i));
			colormap(cmap)
			shading interp
			view(2)
			title(Name(i,1))
			axis([xmin xmax ymin ymax])
		end
	end
	break
end

%{
% Ask user if they want to plot all input sample bivariate KDE contours?
while(1)
	choice = menu('Plot and save sample contours?','Yes','No');
	if choice==1 
		for i = 1:N
			FIG = figure;
			max_density = max(max(density1(:,:,i)));
			max_density_conf = max_density - max_density*99*.01;
			contour3(X1,Y1,density1(:,:,i),[max_density_conf max_density_conf],'k', 'LineWidth', 4);
			view(2)
			grid on
			axis([xmin xmax ymin ymax])
			[file,path] = uiputfile('*.eps','Save file');
			print(FIG,'-depsc','-painters',[path file]);
			epsclean([path file]); 
		end
	end
	break
end
%}

% Calculate intersample 2D Similarity
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
			name_comp(count,1) = strcat(Name(j,1), {' vs '}, Name(k,1));
			count = count + 1;
		end
	end
end

% Ask user if they want to plot all of the Similarity comparison maps
while(1)
	choice = menu('Plot all comparisons?', 'Yes','No');
	if choice==1 
		for i = 1:(N*N-N)/2 % number of comparisons
			figure
			surf(X1,Y1,Similarity_Maps(:,:,i));
			colormap(cmap)
			shading interp
			view(2)
			title(name_comp(i,1))
			axis([xmin xmax ymin ymax])
		end
	end
	break
end

for i = 1:N
	S(i,1) = sum(Similarity_Maps(:,:,i),'all');
end

S_Map = sum(Similarity_Maps,3);

% Ask user if they want to plot the Similarity sum 
while(1)
	choice = menu('Plot the sum of the comparisons?', 'Yes','No');
	if choice==1 
		figure
		surf(X1,Y1,S_Map);
		colormap(cmap)
		shading interp
		view(2)
		title('Similarity Sum')
		xlabel('Age (Ma)')
		ylabel('Similarity Sum')
		axis([xmin xmax ymin ymax])
	end
	break
end

S_time = sum(S_Map,1);

for i = 1:((N*N)-N)/2
	S_time_ind(i,:) = sum(Similarity_Maps(:,:,i),1);
end

while(1)
	choice = menu('Plot the sum of the comparisons and compared to supercontinent tenures?', 'Yes','No');
	if choice==1 
		S_time_x = xmin:(xmax-xmin)/(length(density1(:,:,1))-1):xmax;
		S_time_ind_sum = sum(S_time_ind);
		color = jet(((N*N)-N)/2);
		
		figure 
		hold on
		for i = 1:((N*N)-N)/2
			plot(S_time_x,S_time_ind(i,:),'Color',color(i,:),'LineWidth',4)
		end
		legend(name_comp)
		xlim([xmin xmax])
		xlabel('Age (Ma)')
		ylabel('Similarity')

		color = jet(((N*N)-N)/2);
		
		figure 
		hold on
		for i = 1:((N*N)-N)/2
			plot(S_time_x,S_time_ind(i,:),'Color',color(i,:),'LineWidth',4)
		end
		plot(S_time_x,S_time_ind_sum,'Color','k','LineWidth',5)
		legend([name_comp;{'Sum of all comparisons'}])
		axis([xmin xmax 0 max(S_time)])
		xlabel('Age (Ma)')
		ylabel('Similarity')
		
		figure
		hold on
		rectangle('Position',[200,-60,100,80],'FaceColor','y','EdgeColor','w') % 300 to 200 Ma Pangea (Evans et al., 2016)
		rectangle('Position',[300,-60,280,80],'FaceColor','y','EdgeColor','w') % 580 to 300 Ma Pannotia/Gondwana (Evans et al., 2016)
		rectangle('Position',[750,-60,150,80],'FaceColor','y','EdgeColor','w') % 900 to 750 Ma Rodinia (Evans et al., 2016)
		rectangle('Position',[1350,-60,200,80],'FaceColor','y','EdgeColor','w') % Nuna 1550 to 1350 Ma (Evans et al., 2016)
		rectangle('Position',[2100,-60,400,80],'FaceColor','y','EdgeColor','w') % Kenorland 2500 to 2100 Ma (Evans et al., 2016)
		plot(S_time_x, S_time, 'LineWidth', 4, 'Color', 'k')
		xlabel('Age Ma')
		ylabel('Sum of all comparisons')
		axis([xmin xmax 0 max(S_time)])
	end
	break
end

% Binning routine follows
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

while(1)
	choice = menu('Plot data binned by sample?', 'Yes','No');
	if choice==1 
		figure 
		hold on 
		c = colormap(jet(N));
		for i = N:-1:1
			patch([Xh fliplr(Xh)], [zeros(1,bins) fliplr(Disc_Bins_Cumu(:,i)')], c(i,:))
		end
		axis([xmin xmax 0 100])
		%set(gca, 'XDir','reverse')
		legend(flipud(Name))
	end
	break
end

%2D KDE algorithm sourced from
%https://github.com/rctorres/Matlab/blob/master/kde2d.m or 
%https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/17204/versions/5/previews/kde2d.m/index.html
%Modified by Sundell -- added bandwidth_x and bandwidth_y to the original code for set bandwidths (i.e., no Botev et al. (2010) algorithm)
function [bandwidth,density,X,Y]=kde2d_set_kernel(data,n,MIN_XY,MAX_XY,bandwidth_x,bandwidth_y) 
% bivariate kernel density estimator
% with diagonal bandwidth matrix.
% The kernel is assumed to be Gaussian.
% The two bandwidth parameters are
% chosen optimally without ever
% using/assuming a parametric model for the data or any "rules of thumb".
% Unlike many other procedures, this one
% is immune to accuracy failures in the estimation of
% multimodal densities with widely separated modes (see examples).
% INPUTS: data - an N by 2 array with continuous data
%            n - size of the n by n grid over which the density is computed
%                n has to be a power of 2, otherwise n=2^ceil(log2(n));
%                the default value is 2^8;
% MIN_XY,MAX_XY- limits of the bounding box over which the density is computed;
%                the format is:
%                MIN_XY=[lower_Xlim,lower_Ylim]
%                MAX_XY=[upper_Xlim,upper_Ylim].
%                The dafault limits are computed as:
%                MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
%                MAX_XY=MAX+Range/4; MIN_XY=MIN-Range/4;
% OUTPUT: bandwidth - a row vector with the two optimal
%                     bandwidths for a bivaroate Gaussian kernel;
%                     the format is:
%                     bandwidth=[bandwidth_X, bandwidth_Y];
%          density  - an n by n matrix containing the density values over the n by n grid;
%                     density is not computed unless the function is asked for such an output;
%              X,Y  - the meshgrid over which the variable "density" has been computed;
%                     the intended usage is as follows:
%                     surf(X,Y,density)
% Example (simple Gaussian mixture)
% clear all
%   % generate a Gaussian mixture with distant modes
%   data=[randn(500,2);
%       randn(500,1)+3.5, randn(500,1);];
%   % call the routine
%     [bandwidth,density,X,Y]=kde2d(data);
%   % plot the data and the density estimate
%     contour3(X,Y,density,50), hold on
%     plot(data(:,1),data(:,2),'r.','MarkerSize',5)
%
% Example (Gaussian mixture with distant modes):
%
% clear all
%  % generate a Gaussian mixture with distant modes
%  data=[randn(100,1), randn(100,1)/4;
%      randn(100,1)+18, randn(100,1);
%      randn(100,1)+15, randn(100,1)/2-18;];
%  % call the routine
%    [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,60])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
% Example (Sinusoidal density):
%
% clear all
%   X=rand(1000,1); Y=sin(X*10*pi)+randn(size(X))/3; data=[X,Y];
%  % apply routine
%  [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,70])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
%  Reference:
% Kernel density estimation via diffusion
% Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
% Annals of Statistics, Volume 38, Number 5, pages 2916-2957.

global N A2 I
if nargin<2
    n=2^8;
end
n=2^ceil(log2(n)); % round up n to the next power of 2;
N=size(data,1);
if nargin<3
    MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
    MAX_XY=MAX+Range/2; MIN_XY=MIN-Range/2;
end
scaling=MAX_XY-MIN_XY;
if N<=size(data,2)
    error('data has to be an N by 2 array where each row represents a two dimensional observation')
end
transformed_data=(data-repmat(MIN_XY,N,1))./repmat(scaling,N,1);
%bin the data uniformly using regular grid;
initial_data=ndhist(transformed_data,n);

% discrete cosine transform of initial data
a= dct2d(initial_data);

% now compute the optimal bandwidth^2
  I=(0:n-1).^2; A2=a.^2;
 t_star=root(@(t)(t-evolve(t)),N);
p_02=func([0,2],t_star);p_20=func([2,0],t_star); p_11=func([1,1],t_star);

t_x=(p_20^(3/4)/(4*pi*N*p_02^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);
t_y=(p_02^(3/4)/(4*pi*N*p_20^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);

%bandwidth_opt = sqrt([t_x,t_y]).*scaling;

% Sundell modified this bit for set kernels
bandwidth = [bandwidth_x, bandwidth_y];
t_x = (bandwidth_x(1,1)/scaling(1,1))^2;
t_y = (bandwidth_y(1,1)/scaling(1,2))^2;

% smooth the discrete cosine transform of initial data using t_star
a_t=exp(-(0:n-1)'.^2*pi^2*t_x/2)*exp(-(0:n-1).^2*pi^2*t_y/2).*a; 

% now apply the inverse discrete cosine transform
if nargout>1
    density=idct2d(a_t)*(numel(a_t)/prod(scaling));
	density(density<0)=eps; % remove any negative density values
    [X,Y]=meshgrid(MIN_XY(1):scaling(1)/(n-1):MAX_XY(1),MIN_XY(2):scaling(2)/(n-1):MAX_XY(2));
end

end
%#######################################
function  [out,time]=evolve(t)
global N
Sum_func = func([0,2],t) + func([2,0],t) + 2*func([1,1],t);
time=(2*pi*N*Sum_func)^(-1/3);
out=(t-time)/time;
end
%#######################################
function out=func(s,t)
global N
if sum(s)<=4
    Sum_func=func([s(1)+1,s(2)],t)+func([s(1),s(2)+1],t); const=(1+1/2^(sum(s)+1))/3;
    time=(-2*const*K(s(1))*K(s(2))/N/Sum_func)^(1/(2+sum(s)));
    out=psi(s,time);
else
    out=psi(s,t);
end

end
%#######################################
function out=psi(s,Time)
global I A2
% s is a vector
w=exp(-I*pi^2*Time).*[1,.5*ones(1,length(I)-1)];
wx=w.*(I.^s(1));
wy=w.*(I.^s(2));
out=(-1)^sum(s)*(wy*A2*wx')*pi^(2*sum(s));
end
%#######################################
function out=K(s)
out=(-1)^s*prod((1:2:2*s-1))/sqrt(2*pi);
end
%#######################################
function data=dct2d(data)
% computes the 2 dimensional discrete cosine transform of data
% data is an nd cube
[nrows,ncols]= size(data);
if nrows~=ncols
    error('data is not a square array!')
end
% Compute weights to multiply DFT coefficients
w = [1;2*(exp(-i*(1:nrows-1)*pi/(2*nrows))).'];
weight=w(:,ones(1,ncols));
data=dct1d(dct1d(data)')';
    function transform1d=dct1d(x)

        % Re-order the elements of the columns of x
        x = [ x(1:2:end,:); x(end:-2:2,:) ];

        % Multiply FFT by weights:
        transform1d = real(weight.* fft(x));
    end
end
%#######################################
function data = idct2d(data)
% computes the 2 dimensional inverse discrete cosine transform
[nrows,ncols]=size(data);
% Compute wieghts
w = exp(i*(0:nrows-1)*pi/(2*nrows)).';
weights=w(:,ones(1,ncols));
data=idct1d(idct1d(data)');
    function out=idct1d(x)
        y = real(ifft(weights.*x));
        out = zeros(nrows,ncols);
        out(1:2:nrows,:) = y(1:nrows/2,:);
        out(2:2:nrows,:) = y(nrows:-1:nrows/2+1,:);
    end
end
%#######################################
function binned_data=ndhist(data,M)
% this function computes the histogram
% of an n-dimensional data set;
% 'data' is nrows by n columns
% M is the number of bins used in each dimension
% so that 'binned_data' is a hypercube with
% size length equal to M;
[nrows,ncols]=size(data);
bins=zeros(nrows,ncols);
for i=1:ncols
    [dum,bins(:,i)] = histc(data(:,i),[0:1/M:1],1);
    bins(:,i) = min(bins(:,i),M);
end
% Combine the  vectors of 1D bin counts into a grid of nD bin
% counts.
binned_data = accumarray(bins(all(bins>0,2),:),1/nrows,M(ones(1,ncols)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function t=root(f,N)
% try to find smallest root whenever there is more than one
N=50*(N<=50)+1050*(N>=1050)+N*((N<1050)&(N>50));
tol=10^-12+0.01*(N-50)/1000;
flag=0;
while flag==0
    try
        t=fzero(f,[0,tol]);
        flag=1;
    catch
        tol=min(tol*2,.1); % double search interval
    end
    if tol==.1 % if all else fails
        t=fminbnd(@(x)abs(f(x)),0,.1); flag=1;
    end
end
end

%cell2num sourced from https://www.mathworks.com/matlabcentral/fileexchange/15306-cell2num
function [outputmat]=cell2num(inputcell)
% Function to convert an all numeric cell array to a double precision array
% ********************************************
% Usage: outputmatrix=cell2num(inputcellarray)
% ********************************************
% Output matrix will have the same dimensions as the input cell array
% Non-numeric cell contest will become NaN outputs in outputmat
% This function only works for 1-2 dimensional cell arrays

if ~iscell(inputcell), error('Input cell array is not.'); end

outputmat=zeros(size(inputcell));

for c=1:size(inputcell,2)
  for r=1:size(inputcell,1)
    if isnumeric(inputcell{r,c})
      outputmat(r,c)=inputcell{r,c};
    else
      outputmat(r,c)=NaN;
    end
  end  
end

end




