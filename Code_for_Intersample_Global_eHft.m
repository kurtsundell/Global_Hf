% Two dimensional Similarity %

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

% Percent contour 
conf = 95;

% Range of 2D data
xmin = 0;
xmax = 4000;
ymin = -45;
ymax = 20;

% kernel bandwidths
bandwidth_x = 40;
bandwidth_y = 1.5;

% how many pixels for the images, has to be in powers of 2, ne need to go over go over 2^12, results lookthe same
gridspc = 2^9;

% Read in data, format is name header and two columns of info, for our example we use age + Hf, but any 2D data will work
[numbers text, data] = xlsread(fullpathname);
numbers = num2cell(numbers);

% Fileter out any data that are not pairs of numbers
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
cmap = [1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.0625	0.125	0.1875	0.25	0.3125	0.375	0.4375	0.5	0.5625	0.625	0.6875	0.75	0.8125	0.875	0.9375	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0.9375	0.875	0.8125	0.75	0.6875	0.625	0.5625	0.5
1	0	0	0	0	0	0	0	0.0625	0.125	0.1875	0.25	0.3125	0.375	0.4375	0.5	0.5625	0.625	0.6875	0.75	0.8125	0.875	0.9375	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0.9375	0.875	0.8125	0.75	0.6875	0.625	0.5625	0.5	0.4375	0.375	0.3125	0.25	0.1875	0.125	0.0625	0	0	0	0	0	0	0	0	0
1	0.625	0.6875	0.75	0.8125	0.875	0.9375	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0.9375	0.875	0.8125	0.75	0.6875	0.625	0.5625	0.5	0.4375	0.375	0.3125	0.25	0.1875	0.125	0.0625	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0]';

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

S = sum(Similarity_Maps,3);

% Ask user if they want to plot the Similarity sum 
while(1)
	choice = menu('Plot the sum of the comparisons?', 'Yes','No');
	if choice==1 
		figure
		surf(X1,Y1,S);
		colormap(cmap)
		shading interp
		view(2)
		title('Similarity Sum')
		axis([xmin xmax ymin ymax])
	end
	break
end

%{
% plot supercontinent timing, need reference for this
figure
hold on
rectangle('Position',[200,-60,100,80],'FaceColor','y','EdgeColor','w') % Pangea
rectangle('Position',[480,-60,190,80],'FaceColor','y','EdgeColor','w') % Gondwana
rectangle('Position',[900,-60,400,80],'FaceColor','y','EdgeColor','w') % Rodinia
rectangle('Position',[1450,-60,525,80],'FaceColor','y','EdgeColor','w') % Columbia/Nuna
rectangle('Position',[2400,-60,300,80],'FaceColor','y','EdgeColor','w') % Kenorland?
axis([xmin xmax ymin ymax])
%}



%{
% Binning routine follows
bins = 100;

edges = 0:(xmax-xmin)/bins:xmax;

AFRd = discretize(AFR(:,1),edges);
AUSd = discretize(AUS(:,1),edges);
EURd = discretize(EUR(:,1),edges);
NAMd = discretize(NAM(:,1),edges);
SAMd = discretize(SAM(:,1),edges);

for i = 1:bins
	AFRh(i,1) = sum(AFRd==i);
	AUSh(i,1) = sum(AUSd==i);
	EURh(i,1) = sum(EURd==i);
	NAMh(i,1) = sum(NAMd==i);
	SAMh(i,1) = sum(SAMd==i);
end


Hall = AFRh + EURh + NAMh + SAMh + AUSh;

H1 = AFRh./Hall;
H2 = AFRh./Hall + EURh./Hall;
H3 = AFRh./Hall + EURh./Hall + NAMh./Hall;
H4 = AFRh./Hall + EURh./Hall + NAMh./Hall + SAMh./Hall;

Xh = 0:(xmax-xmin)/(bins-1):xmax;

while(1)
	choice = menu('Plot data binned by sample?', 'Yes','No');
	if choice==1 
		figure
		hold on
		patch([Xh fliplr(Xh)], [zeros(1,bins) fliplr(H2')], [69/255 35/255 125/255])
		patch([Xh fliplr(Xh)], [H1' fliplr(H2')], [74/255 113/255 209/255])
		patch([Xh fliplr(Xh)], [H2' fliplr(H3')], [83/255 176/255 179/255])
		patch([Xh fliplr(Xh)], [H3' fliplr(H4')], [146/255 198/255 106/255])
		patch([Xh fliplr(Xh)], [ones(1,bins) fliplr(H4')], [232/255 193/255 58/255])
		axis([xmin,xmax,0,1])
		set(gca, 'XDir','reverse')		
	end
	break
end

%}





% Weighting for bias test of Similarity sum








