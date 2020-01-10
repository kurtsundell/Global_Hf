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


while(1)
	choice = menu('Plot the sum of the comparisons?', 'Yes','No');
	if choice==1 
		figure
		hold on
		rectangle('Position',[200,-60,100,80],'FaceColor','y','EdgeColor','w') % Pangea
		rectangle('Position',[480,-60,190,80],'FaceColor','y','EdgeColor','w') % Gondwana
		rectangle('Position',[900,-60,400,80],'FaceColor','y','EdgeColor','w') % Rodinia
		rectangle('Position',[1450,-60,525,80],'FaceColor','y','EdgeColor','w') % Columbia/Nuna
		rectangle('Position',[2400,-60,300,80],'FaceColor','y','EdgeColor','w') % Kenorland?
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


%{

for i = 1:N
	density1_scaled(:,:,i) = density1(:,:,1).*disc(:,i)';
end




% Ask user if they want to plot all input sample bivariate KDEs?
while(1)
	choice = menu('Plot all samples SCALED?','Yes','No');
	if choice==1 
		for i = 1:N
			figure
			surf(X1,Y1,density1_scaled(:,:,i));
			colormap(cmap)
			shading interp
			view(2)
			title(Name(i,1))
			axis([xmin xmax ymin ymax])
		end
	end
	break
end

% Calculate intersample 2D Similarity SCALED
%for i = 1:(N*N-N)/2 % number of comparisons
count = 1;
for j = 1:N
	for k = 1:N
		if j > k
			for m = 1:size(density1_scaled,1)
				for n = 1:size(density1_scaled,2)
					Similarity_Maps_Scaled(m,n,count) = sqrt(density1_scaled(m,n,j).*density1_scaled(m,n,k)); % Similarity map
				end
			end
			%name_comp(count,1) = strcat(Name(j,1), {' vs '}, Name(k,1));
			count = count + 1;
		end
	end
end

% Ask user if they want to plot all of the Similarity comparison maps
while(1)
	choice = menu('Plot all comparisons SCALED?', 'Yes','No');
	if choice==1 
		for i = 1:(N*N-N)/2 % number of comparisons
			figure
			surf(X1,Y1,Similarity_Maps_Scaled(:,:,i));
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
	S_scaled(i,1) = sum(Similarity_Maps_Scaled(:,:,i),'all');
end

S_Map_scaled = sum(Similarity_Maps_Scaled,3);

% Ask user if they want to plot the Similarity sum 
while(1)
	choice = menu('Plot the sum of the comparisons SCALED?', 'Yes','No');
	if choice==1 
		figure
		surf(X1,Y1,S_Map_scaled);
		colormap(cmap)
		shading interp
		view(2)
		title('Similarity Sum')
		axis([xmin xmax ymin ymax])
	end
	break
end

















%}













