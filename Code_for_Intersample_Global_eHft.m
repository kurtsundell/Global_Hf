clear all
close all
clc

global botev
global bandwidth_x
global bandwidth_y
global bandwidth_opt

botev = 0;

conf = 95;

xmin = 0;
xmax = 4000;
ymin = -45;
ymax = 20;

bandwidth_x = 40;
bandwidth_y = 1.5;

gridspc = 2^9;

[filename pathname] = uigetfile({'*'},'File Selector');
[numbers text1, data] = xlsread([strcat(pathname, filename)]);
numbers = num2cell(numbers);

for i = 1:size(numbers,1)
	for j = 1:size(numbers,2)
		if cellfun('isempty', numbers(i,j)) == 0
			if cellfun(@isnan, numbers(i,j)) == 1
				numbers(i,j) = {[]};
			end	
		end
	end
end

for i = 1:(size(data,2)+1)/2
	Name(i,1) = data(1,i*2-1);
end


data_tmp = numbers(1:end,:);
%tmp = get(H.uitable8, 'data');
%Selected = tmp(:,1);
N = size(data_tmp,2)/2;


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

MIN_XY=[xmin,ymin];
MAX_XY=[xmax,ymax];

for k = 1:N
		data2 = data1(:,k*2-1:k*2);
		data2 = data2(any(data2 ~= 0,2),:);
		if k == 1
			AFR = data2;
		elseif k == 2
			AUS = data2;
		elseif k == 3
			EUR = data2;
		elseif k == 4
			NAM = data2;
		elseif k == 5
			SAM = data2;
		end
		[bandwidth1,density1(:,:,k),X1,Y1]=kde2d(data2, gridspc, MIN_XY, MAX_XY);
		density1(:,:,k) = density1(:,:,k)./sum(sum(density1(:,:,k)));
end

%bandwidth1 = bandwidth1(any(bandwidth1 ~= 0,2),:);

for i = 1:N
	maxdens(i,1) = max(max(density1(:,:,i)))
end

for i = 1:N
	addbase(i,1) = sum(maxdens(1:i-1,1)) + maxdens(1,1)
end

transpL = 1;


cmap = [1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.0625	0.125	0.1875	0.25	0.3125	0.375	0.4375	0.5	0.5625	0.625	0.6875	0.75	0.8125	0.875	0.9375	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0.9375	0.875	0.8125	0.75	0.6875	0.625	0.5625	0.5
1	0	0	0	0	0	0	0	0.0625	0.125	0.1875	0.25	0.3125	0.375	0.4375	0.5	0.5625	0.625	0.6875	0.75	0.8125	0.875	0.9375	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0.9375	0.875	0.8125	0.75	0.6875	0.625	0.5625	0.5	0.4375	0.375	0.3125	0.25	0.1875	0.125	0.0625	0	0	0	0	0	0	0	0	0
1	0.625	0.6875	0.75	0.8125	0.875	0.9375	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0.9375	0.875	0.8125	0.75	0.6875	0.625	0.5625	0.5	0.4375	0.375	0.3125	0.25	0.1875	0.125	0.0625	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0]';


%{
figure
hold on
for i = 1:N
	%if get(H.plot_heat,'Value')==1
		if i == 1
			s(1) = surf(X1,Y1,density1(:,:,i));
		end
		if i > 1
			s(i) = pcolor(X1,Y1,density1(:,:,i));
			set(s(i),'ZData', density1(:,:,i) + addbase(i,1))
		end
	%end
end

	colormap(cmap)
	shading interp
	view(3)
%}

%{
for i = 1:N
	figure
	surf(X1,Y1,density1(:,:,i));
	colormap(cmap)
	shading interp
	view(2)
	title(Name(i,1))
end
%}
	
	
	



for j = 1:N
	for k = 1:N
		for m = 1:size(density1,1)
			for n = 1:size(density1,2)
				Sall(m,n) = sqrt(density1(m,n,j).*density1(m,n,k)); % Similarity map	
			end
		end
		
		
		if j == 2 && k == 1
			S_OUT(:,:,1) = Sall;
			name_comp(1,1) = strcat(Name(2,1), {' vs '}, Name(1,1));
			
		elseif j == 3 && k == 1 
			S_OUT(:,:,2) = Sall;
			name_comp(2,1) = strcat(Name(3,1), {' vs '}, Name(1,1));
			
		elseif j == 4 && k == 1 
			S_OUT(:,:,3) = Sall;
			name_comp(3,1) = strcat(Name(4,1), {' vs '}, Name(1,1));
			
		elseif j == 5 && k == 1
			S_OUT(:,:,4) = Sall;
			name_comp(4,1) = strcat(Name(5,1), {' vs '}, Name(1,1));
		

		elseif j == 3 && k == 2 
			S_OUT(:,:,5) = Sall;
			name_comp(5,1) = strcat(Name(3,1), {' vs '}, Name(2,1));
			
		elseif j == 4 && k == 2 
			S_OUT(:,:,6) = Sall;
			name_comp(6,1) = strcat(Name(4,1), {' vs '}, Name(2,1));
			
		elseif j == 5 && k == 2 
			S_OUT(:,:,7) = Sall;
			name_comp(7,1) = strcat(Name(5,1), {' vs '}, Name(2,1));
		
			
		elseif j == 4 && k == 3 
			S_OUT(:,:,8) = Sall;			
			name_comp(8,1) = strcat(Name(4,1), {' vs '}, Name(3,1));
			
		elseif j == 5 && k == 3 
			S_OUT(:,:,9) = Sall;
			name_comp(9,1) = strcat(Name(5,1), {' vs '}, Name(3,1));
	
			
		elseif j == 5 && k == 4 
			S_OUT(:,:,10) = Sall;
			name_comp(10,1) = strcat(Name(5,1), {' vs '}, Name(4,1));
		
		end
		
		
		
		
		S2D_tmp = sum(Sall);
		S2D(j,k) = sum(S2D_tmp);
	end
end



%{
figure
hold on

evo = 'k';

DM_Slider = 4000;

density = density1;

Epsilon_plot = [16.5,14.6,0,15.6,0;15.0,13.0,500,14.0,0;13.4,11.5,1000,12.5,0;11.9,9.9,1500,10.9,0;10.3,8.3,2000,9.3,0; ...
	8.7,6.7,2500,7.7,0;5.4,3.4,3500,4.4,0;3.7,1.7,4000,2.7,0;2.0,0.0,4500,1.0,0];

Decay_const_176Lu = 0.01867; %176Lu decay constant (Scherer et al., 2001) 1.867*10^-11 (same as Soderland et al., 2004)
DM_176Hf_177Hf = 0.283225; %Vervoort and Blichert-Toft, 1999
DM_176Lu_177Hf = 0.0383; %Vervoort and Blichert-Toft, 1999
BSE_176Hf_177Hf = 0.282785; %Bouvier et al. 2008
BSE_176Lu_177Hf = 0.0336; %Bouvier et al. 2008

t_176Hf_177Hf = DM_176Hf_177Hf - (DM_176Lu_177Hf*(exp(Decay_const_176Lu*DM_Slider/1000)-1));

CHURt = BSE_176Hf_177Hf - (BSE_176Lu_177Hf*(exp(Decay_const_176Lu*DM_Slider/1000)-1));
 
DMpoint_Epsi_x = DM_Slider;
DMpoint_Epsi_y = 10000*((t_176Hf_177Hf/CHURt)-1);

Y0_Evol_DM_176Lu_177Hf = t_176Hf_177Hf + (0.0115*(exp(Decay_const_176Lu*DM_Slider/1000)-1));
Y0_u_Evol_DM_176Lu_177Hf = t_176Hf_177Hf + (0.0193*(exp(Decay_const_176Lu*DM_Slider/1000)-1));
Y0_l_Evol_DM_176Lu_177Hf = t_176Hf_177Hf + (0.0036*(exp(Decay_const_176Lu*DM_Slider/1000)-1));
Ys_Evol_DM_176Lu_177Hf = t_176Hf_177Hf;

Y0_Epsi_DM_176Lu_177Hf = 10000*((Y0_Evol_DM_176Lu_177Hf/BSE_176Hf_177Hf)-1);
Y0_u_Epsi_DM_176Lu_177Hf = 10000*((Y0_u_Evol_DM_176Lu_177Hf/BSE_176Hf_177Hf)-1);
Y0_l_Epsi_DM_176Lu_177Hf = 10000*((Y0_l_Evol_DM_176Lu_177Hf/BSE_176Hf_177Hf)-1);
Ys_Epsi_DM_176Lu_177Hf = DMpoint_Epsi_y;

plot(Epsilon_plot(:,3),Epsilon_plot(:,5),evo,'LineWidth',2)
plot(Epsilon_plot(:,3),Epsilon_plot(:,4),evo,'LineWidth',2)
plot(Epsilon_plot(:,3),Epsilon_plot(:,1),strcat('--',evo),'LineWidth',2)
plot(Epsilon_plot(:,3),Epsilon_plot(:,2),strcat('--',evo),'LineWidth',2)
plot([0 DM_Slider],[Y0_u_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf], 'Color', evo, 'LineWidth', 2)
plot([0 DM_Slider],[Y0_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf], 'Color', evo, 'LineWidth', 2)
plot([0 DM_Slider],[Y0_l_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf], 'Color', evo, 'LineWidth', 2)

axis([xmin xmax ymin ymax])
%}




for i = 1:10
figure
hold on
surf(X1,Y1,S_OUT(:,:,i))
shading interp
colormap(cmap)
view(2)
%{
plot3(Epsilon_plot(:,3),Epsilon_plot(:,5),ones(9,1).*max(max(S_OUT(:,:,i))),evo,'LineWidth',2)
plot3(Epsilon_plot(:,3),Epsilon_plot(:,4),ones(9,1).*max(max(S_OUT(:,:,i))),evo,'LineWidth',2)
plot3(Epsilon_plot(:,3),Epsilon_plot(:,1),ones(9,1).*max(max(S_OUT(:,:,i))),strcat('--',evo),'LineWidth',2)
plot3(Epsilon_plot(:,3),Epsilon_plot(:,2),ones(9,1).*max(max(S_OUT(:,:,i))),strcat('--',evo),'LineWidth',2)
plot3([0 DM_Slider],[Y0_u_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf],ones(2,1).*max(max(S_OUT(:,:,i))), 'Color', evo, 'LineWidth', 2)
plot3([0 DM_Slider],[Y0_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf],ones(2,1).*max(max(S_OUT(:,:,i))), 'Color', evo, 'LineWidth', 2)
plot3([0 DM_Slider],[Y0_l_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf],ones(2,1).*max(max(S_OUT(:,:,i))), 'Color', evo, 'LineWidth', 2)
%}
title(name_comp(i,1))
axis([xmin xmax ymin ymax])
end



for i = 1:N
figure
surf(X1,Y1,density1(:,:,i))
shading interp
colormap(cmap)
view(2)
title(Name(i,1))
axis([xmin xmax ymin ymax])
end

figure
ss = sum(S_OUT,3)./15;
surf(X1,Y1,ss)
colormap(cmap)
shading interp
view(2)
title('Similarity Sum')
axis([xmin xmax ymin ymax])









%{
figure
hold on
rectangle('Position',[200,-60,100,80],'FaceColor','y','EdgeColor','w') % Pangea
rectangle('Position',[480,-60,190,80],'FaceColor','y','EdgeColor','w') % Gondwana
rectangle('Position',[900,-60,400,80],'FaceColor','y','EdgeColor','w') % Rodinia
rectangle('Position',[1450,-60,525,80],'FaceColor','y','EdgeColor','w') % Columbia/Nuna
rectangle('Position',[2400,-60,300,80],'FaceColor','y','EdgeColor','w') % Kenorland?
axis([xmin xmax ymin ymax])











for j = 1:N
	for k = 1:N
		for m = 1:size(density1,1)
			for n = 1:size(density1,2)
				Lall(m,n) = abs(density1(m,n,j) - density1(m,n,k)); % Mismatch map
			end
		end
		L2D_tmp = sum(Lall);
		M2D(j,k) = sum(L2D_tmp)/2;
		L2D(j,k) = 1 - (sum(L2D_tmp)/2);
	end
end


for j = 1:N
	for k = 1:N
		xx = reshape(density1(:,:,j),size(density1,1)*size(density1,2),1);
		yy = reshape(density1(:,:,k),size(density1,1)*size(density1,2),1);
		R22D(j,k) = ((sum((xx - mean(xx)).*(yy - mean(yy))))/(sqrt((sum((xx - mean(xx)).*(xx - mean(xx))))*(sum((yy - mean(yy)).*(yy - mean(yy)))))))*...
			((sum((xx - mean(xx)).*(yy - mean(yy))))/(sqrt((sum((xx - mean(xx)).*(xx - mean(xx))))*(sum((yy - mean(yy)).*(yy - mean(yy)))))));
	end
end



%crit='metricstress';
crit='metricsstress';
%crit='stress';




diss = M2D;

i=1
[XY,stress(1,i),disparities] = mdscale(diss,i,'Criterion',crit);
while stress(1,i)>0.05 || i<3;
	i=i+1;
	[XY,stress(1,i),disparities] = mdscale(diss,i,'Criterion',crit);
end   

figure
colours = colormap(jet(N));
dx=0.01;
dim_in = 2;

X=XY(:,1);
Y=XY(:,2);
Z=XY(:,3);
hold on; 
set(gca,'Units', 'normalized', 'outerposition', [0.15 0.15 0.65 0.65]);
for i=1:N
	scatter3(X(i),Y(i),Z(i),250, 'MarkerFaceColor',colours(i, :),'MarkerEdgeColor','black');
end

headers = Name';
            
[rubbish,i] = sort(M2D,1,'ascend');
xlim([min(X) max(X)]);
ylim([min(Y) max(Y)]);
zlim([min(Z) max(Z)]);
YX=XY(i(2,:),1:3);
YZ=XY(i(3,:),1:3);
XZ=XY(i(1,:),1:3);
arrow3(XZ,YZ,':l1',0.4);
arrow3(XZ,YX,'k1',0.5);
text(X+dx,Y+dx,Z+dx,Name);
view(3)
title('diss = Mismatch (1 - L)')
grid on





diss = 1-R22D;
diss(diss < 0) = 0

i=1
[XY,stress(1,i),disparities] = mdscale(diss,i,'Criterion',crit);
while stress(1,i)>0.05 || i<3;
	i=i+1;
	[XY,stress(1,i),disparities] = mdscale(diss,i,'Criterion',crit);
end   

figure
colours = colormap(jet(N));
dx=0.01;
dim_in = 3;

X=XY(:,1);
Y=XY(:,2);
Z=XY(:,3);
hold on; 
set(gca,'Units', 'normalized', 'outerposition', [0.15 0.15 0.65 0.65]);
for i=1:N
	scatter3(X(i),Y(i),Z(i),250, 'MarkerFaceColor',colours(i, :),'MarkerEdgeColor','black');
end

headers = Name';
            
[rubbish,i] = sort(R22D,1,'ascend');
xlim([min(X) max(X)]);
ylim([min(Y) max(Y)]);
zlim([min(Z) max(Z)]);
YX=XY(i(2,:),1:3);
YZ=XY(i(3,:),1:3);
XZ=XY(i(1,:),1:3);
arrow3(XZ,YZ,':l1',0.4);
arrow3(XZ,YX,'k1',0.5);
text(X+dx,Y+dx,Z+dx,Name);
view(3)
title('diss = 1 - Cross corr.')
grid on




diss = 1 - S2D;
diss(diss < 0.01) = 0;

i=1
[XY,stress(1,i),disparities] = mdscale(diss,i,'Criterion',crit);
while stress(1,i)>0.05 || i<3;
	i=i+1;
	[XY,stress(1,i),disparities] = mdscale(diss,i,'Criterion',crit);
end   

figure
colours = colormap(jet(N));
dx=0.01;
dim_in = 3;

X=XY(:,1);
Y=XY(:,2);
Z=XY(:,3);
hold on; 
set(gca,'Units', 'normalized', 'outerposition', [0.15 0.15 0.65 0.65]);
for i=1:N
	scatter3(X(i),Y(i),Z(i),250, 'MarkerFaceColor',colours(i, :),'MarkerEdgeColor','black');
end

headers = Name';
            
[rubbish,i] = sort(M2D,1,'ascend');
xlim([min(X) max(X)]);
ylim([min(Y) max(Y)]);
zlim([min(Z) max(Z)]);
YX=XY(i(2,:),1:3);
YZ=XY(i(3,:),1:3);
XZ=XY(i(1,:),1:3);
arrow3(XZ,YZ,':l1',0.4);
arrow3(XZ,YX,'k1',0.5);
text(X+dx,Y+dx,Z+dx,Name);
view(3)
title('diss = 1 - Similarity')
grid on

%}





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%{
AFRh = hist(AFR(:,1),bins)';
AUSh = hist(AUS(:,1),bins)';
EURh = hist(EUR(:,1),bins)';
NAMh = hist(NAM(:,1),bins)';
SAMh = hist(SAM(:,1),bins)';

Hall = AFRh + EURh + NAMh + SAMh + AUSh;

H1 = AFRh./Hall;
H2 = AFRh./Hall + EURh./Hall;
H3 = AFRh./Hall + EURh./Hall + NAMh./Hall;
H4 = AFRh./Hall + EURh./Hall + NAMh./Hall + SAMh./Hall;

Xh = (xmax-xmin)/bins:(xmax-xmin)/bins:xmax;

figure
hold on
plot(Xh,H1)
plot(Xh,H2)
plot(Xh,H3)
plot(Xh,H4)
axis([xmin,xmax,0,1])
%}


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

%Xh = (xmax-xmin)/bins:(xmax-xmin)/bins:xmax;

Xh = 0:(xmax-xmin)/(bins-1):xmax;

%{
figure
hold on
plot(Xh,H1)
plot(Xh,H2)
plot(Xh,H3)
plot(Xh,H4)
axis([xmin,xmax,0,1])
set(gca, 'XDir','reverse')
%}

figure
hold on
patch([Xh fliplr(Xh)], [zeros(1,bins) fliplr(H2')], [69/255 35/255 125/255])
patch([Xh fliplr(Xh)], [H1' fliplr(H2')], [74/255 113/255 209/255])
patch([Xh fliplr(Xh)], [H2' fliplr(H3')], [83/255 176/255 179/255])
patch([Xh fliplr(Xh)], [H3' fliplr(H4')], [146/255 198/255 106/255])
patch([Xh fliplr(Xh)], [ones(1,bins) fliplr(H4')], [232/255 193/255 58/255])
axis([xmin,xmax,0,1])
set(gca, 'XDir','reverse')










