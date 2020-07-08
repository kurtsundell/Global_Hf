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

TITLE_MOD = '';

% kernel bandwidths
bandwidth_x = 40;
bandwidth_y = 2;


%{
xmin = 485;
xmax = 635;
ymin = -41;
ymax = 20;
%}

TITLE_MOD = '';

%{
% kernel bandwidths
bandwidth_x = 5;
bandwidth_y = .5;
%}

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

% jet colormap that clips 0 values
cmap =[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000;1,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]';





DM_Slider = 4400;

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

















% Perceptually uniform colormaps, clipped to white for values at 95%
%viridis
%cmap = [1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;1,1,1;0.280893580000000,0.0789070300000000,0.402329440000000;0.281445810000000,0.0843197000000000,0.407414040000000;0.281923580000000,0.0896662200000000,0.412415210000000;0.282327390000000,0.0949554500000000,0.417330860000000;0.282656330000000,0.100195760000000,0.422160320000000;0.282910490000000,0.105393450000000,0.426902020000000;0.283090950000000,0.110553070000000,0.431553750000000;0.283197040000000,0.115679660000000,0.436114820000000;0.283228820000000,0.120777010000000,0.440584040000000;0.283186840000000,0.125847990000000,0.444960000000000;0.283072000000000,0.130894770000000,0.449241270000000;0.282883890000000,0.135920050000000,0.453427340000000;0.282622970000000,0.140925560000000,0.457517260000000;0.282290370000000,0.145912330000000,0.461509950000000;0.281886760000000,0.150881470000000,0.465404740000000;0.281412280000000,0.155834250000000,0.469201280000000;0.280867730000000,0.160771320000000,0.472899090000000;0.280254680000000,0.165692720000000,0.476497620000000;0.279573990000000,0.170598840000000,0.479996750000000;0.278826180000000,0.175490200000000,0.483396540000000;0.278012360000000,0.180366840000000,0.486697020000000;0.277134370000000,0.185228360000000,0.489898310000000;0.276193760000000,0.190074470000000,0.493000740000000;0.275191160000000,0.194905400000000,0.496004880000000;0.274128020000000,0.199720860000000,0.498911310000000;0.273005960000000,0.204520490000000,0.501720760000000;0.271828120000000,0.209303060000000,0.504434130000000;0.270594730000000,0.214068990000000,0.507052430000000;0.269307560000000,0.218817820000000,0.509576780000000;0.267968460000000,0.223549110000000,0.512008400000000;0.266579840000000,0.228262100000000,0.514348700000000;0.265144500000000,0.232955930000000,0.516599300000000;0.263663200000000,0.237630780000000,0.518761630000000;0.262138010000000,0.242286190000000,0.520837360000000;0.260571030000000,0.246921700000000,0.522828220000000;0.258964510000000,0.251536850000000,0.524736090000000;0.257322440000000,0.256130400000000,0.526563320000000;0.255645190000000,0.260702840000000,0.528311520000000;0.253934980000000,0.265253840000000,0.529982730000000;0.252194040000000,0.269783060000000,0.531579050000000;0.250424620000000,0.274290240000000,0.533102610000000;0.248628990000000,0.278775090000000,0.534555610000000;0.246811400000000,0.283236620000000,0.535940930000000;0.244972080000000,0.287675470000000,0.537260180000000;0.243113240000000,0.292091540000000,0.538515610000000;0.241237080000000,0.296484710000000,0.539709460000000;0.239345750000000,0.300854940000000,0.540843980000000;0.237441380000000,0.305202220000000,0.541921400000000;0.235526060000000,0.309526570000000,0.542943960000000;0.233602770000000,0.313827730000000,0.543914240000000;0.231673500000000,0.318105800000000,0.544834440000000;0.229739260000000,0.322361270000000,0.545706330000000;0.227801920000000,0.326594320000000,0.546532000000000;0.225863300000000,0.330805150000000,0.547313530000000;0.223925150000000,0.334994000000000,0.548052910000000;0.221989150000000,0.339161140000000,0.548752110000000;0.220056910000000,0.343306880000000,0.549413040000000;0.218129950000000,0.347431540000000,0.550037550000000;0.216209710000000,0.351535480000000,0.550627430000000;0.214297570000000,0.355619070000000,0.551184400000000;0.212394770000000,0.359682730000000,0.551710110000000;0.210503100000000,0.363726710000000,0.552206460000000;0.208623420000000,0.367751510000000,0.552674860000000;0.206756280000000,0.371757750000000,0.553116530000000;0.204902570000000,0.375745890000000,0.553532820000000;0.203063090000000,0.379716440000000,0.553925050000000;0.201238540000000,0.383669890000000,0.554294410000000;0.199429500000000,0.387606780000000,0.554642050000000;0.197636500000000,0.391527620000000,0.554969050000000;0.195859930000000,0.395432970000000,0.555276370000000;0.194100090000000,0.399323360000000,0.555564940000000;0.192357190000000,0.403199340000000,0.555835590000000;0.190631350000000,0.407061480000000,0.556089070000000;0.188922590000000,0.410910330000000,0.556326060000000;0.187230830000000,0.414746450000000,0.556547170000000;0.185555930000000,0.418570400000000,0.556752920000000;0.183897630000000,0.422382750000000,0.556943770000000;0.182255610000000,0.426184050000000,0.557120100000000;0.180629490000000,0.429974860000000,0.557282210000000;0.179018790000000,0.433755720000000,0.557430350000000;0.177422980000000,0.437527200000000,0.557564660000000;0.175841480000000,0.441289810000000,0.557685260000000;0.174273630000000,0.445044100000000,0.557792160000000;0.172718760000000,0.448790600000000,0.557885320000000;0.171176150000000,0.452529800000000,0.557964640000000;0.169645730000000,0.456262090000000,0.558030340000000;0.168126410000000,0.459988020000000,0.558081990000000;0.166617100000000,0.463708130000000,0.558119130000000;0.165117030000000,0.467422900000000,0.558141410000000;0.163625430000000,0.471132780000000,0.558148420000000;0.162141550000000,0.474838210000000,0.558139670000000;0.160664670000000,0.478539610000000,0.558114660000000;0.159194130000000,0.482237400000000,0.558072800000000;0.157729330000000,0.485931970000000,0.558013470000000;0.156269730000000,0.489623700000000,0.557936000000000;0.154814880000000,0.493312930000000,0.557839670000000;0.153364450000000,0.497000030000000,0.557723710000000;0.151918200000000,0.500685290000000,0.557587330000000;0.150476050000000,0.504369040000000,0.557429680000000;0.149039180000000,0.508051360000000,0.557250500000000;0.147607310000000,0.511732630000000,0.557048610000000;0.146180260000000,0.515413160000000,0.556822710000000;0.144758630000000,0.519093190000000,0.556571810000000;0.143343270000000,0.522772920000000,0.556294910000000;0.141935270000000,0.526452540000000,0.555990970000000;0.140535990000000,0.530132190000000,0.555658930000000;0.139147080000000,0.533812010000000,0.555297730000000;0.137770480000000,0.537492130000000,0.554906250000000;0.136408500000000,0.541172640000000,0.554483390000000;0.135065610000000,0.544853350000000,0.554029060000000;0.133742990000000,0.548534580000000,0.553541080000000;0.132444010000000,0.552216370000000,0.553018280000000;0.131172490000000,0.555898720000000,0.552459480000000;0.129932700000000,0.559581620000000,0.551863540000000;0.128729380000000,0.563265030000000,0.551229270000000;0.127567710000000,0.566948910000000,0.550555510000000;0.126453380000000,0.570633160000000,0.549841100000000;0.125393830000000,0.574317540000000,0.549085640000000;0.124394740000000,0.578002050000000,0.548287400000000;0.123462810000000,0.581686610000000,0.547444980000000;0.122605620000000,0.585371050000000,0.546557220000000;0.121831220000000,0.589055210000000,0.545622980000000;0.121148070000000,0.592738890000000,0.544641140000000;0.120565010000000,0.596421870000000,0.543610580000000;0.120091540000000,0.600103870000000,0.542530430000000;0.119737560000000,0.603784590000000,0.541399990000000;0.119511630000000,0.607463880000000,0.540217510000000;0.119423410000000,0.611141460000000,0.538981920000000;0.119482550000000,0.614817020000000,0.537692190000000;0.119698580000000,0.618490250000000,0.536347330000000;0.120080790000000,0.622160810000000,0.534946330000000;0.120638240000000,0.625828330000000,0.533488340000000;0.121379720000000,0.629492420000000,0.531972750000000;0.122312440000000,0.633152770000000,0.530398080000000;0.123443580000000,0.636808990000000,0.528763430000000;0.124779530000000,0.640460690000000,0.527067920000000;0.126325810000000,0.644107440000000,0.525310690000000;0.128087030000000,0.647748810000000,0.523490920000000;0.130066880000000,0.651384360000000,0.521607910000000;0.132267970000000,0.655013630000000,0.519660860000000;0.134691830000000,0.658636190000000,0.517648800000000;0.137339210000000,0.662251570000000,0.515571010000000;0.140209910000000,0.665859270000000,0.513426800000000;0.143302910000000,0.669458810000000,0.511215490000000;0.146616400000000,0.673049680000000,0.508936440000000;0.150147820000000,0.676631390000000,0.506588900000000;0.153894050000000,0.680203430000000,0.504172170000000;0.157851460000000,0.683765250000000,0.501685740000000;0.162015980000000,0.687316320000000,0.499129060000000;0.166383200000000,0.690856110000000,0.496501630000000;0.170948400000000,0.694384050000000,0.493802940000000;0.175706710000000,0.697899600000000,0.491032520000000;0.180653140000000,0.701402220000000,0.488189380000000;0.185782660000000,0.704891330000000,0.485273260000000;0.191090180000000,0.708366350000000,0.482283950000000;0.196570630000000,0.711826680000000,0.479221080000000;0.202219020000000,0.715271750000000,0.476084310000000;0.208030450000000,0.718700950000000,0.472873300000000;0.214000150000000,0.722113710000000,0.469587740000000;0.220123810000000,0.725509450000000,0.466226380000000;0.226396900000000,0.728887530000000,0.462789340000000;0.232814980000000,0.732247350000000,0.459276750000000;0.239373900000000,0.735588280000000,0.455688380000000;0.246069680000000,0.738909720000000,0.452024050000000;0.252898510000000,0.742211040000000,0.448283550000000;0.259856760000000,0.745491620000000,0.444466730000000;0.266941270000000,0.748750840000000,0.440572840000000;0.274149220000000,0.751988070000000,0.436600900000000;0.281476810000000,0.755202660000000,0.432552070000000;0.288921020000000,0.758393990000000,0.428426260000000;0.296478990000000,0.761561420000000,0.424223410000000;0.304147960000000,0.764704330000000,0.419943460000000;0.311925340000000,0.767822070000000,0.415586380000000;0.319808600000000,0.770914030000000,0.411152150000000;0.327795800000000,0.773979530000000,0.406640110000000;0.335885390000000,0.777017900000000,0.402049170000000;0.344074110000000,0.780028550000000,0.397381030000000;0.352359850000000,0.783010860000000,0.392635790000000;0.360740530000000,0.785964190000000,0.387813530000000;0.369214200000000,0.788887930000000,0.382914380000000;0.377778920000000,0.791781460000000,0.377938500000000;0.386432820000000,0.794644150000000,0.372886060000000;0.395174080000000,0.797475410000000,0.367757260000000;0.404001010000000,0.800274610000000,0.362552230000000;0.412913500000000,0.803040990000000,0.357268930000000;0.421908130000000,0.805774120000000,0.351910090000000;0.430983170000000,0.808473430000000,0.346476070000000;0.440136910000000,0.811138360000000,0.340967300000000;0.449367630000000,0.813768350000000,0.335384260000000;0.458673620000000,0.816362880000000,0.329727490000000;0.468053140000000,0.818921430000000,0.323997610000000;0.477504460000000,0.821443510000000,0.318195290000000;0.487025800000000,0.823928620000000,0.312321330000000;0.496615360000000,0.826376330000000,0.306376610000000;0.506271300000000,0.828786210000000,0.300362110000000;0.515991820000000,0.831157840000000,0.294278880000000;0.525776220000000,0.833490640000000,0.288126500000000;0.535621100000000,0.835784520000000,0.281908320000000;0.545524400000000,0.838039180000000,0.275626020000000;0.555483970000000,0.840254370000000,0.269281470000000;0.565497600000000,0.842429900000000,0.262876830000000;0.575562970000000,0.844565610000000,0.256414570000000;0.585677720000000,0.846661390000000,0.249897480000000;0.595839340000000,0.848717220000000,0.243328780000000;0.606045280000000,0.850733100000000,0.236712140000000;0.616292830000000,0.852709120000000,0.230051790000000;0.626579230000000,0.854645430000000,0.223352580000000;0.636901570000000,0.856542260000000,0.216620120000000;0.647256850000000,0.858399910000000,0.209860860000000;0.657641970000000,0.860218780000000,0.203082290000000;0.668053690000000,0.861999320000000,0.196293070000000;0.678488680000000,0.863742110000000,0.189503260000000;0.688943510000000,0.865447790000000,0.182724550000000;0.699414630000000,0.867117110000000,0.175970550000000;0.709898420000000,0.868750920000000,0.169257120000000;0.720391150000000,0.870350150000000,0.162602730000000;0.730889020000000,0.871915840000000,0.156028940000000;0.741388030000000,0.873449180000000,0.149561010000000;0.751884140000000,0.874951430000000,0.143228280000000;0.762373420000000,0.876423920000000,0.137064490000000;0.772851830000000,0.877868080000000,0.131108640000000;0.783315350000000,0.879285450000000,0.125405380000000;0.793759940000000,0.880677630000000,0.120005320000000;0.804181590000000,0.882046320000000,0.114965050000000;0.814576340000000,0.883393290000000,0.110346780000000;0.824940280000000,0.884720360000000,0.106217240000000;0.835269590000000,0.886029430000000,0.102645900000000;0.845560560000000,0.887322430000000,0.0997021900000000;0.855809600000000,0.888601340000000,0.0974518600000000;0.866013250000000,0.889868150000000,0.0959527700000000;0.876168240000000,0.891124870000000,0.0952504600000000;0.886271460000000,0.892373530000000,0.0953743900000000;0.896320020000000,0.893616140000000,0.0963353800000000;0.906311210000000,0.894854670000000,0.0981249600000000;0.916242120000000,0.896091270000000,0.100716800000000;0.926105790000000,0.897329770000000,0.104070670000000;0.935904440000000,0.898570400000000,0.108130940000000;0.945636260000000,0.899815000000000,0.112837730000000;0.955299720000000,0.901065340000000,0.118128320000000;0.964893530000000,0.902323110000000,0.123940510000000;0.974416650000000,0.903589910000000,0.130214940000000;0.983868290000000,0.904867260000000,0.136896710000000;0.993247890000000,0.906156570000000,0.143936200000000];

% make bivariate kdes for samples, save as 3D matrix block
for k = 1:N
	data2 = data1(:,k*2-1:k*2);
	data2 = data2(any(data2 ~= 0,2),:);
	[bandwidth1,density1(:,:,k),X1,Y1] = kde2d_set_kernel(data2, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y);
	density1(:,:,k) = density1(:,:,k)./sum(sum(density1(:,:,k)));
	max_density1 = max(max(density1(:,:,k))); 
	max_density_conf(k,1) = max_density1*(1/length(cmap(:,1))); %clip lowest colormap value to avoid excessive use of color
	clear data2
end

% Ask user if they want to plot all input sample bivariate KDEs?
while(1)
	choice = menu('Plot all samples?','Yes','No');
	if choice==1 
		for i = 1:N
			figure
			hold on
			surf(X1,Y1,density1(:,:,i));
			colormap(cmap)
			shading interp
			view(2)
			title(strcat(Name(i,1),'{ - }',TITLE_MOD),'FontSize',40)
			xlabel('Age (Ma)','FontSize',20)
			ylabel('eHfT','FontSize',20)
			axis([xmin xmax ymin ymax])
			
			set(gca,'FontSize',20)
			%plot3(Epsilon_plot(:,3),Epsilon_plot(:,4),ones(9,1).*max(max(density1(:,:,i))),'k','LineWidth',4)
			%plot3(Epsilon_plot(:,3),Epsilon_plot(:,4)-10,ones(9,1).*max(max(density1(:,:,i))),'k','LineWidth',4)
			%plot3(Epsilon_plot(:,3),Epsilon_plot(:,4)-20,ones(9,1).*max(max(density1(:,:,i))),'k','LineWidth',4)
			%plot3(Epsilon_plot(:,3),Epsilon_plot(:,4)-30,ones(9,1).*max(max(density1(:,:,i))),'k','LineWidth',4)
			%view(2)
			%plot3([0 DM_Slider],[Y0_u_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf],ones(2,1).*max(max(density1(:,:,i))), 'Color', 'k', 'LineWidth', 3)
			%plot3([0 DM_Slider],[Y0_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf],ones(2,1).*max(max(density1(:,:,i))), 'Color', 'k', 'LineWidth', 3)
			%plot3([0 DM_Slider],[Y0_l_Epsi_DM_176Lu_177Hf, Ys_Epsi_DM_176Lu_177Hf],ones(2,1).*max(max(density1(:,:,i))), 'Color', 'k', 'LineWidth', 3)

		end
	end
	break
end

%{
for i = 1:N
	F1 = figure;
	contour3(X1,Y1,density1(:,:,i),[max_density_conf(i,1) max_density_conf(i,1)],'k', 'LineWidth', 3);
	grid off
	view(2)
	%[file,path] = uiputfile('*.eps','Save file'); print(F1,'-depsc','-painters',[path file]); epsclean([path file]); % save simplified contours 
	axis([xmin xmax ymin ymax])
end
%}

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
			title(strcat(name_comp(i,1),'{ - }',TITLE_MOD),'FontSize',40)
			xlabel('Age (Ma)','FontSize',20)
			ylabel('eHfT','FontSize',20)
			axis([xmin xmax ymin ymax])
			set(gca,'FontSize',20)
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
		title(strcat('Similarity Sum',{' - '},TITLE_MOD),'FontSize',40)
		xlabel('Age (Ma)','FontSize',20)
		ylabel('eHfT','FontSize',20)
		axis([xmin xmax ymin ymax])
		set(gca,'FontSize',20)
		
		max_density1S = max(max(S_Map)); 
		max_density_confS(k,1) = max_density1S*.32; %clip lowest colormap value to avoid excessive use of color
		
		F1 = figure;
		contour3(X1,Y1,S_Map,[max_density_confS(i,1) max_density_confS(i,1)],'k', 'LineWidth', 3);
		grid off
		view(2)
		
		axis([xmin xmax ymin ymax])
		%[file,path] = uiputfile('*.eps','Save file'); print(F1,'-depsc','-painters',[path file]); epsclean([path file]); % save simplified contours 
		
		
		
		
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
		plot(S_time_x,S_time_ind_sum./sum(S_time_ind_sum),'Color','k','LineWidth',5)
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
		plot(S_time_x, S_time./(sum(S_time)), 'LineWidth', 4, 'Color', 'k')
		xlabel('Age Ma')
		ylabel('Sum of all comparisons')
		axis([xmin xmax 0 max(S_time./sum(S_time))])
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
		title(strcat('Data binned by sample'),'FontSize',40)
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

function epsclean( file, varargin )
% EPSCLEAN Cleans up a MATLAB exported .eps file.
%
%   EPSCLEAN(F,...) cleans the .eps file F without removing box elements and optional parameters.
%   EPSCLEAN(F,O,...) cleans the .eps file F, writes the result to file O and optional parameters.
%   EPSCLEAN(F,O,R,G) (deprecated) cleans the .eps file F, writes the result to file O and optionally removes box
%                     elements if R = true. Optionally it groups elements 'softly' if G = true.
%
%   Optional parameters (key/value pairs) - see examples below
%   - outFile      ... Defines the output file for the result. Default is overwriting the input file.
%   - groupSoft    ... Groups elements only if they occur sequentially. Can help with Z-order problems. Defaults to false.
%   - combineAreas ... Combines filled polygons to larger ones. Can help with artifacts. Defaults to false.
%   - removeBoxes  ... Removes box (rectangle) elements. Defaults to false.
%   - closeGaps    ... For every filled polygon, also draw a fine polyline to close potential gaps between adjacent polygon areas. Defaults to false.
%   - gapWidth     ... The width of polylines to cover gaps. Defaults to 0.01.
%
%   When exporting a figure with Matlab's 'saveas' function to vector graphics multiple things might occur:
%   - Paths are split up into multiple segments and white lines are created on patch objects
%     see https://de.mathworks.com/matlabcentral/answers/290313-why-is-vector-graphics-chopped-into-pieces
%   - There are unnecessary box elements surrounding the paths
%   - Lines which actually should be continuous are split up in small line segments
%
%   Especially the fragmentation is creating highly unusable vector graphics for further post-processing.
%   This function fixes already exported figures in PostScript file format by grouping paths together according to their
%   properties (line width, line color, transformation matrix, ...). Small line segments which logically should belong
%   together are replaced by one continous line.
%   It also removes paths with 're' (rectangle) elements when supplying the parameter 'removeBoxes' with true.
%   In case the 'groupSoft' parameter is true it does not group elements according to their properties over the whole
%   document. It will rather group them only if the same elements occur sequentially, but not if they are interrupted by
%   elements with different properties. This will result in more fragmentation, but the Z-order will be kept intact. Use
%   this (set to true) if you have trouble with the Z-order.
%   If the 'combineAreas' parameter is true it combines filled polygons with the same properties to larger polygons of
%   the same type. It reduces clutter and white-line artifacts. The downside is that it's about 10 times slower.
%
%   Example 1
%   ---------
%       z = peaks;
%       contourf(z);
%       print(gcf,'-depsc','-painters','out.eps');
%       epsclean('out.eps'); % cleans and overwrites the input file
%       epsclean('out.eps','clean.eps'); % leaves the input file intact
%       epsclean('out.eps','clean.eps','combineAreas',true); % result in 'clean.eps', combines polygon areas
%       epsclean('out.eps','groupSoft',true,'combineAreas',true); % overwrites file, combines polygons, Z-order preserved
%
%   Example 2
%   ---------
%       [X,Y,Z] = peaks(100);
%       [~,ch] = contourf(X,Y,Z);
%       ch.LineStyle = 'none';
%       ch.LevelStep = ch.LevelStep/10;
%       colormap('hot')
%       saveas(gcf, 'out.eps', 'epsc');
%       epsclean('out.eps');
%
%   Notes
%   -----
%   - A block is a starting with GS (gsave) and ends with GR (grestore)
%   - Only text after %%EndPageSetup is analyzed
%   - Removing boxes will also remove the clipping area (if any)
%   - Tested on Windows with Matlab R2016b
%
%   Changes
%   -------
%   2017-04-03 (YYYY-MM-DD)
%   - Line segments with the same properties are converted to one continous polyline
%      o As a side effect this will cause multiple equal lines on top of each other to merge
%   - The Z-order of elements can be preserved by using 'groupSoft = true'
%      o See https://github.com/Conclusio/matlab-epsclean/issues/6
%      o This will cause additional fragmentation which might or might not be what you want
%   2017-04-18 (YYYY-MM-DD)
%   - Major performance increase for creating the adjacency matrix (for creating continous polylines)
%   - A lot of other performance enhancements
%   2017-05-28 (YYYY-MM-DD)
%   - Added the possibility to merge adjacent polygons to avoid artifacts
%     o See https://github.com/Conclusio/matlab-epsclean/issues/9
%   - Changed argument style
%   2018-04-12 (YYYY-MM-DD)
%   - Added parameter 'closeGaps' to hide lines between filled areas
%     o See https://github.com/Conclusio/matlab-epsclean/issues/9
%   - Added parameter 'gapWidth' to control the line width
%
%   ------------------------------------------------------------------------------------------
%   Copyright 2017,2018, Stefan Spelitz, Vienna University of Technology (TU Wien).
%   This code is distributed under the terms of the GNU Lesser General Public License (LGPL).
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU Lesser General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU Lesser General Public License for more details.
% 
%   You should have received a copy of the GNU Lesser General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.

% default values:
removeBoxes = false;
groupSoft = false;
combineAreas = false;
closeGaps = false;
gapWidth = 0.01;
outfile = file;

fromIndex = 1;
% check for old argument style (backward compatibility)
if nargin >= 2 && ischar(varargin{1}) && ~strcmpi(varargin{1},'removeBoxes') && ~strcmpi(varargin{1},'groupSoft') && ~strcmpi(varargin{1},'combineAreas') && ~strcmpi(varargin{1},'closeGaps') && ~strcmpi(varargin{1},'gapWidth')
    fromIndex = 2;
    outfile = varargin{1};
    if nargin >= 3
        if islogical(varargin{2})
            fromIndex = 3;
            removeBoxes = varargin{2};
            if nargin >= 4 && islogical(varargin{3})
                fromIndex = 4;
                groupSoft = varargin{3};
            end
        end
    end
end

p = inputParser;
p.CaseSensitive = false;
p.KeepUnmatched = false;

addParameter(p,'outFile',outfile,@ischar);
addParameter(p,'removeBoxes',removeBoxes,@islogical);
addParameter(p,'groupSoft',groupSoft,@islogical);
addParameter(p,'combineAreas',combineAreas,@islogical);
addParameter(p,'closeGaps',closeGaps,@islogical);
addParameter(p,'gapWidth',gapWidth,@isfloat);

parse(p,varargin{fromIndex:end});
outfile = p.Results.outFile;
removeBoxes = p.Results.removeBoxes;
groupSoft = p.Results.groupSoft;
combineAreas = p.Results.combineAreas;
closeGaps = p.Results.closeGaps;
gapWidth = p.Results.gapWidth;

keepInput = true;
if strcmp(file, outfile)
    outfile = [file '_out']; % tmp file
    keepInput = false;
end

fid1 = fopen(file,'r');
fid2 = fopen(outfile,'W');

previousBlockPrefix = [];
operation = -1; % -1 .. wait for 'EndPageSetup', 0 .. wait for blocks, 1 .. create id, 2 .. analyze block content, 3 .. analyzed
insideAxg = false;
blockGood = true;
hasLineCap = false;
isDashMode = false;
blockList = [];

nested = 0;
lastMLine = [];
lastLLine = [];
blockMap = containers.Map(); % key=blockPrefix -> MAP with connection information and content for blocks

% current block (cb) data:
cbPrefix = '';
cbContentLines = -ones(1,100);
cbContentLinesFull = -ones(1,100);
cbContentLinesIdx = 1;
cbContentLinesFullIdx = 1;
cbConn = {};
cbIsFill = false;

% load whole file into memory:
fileContent = textscan(fid1,'%s','delimiter','\n','whitespace','');
fileContent = fileContent{1}';
lineCount = length(fileContent);
lineIdx = 0;

while lineIdx < lineCount
    lineIdx = lineIdx + 1;
    thisLine = cell2mat(fileContent(lineIdx));
    
    % normal read until '%%EndPageSetup'
    if operation == -1
        if closeGaps && startsWith(thisLine,'/f/fill')
            fileContent(lineIdx) = { sprintf('/f{GS %.5f setlinewidth S GR fill}bd', gapWidth) };
        elseif equalsWith(thisLine, '%%EndPageSetup')
            operation = 0;
            fprintf(fid2, '%s\n', strjoin(fileContent(1:lineIdx),'\n')); % dump prolog
        end
        continue;
    end
    
    if operation == 3 % block was analyzed
        if blockGood
            if groupSoft && ~strcmp(cbPrefix, previousBlockPrefix)
                % SOFT GROUPING. different block -> dump all existent ones except the current one

                currentBlock = [];
                if blockMap.isKey(cbPrefix)
                    currentBlock = blockMap(cbPrefix);
                    blockMap.remove(cbPrefix);
                end
                
                writeBlocks(blockList, blockMap, fid2, fileContent);
                
                blockList = [];
                blockMap = containers.Map();
                if ~isempty(currentBlock)
                    blockMap(cbPrefix) = currentBlock;
                end
            end

            [cbNewBlock,oldConn,oldConnFill] = getBlockData(blockMap,cbPrefix);
            removeLastContentLine = false;
            if cbIsFill
                if combineAreas
                    oldConnFill = [oldConnFill cbConn]; %#ok<AGROW>
                else
                    removeLastContentLine = true;
                end
            else
                oldConn = [oldConn cbConn]; %#ok<AGROW>
            end
            setBlockData(blockMap,cbPrefix,cbContentLines(1:cbContentLinesIdx-1),oldConn,oldConnFill,removeLastContentLine);
            if cbNewBlock
                % new block
                block = struct('prefix', cbPrefix);
                blockList = [blockList block]; %#ok<AGROW>
            end
        end
        operation = 0;
        previousBlockPrefix = cbPrefix;
        cbPrefix = '';
    end


    if operation == 0 % waiting for blocks
        if equalsWith(thisLine,'GS')
            % start of a block
            operation = 1;
            hasLineCap = false;
            isDashMode = false;
            nested = 0;
        elseif equalsWith(thisLine,'%%Trailer')
            % end of figures -> dump all blocks
            writeBlocks(blockList, blockMap, fid2, fileContent);
            fprintf(fid2, '%s\n', thisLine);
        elseif equalsWith(thisLine,'GR')
            % unexpected GR before a corresponding GS -> ignore
        else
            % not inside a block and not the start of a block -> just take it
            fprintf(fid2, '%s\n', thisLine);
        end
    elseif operation == 1 % inside GS/GR block
        % build prefix
        if startsWith(thisLine,'%AXGBegin')
            % this could be the beginning of a raw bitmap data block -> just take it
            insideAxg = true;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif startsWith(thisLine,'%AXGEnd')
            insideAxg = false;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif insideAxg
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif equalsWith(thisLine,'N')
            % begin analyzing
            operation = 2;
            blockGood = true;
            cbContentLinesIdx = 1;
            cbContentLinesFullIdx = 1;
            lastMLine = [];
            cbConn = {};
            cbIsFill = false;
        elseif equalsWith(thisLine,'GS')
            nested = nested + 1;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif equalsWith(thisLine,'GR')
            nested = nested - 1;
            if nested >= 0
                cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
            else
                % end of block without a 'N' = newpath command
                % we don't know what it is, but we take it as a whole
                operation = 3;
                blockGood = true;
                cbContentLinesIdx = 1;
                cbContentLinesFullIdx = 1;
                cbConn = {};
                cbIsFill = false;
            end
        elseif endsWith(thisLine,'setdash')
            isDashMode = true;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif endsWith(thisLine,'setlinecap')
            hasLineCap = true;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif endsWith(thisLine,'LJ')
            if hasLineCap
                cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
            elseif ~isDashMode
                % add '1 linecap' if no linecap is specified
                cbPrefix = sprintf('%s%s\n%s\n',cbPrefix,'1 setlinecap',thisLine);
            end
        else
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        end
    elseif operation == 2 % analyze block content
        if startsWith(thisLine,'%AXGBegin')
            % this could be the beginning of a raw bitmap data block -> just take it
            insideAxg = true;
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif startsWith(thisLine,'%AXGEnd')
            insideAxg = false;
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif insideAxg
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif endsWith(thisLine,'re')
            if removeBoxes
                blockGood = false;
            else
                [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
            end
        elseif equalsWith(thisLine,'clip')
            cbPrefix = sprintf('%sN\n%s\n%s\n', cbPrefix, strjoin(fileContent(cbContentLinesFull(1:cbContentLinesFullIdx-1))), thisLine);
            cbContentLinesIdx = 1;
            cbContentLinesFullIdx = 1;
            cbConn = {};
            cbIsFill = false;
        elseif endsWith(thisLine,'M')
            lastMLine = thisLine;
            lineIdx = lineIdx + 1;
            nextline = cell2mat(fileContent(lineIdx)); % ASSUMPTION: there is an L directly after an M
            lastLLine = nextline;
            
            moveId = thisLine(1:end-1);
            lineId = nextline(1:end-1);
            
            [cbConn] = addConnection(moveId,lineId,cbConn);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx-1,false);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
        elseif equalsWith(thisLine,'cp')
            moveId = lastLLine(1:end-1);
            lineId = lastMLine(1:end-1);
            lastLLine = lastMLine;

            [cbConn] = addConnection(moveId,lineId,cbConn);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
        elseif endsWith(thisLine,'L')
            moveId = lastLLine(1:end-1);
            lineId = thisLine(1:end-1);
            lastLLine = thisLine;

            [cbConn] = addConnection(moveId,lineId,cbConn);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
        elseif equalsWith(thisLine,'S')
            % ignore stroke command
        elseif equalsWith(thisLine,'f')
            % special handling for filled areas
            cbIsFill = true;
            if combineAreas
                lastLine = cell2mat(fileContent(lineIdx-1));
                if ~equalsWith(lastLine, 'cp')
                    [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
                end
            else
                [~,cbContentLinesFull,~,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
                cbContentLines = cbContentLinesFull;
                cbContentLinesIdx = cbContentLinesFullIdx;
                % remove all connections:
                cbConn = {};
            end
        elseif equalsWith(thisLine,'GS')
            nested = nested + 1;
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif equalsWith(thisLine,'GR')
            % end of block content
            nested = nested - 1;
            if nested >= 0
                [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
            else
                operation = 3; % end of block content
            end
        else
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        end
    end
        
end %while

fclose(fid1);
fclose(fid2);

if ~keepInput
    delete(file);
    movefile(outfile, file, 'f');
end

end

function r = startsWith(string1, pattern)
    l = length(pattern);
    if length(string1) < l
        r = false;
    else
        r = strcmp(string1(1:l),pattern);
    end
end

function r = endsWith(string1, pattern)
    l = length(pattern);
    if length(string1) < l
        r = false;
    else
        r = strcmp(string1(end-l+1:end),pattern);
    end
end

function r = equalsWith(string1, pattern)
    r = strcmp(string1,pattern);
end

function [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,both)
    if cbContentLinesFullIdx > length(cbContentLinesFull)
        cbContentLinesFull = [cbContentLinesFull -ones(1,100)];
    end
    cbContentLinesFull(cbContentLinesFullIdx) = lineIdx;
    cbContentLinesFullIdx = cbContentLinesFullIdx + 1;

    if both
        if cbContentLinesIdx > length(cbContentLines)
            cbContentLines = [cbContentLines -ones(1,100)];
        end
        cbContentLines(cbContentLinesIdx) = lineIdx;
        cbContentLinesIdx = cbContentLinesIdx + 1;
    end
end

function setBlockData(blockMap,blockId,contentLines,conn,connFill,removeLastContentLine)
    if ~blockMap.isKey(blockId)
        return; % a block without nodes. shouldn't happen.
    end
    theblock = blockMap(blockId);
    if removeLastContentLine
        theblock.contentLines = [theblock.contentLines(1:end-1) contentLines];
    else
        theblock.contentLines = [theblock.contentLines contentLines];
    end    
    theblock.conn = conn;
    theblock.connFill = connFill;
    blockMap(blockId) = theblock; %#ok<NASGU>
end

function [newBlock,conn,connFill] = getBlockData(blockMap,blockId)
    if blockMap.isKey(blockId)
        newBlock = false;
        theblock = blockMap(blockId);
        conn = theblock.conn;
        connFill = theblock.connFill;
    else
        newBlock = true;
        conn = {};
        connFill = {};
        
        s = struct();
        s.contentLines = [];
        s.conn = conn;
        s.connFill = connFill;
        
        blockMap(blockId) = s; %#ok<NASGU>
    end
end

function [conn] = addConnection(nodeId1, nodeId2, conn)
    conn{1,end+1} = nodeId1; % from
    conn{2,end}   = nodeId2; % to
end

function [am,idx2idArray,edge2idxMat,connCount,total] = buildAdjacencyMatrix(conn)
    if isempty(conn)
        am = [];
        idx2idArray = [];
        edge2idxMat = [];
    else
        from = conn(1,:); % from nodes
        to = conn(2,:); % to nodes
        [idx2idArray,~,ic] = unique([from to]);

        fromIdx = ic(1:length(from));
        toIdx = ic(length(from)+1:end);
        edge2idxMat = [fromIdx' ; toIdx'];

        nodeCount = max(ic);
        am = false(nodeCount); % adjacency matrix

        idx1 = sub2ind(size(am),fromIdx,toIdx);
        idx2 = sub2ind(size(am),toIdx,fromIdx);
        idxD = sub2ind(size(am),1:nodeCount,1:nodeCount);
        am(idx1) = true;
        am(idx2) = true;
        am(idxD) = false; % diagonal
    end

    connCount = sum(am,1);
    total = sum(connCount,2);
end

function printLines(fileId,am,idx2idArray,connCount,total)
    if total == 0
        return;
    end

    fprintf(fileId, 'N\n');

    [~,sidx] = sort(connCount);
    for ni = sidx
        firstNode = -1;
        first = true;
        search = true;
        node = ni;

        while(search)
            neighbours = find(am(node,:));
            search = false;
            for nni = neighbours
                if ~am(node,nni)
                    continue; % edge visited
                end
                if first
                    fprintf(fileId, '%sM\n', cell2mat(idx2idArray(node)));
                    first = false;
                    firstNode = node;
                end
                am(node,nni) = false;
                am(nni,node) = false;
                if nni == firstNode
                    % closed path (polygon) -> use a 'closepath' command instead of a line
                    fprintf(fileId, 'cp\n');
                else
                    fprintf(fileId, '%sL\n', cell2mat(idx2idArray(nni)));
                end
                node = nni;
                search = true;
                break;
            end
        end
    end
    
    fprintf(fileId, 'S\n');
end

function printFills(fileId,am,idx2idArray,total,edge2idxMat)
    if total == 0
        return;
    end
    
    edgepolymat = zeros(size(am));
    edgeusemat = zeros(size(am));
    
    nodeCount = size(idx2idArray,2);
    edgeCount = size(edge2idxMat,2);
    polyIdxs = zeros(1,edgeCount);

    % determine connections -> polygon:
    polyIdx = 0;
    edge = 1;
    while true
        if edge <= edgeCount
            startIdx = edge2idxMat(1,edge);
        else
            break;
        end
        polyIdx = polyIdx + 1;
        
        while edge <= size(edge2idxMat,2)
            tidx = edge2idxMat(2,edge);
            polyIdxs(edge) = polyIdx;
            
            edge = edge + 1;
            if startIdx == tidx
                break; % polygon finished
            end
        end
    end
    
    % check whether or not a polygon has the same edge defined twice
    polyCount = polyIdx;
    selfEdges = false(1,polyCount);
    for ii = 1:polyCount
        selfEdges(ii) = hasEdgeWithItself(edge2idxMat,polyIdxs,ii);
    end    
    
    % check if there are initial self edges and if so, just pretend we have been visiting those polygons already:
    k=find(selfEdges);
    for kk = k
        ii = edge2idxMat(:,polyIdxs == kk);
        idxs1 = sub2ind(size(edgeusemat), ii(1,:), ii(2,:));
        idxs2 = sub2ind(size(edgeusemat), ii(2,:), ii(1,:));
        idxs = [idxs1 idxs2];
        edgeusemat(idxs) = edgeusemat(idxs) + 1;
        edgeusemat(idxs) = edgeusemat(idxs) + 1;
        edgepolymat(idxs) = kk;
        edgepolymat(idxs) = kk;
    end
    
    
    polyIdx = 0;
    edge = 1;
    initialEdgeCount = size(edge2idxMat,2);
    while true
        if edge <= initialEdgeCount
            startIdx = edge2idxMat(1,edge);
        else
            break;
        end
        polyIdx = polyIdx + 1;
        
        if selfEdges(polyIdx)
            % polygon has edge with itself, don't try to merge and skip polygon instead
            edge = edge + find(edge2idxMat(2,edge:end) == tidx,1);
        else
            handledPolyMap = containers.Map('KeyType','double','ValueType','any');

            while edge <= initialEdgeCount
                fidx = edge2idxMat(1,edge);
                tidx = edge2idxMat(2,edge);
                                
                removeEdge = false;
                nPolyIdx = edgepolymat(fidx,tidx);
                if nPolyIdx > 0
                    if ~selfEdges(nPolyIdx)
                        if handledPolyMap.isKey(nPolyIdx)
                            % leave the edge intact, except if it's connected to the shared edge
                            val = handledPolyMap(nPolyIdx);
                            f = val(1);
                            t = val(2);
                            connected = true;
                            if f == fidx
                                f = tidx;
                            elseif f == tidx
                                f = fidx;
                            elseif t == fidx
                                t = tidx;
                            elseif t == tidx
                                t = fidx;
                            else
                                connected = false;
                            end
                            if connected
                                fusage = sum(edgeusemat(fidx,:) > 0);
                                tusage = sum(edgeusemat(tidx,:) > 0);
                                removeEdge = (fusage == 1 || tusage == 1);
                                if removeEdge
                                    handledPolyMap(nPolyIdx) = [f t];
                                end
                            end
                        else
                            % remove the first common shared edge
                            handledPolyMap(nPolyIdx) = [fidx tidx];
                            removeEdge = true;
                        end
                    end
                else
                    edgepolymat(fidx,tidx) = polyIdx;
                    edgepolymat(tidx,fidx) = polyIdx;
                end
                
                if removeEdge
                    edgepolymat(fidx,tidx) = 0;
                    edgepolymat(tidx,fidx) = 0;
                    edgeusemat(fidx,tidx) = 0;
                    edgeusemat(tidx,fidx) = 0;
                    polyIdxs(edge) = 0;
                else
                    edgeusemat(fidx,tidx) = edgeusemat(fidx,tidx) + 1;
                    edgeusemat(tidx,fidx) = edgeusemat(tidx,fidx) + 1;
                end
                
                edge = edge + 1;
                if startIdx == tidx
                    break; % polygon finished
                end
            end

            % merge all handled polygons:
            for k = cell2mat(handledPolyMap.keys())
                edgepolymat(edgepolymat == k) = polyIdx;
                polyIdxs(polyIdxs == k) = polyIdx;
            end
            selfEdges(polyIdx) = hasEdgeWithItself(edge2idxMat,polyIdxs,polyIdx);
        end
    end
    
        
    
    connCount = sum(edgeusemat, 1);

    coordinates = zeros(nodeCount,2);
    remainingNodes = find(connCount);
    for c = remainingNodes
        coordinates(c,:) = extractCoords(idx2idArray(c));
    end

    fprintf(fileId, 'N\n');

    [~,sidx] = sort(connCount); % sort by lowest connection count
    for ni = sidx
        firstNode = -1;
        prevNode = -1;
        first = true;
        search = true;
        node = ni;
        unkLeftRight = 0;

        while(search)
            c = edgeusemat(node,:);
            [~,sidx2] = sort(c(c>0),'descend'); % sort by edge-usage (select higher usage first)
            neighbours = find(c);
            neighbours = neighbours(sidx2);
            neighbours(neighbours == prevNode) = []; % don't go backwards
            search = false;
            nidx = 0;
            for nni = neighbours
                nidx = nidx + 1;
                if edgeusemat(node,nni) == 0
                    continue; % edge already visited
                end
                
                if length(neighbours) >= 2
                    if unkLeftRight > 0
                        p = coordinates(prevNode,:);
                        c = coordinates(node,:);
                        n = coordinates(nni,:);
                        
                        valid = true;
                        for nni2 = neighbours
                            if nni2 == nni
                                continue;
                            end
                            
                            a = coordinates(nni2,:);
                            leftRight = isNodeRight(p,c,n,a);

                            if unkLeftRight ~= leftRight
                                valid = false;
                                break;
                            end
                        end
                        
                        if ~valid
                            continue; % other neighbour
                        end
                    elseif edgeusemat(node,nni) == 2 && prevNode ~= -1
                        % a double edge with more than one option -> remember which way we go (ccw or cw)
                        p = coordinates(prevNode,:); % previous node
                        c = coordinates(node,:); % current node
                        n = coordinates(nni,:); % next node
                        a = coordinates(neighbours(1 + ~(nidx-1)),:); % alternative node
                        
                        unkLeftRight = isNodeRight(p,c,n,a);
                    end
                end
                
                if first
                    fprintf(fileId, '%sM\n', cell2mat(idx2idArray(node)));
                    first = false;
                    firstNode = node;
                end
                
                edgeusemat(node,nni) = edgeusemat(node,nni) - 1;
                edgeusemat(nni,node) = edgeusemat(nni,node) - 1;
                if nni == firstNode
                    % closed path (polygon) -> use a 'closepath' command instead of a line
                    fprintf(fileId, 'cp\n');
                else
                    fprintf(fileId, '%sL\n', cell2mat(idx2idArray(nni)));
                end
                prevNode = node;
                node = nni;
                search = true;
                break;
            end
        end
    end
    
    fprintf(fileId, 'f\n');
end

function value = hasEdgeWithItself(id2idxMat,polyIdxs,polyIdx)
    % check if same edge exists twice in polygon
    edgePoly = id2idxMat(:,polyIdxs == polyIdx);
    edgePoly2 = [edgePoly(2,:) ; edgePoly(1,:)];
    [~,~,ic] = unique([edgePoly' ; edgePoly2'],'rows');
    ic = accumarray(ic,1); % count the number of identical elements
    value = any(ic > 1);
end

function leftRight = isNodeRight(p,c,n,a)
    v1 = c - p; v1 = v1 ./ norm(v1);
    v2 = n - c; v2 = v2 ./ norm(v2);
    v3 = a - c; v3 = v3 ./ norm(v3);

    s2 = sign(v2(1) * v1(2) - v2(2) * v1(1));
    side = s2 - sign(v3(1) * v1(2) - v3(2) * v1(1));
    if side == 0
        % both vectors on the same side
        if s2 == 1
            % both vectors left
            right = dot(v1,v2) > dot(v1,v3);
        else
            % both vectors right
            right = dot(v1,v2) < dot(v1,v3);
        end
    else
        right = side < 0;
    end
    
    leftRight = 1;
    if right
        leftRight = 2;
    end
end

function p = extractCoords(nodeId)
    nodeId = cell2mat(nodeId);
    k = strfind(nodeId, ' ');
    x = str2double(nodeId(1:k(1)));
    y = str2double(nodeId(k(1)+1:end));
    p = [x y];
end

function writeBlocks(blockList, blockMap, fileId, fileContent)
    for ii = 1:length(blockList)
        blockId = blockList(ii).prefix;
        fprintf(fileId, 'GS\n%s', blockId);
        
        theblock = blockMap(blockId);
        contentLines = theblock.contentLines;

        % build adjacency matrix from connections:
        [amL,idx2idArrayL,~,connCountL,totalL] = buildAdjacencyMatrix(theblock.conn);
        [amF,idx2idArrayF,edge2idxMatF,~,totalF] = buildAdjacencyMatrix(theblock.connFill);
        
        total = totalL + totalF;

        if total == 0
            if ~isempty(contentLines)
                if isempty(regexp(blockId, sprintf('clip\n$'), 'once')) % prefix does not end with clip
                    fprintf(fileId, 'N\n');
                end

                fprintf(fileId, '%s\n', strjoin(fileContent(contentLines),'\n'));
            end
        else
            printLines(fileId,amL,idx2idArrayL,connCountL,totalL);
            printFills(fileId,amF,idx2idArrayF,totalF,edge2idxMatF);

            if ~isempty(contentLines)
                fprintf(fileId, '%s\n', strjoin(fileContent(contentLines),'\n'));
            end
        end

        fprintf(fileId, 'GR\n');
    end
end




