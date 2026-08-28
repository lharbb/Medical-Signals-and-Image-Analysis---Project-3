%% PROJECT 3 - PHASE 2
% Filtered Backprojection (FBP)
%
% REQUIREMENTS COVERED:
% 1) FBP from scratch
% 2) Ram-Lak, Shepp-Logan, Cosine, Hamming-windowed ramp
% 3) Quantitative noise-resolution trade-off using a resolution phantom
%    and a uniform region for noise measurement
% 4) 1000, 500, 250, 100, 50, 25 views
% 5) Quantitative streak-artifact/MSE study
% 6) Limited-angle: 180, 120, 90 degrees
% 7) Directional loss of information
% 8) Comparison against ground-truth/reference images
%
% Run Phase 1 first so Project3_Phase1_Final.mat exists.
% No radon(), iradon(), deconvwnr(), or deconvreg() are used.

clear; clc; close all;
rng(2);

N = 362;
D = 513;
numViews = 1000;
theta = linspace(0,179,numViews);

views = [1000 500 250 100 50 25];
limitedAngles = [180 120 90];
filters = {'Ram-Lak','Shepp-Logan','Cosine','Hamming'};

fprintf('\n===== PROJECT 3 - PHASE 2 =====\n');

%% 1. LOAD PHASE 1 DATA
if ~isfile('Project3_Phase1_Final.mat')
    error('Project3_Phase1_Final.mat not found. Run Project_Phase_1.m first.');
end

load('Project3_Phase1_Final.mat','phantomImg','sinoPhantom','s',...
    'GT','computedSino');

images = cell(4,1);
names = {'Shepp-Logan Phantom','Ground Truth 000',...
         'Ground Truth 001','Ground Truth 002'};
images{1} = phantomImg;
for i=1:3
    images{i+1} = GT{i};
end

sinograms = cell(4,1);
sinograms{1} = sinoPhantom;
for i=1:3
    sinograms{i+1} = computedSino{i};
end

%% 2. FOUR FILTER FBP FOR PHANTOM + THREE GROUND-TRUTH SLICES
filterMSE = zeros(4,4);
filterPSNR = zeros(4,4);
allRecon = cell(4,4);

for im = 1:4
    f = images{im};
    fprintf('\n--- %s ---\n',names{im});

    figure('Name',sprintf('Phase 2 - Four Filters - %s',names{im}));
    for k=1:4
        fsino = filterSinogram(sinograms{im},filters{k});
        r = normalizeImage(myBackprojection(fsino,theta,N,s));
        allRecon{im,k}=r;
        filterMSE(im,k)=imageMSE(f,r);
        filterPSNR(im,k)=imagePSNR(f,r);

        subplot(2,2,k);
        imagesc(r); axis image off; colormap gray; colorbar;
        title(sprintf('%s\nMSE %.5g | PSNR %.2f dB',...
            filters{k},filterMSE(im,k),filterPSNR(im,k)));
    end
    sgtitle(sprintf('Filtered Backprojection - %s',names{im}));

    for k=1:4
        fprintf('%-12s MSE %.6g | PSNR %.3f dB\n',...
            filters{k},filterMSE(im,k),filterPSNR(im,k));
    end
end

%% 3. NOISE-RESOLUTION TRADE-OFF
% A controlled resolution phantom is used. The central uniform disk
% supplies a homogeneous ROI for noise standard deviation.
resolutionPhantom = makeResolutionPhantom(N);

[resolutionSino,sRes] = myRadon(resolutionPhantom,theta,D);

% Gaussian noise added to projection data for controlled comparison.
noiseSigma = 0.05*max(resolutionSino(:));
noisyResolutionSino = resolutionSino + noiseSigma*randn(size(resolutionSino));

noiseStd = zeros(4,1);
resolutionMSE = zeros(4,1);
resolutionPSNR = zeros(4,1);

% Uniform ROI deliberately placed away from line-pair structures.
roi = false(N,N);
[X,Y] = meshgrid(1:N,1:N);
cx=(N+1)/2; cy=(N+1)/2;
roi = (X-(cx-85)).^2 + (Y-(cy+85)).^2 <= 25^2;

figure('Name','Phase 2 - Resolution Phantom');
subplot(1,3,1); imagesc(resolutionPhantom); axis image off; colormap gray;
title('Resolution phantom');
subplot(1,3,2); imagesc(theta,sRes,resolutionSino); axis xy; colormap gray;
title('Clean sinogram');
subplot(1,3,3); imagesc(theta,sRes,noisyResolutionSino); axis xy; colormap gray;
title('Noisy sinogram');

figure('Name','Phase 2 - Noise Resolution');
for k=1:4
    fs=filterSinogram(noisyResolutionSino,filters{k});
    r=normalizeImage(myBackprojection(fs,theta,N,sRes));

    noiseStd(k)=std(r(roi));
    resolutionMSE(k)=imageMSE(resolutionPhantom,r);
    resolutionPSNR(k)=imagePSNR(resolutionPhantom,r);

    subplot(2,2,k);
    imagesc(r); axis image off; colormap gray; colorbar;
    title(sprintf('%s\nNoise SD %.5g | MSE %.5g',...
        filters{k},noiseStd(k),resolutionMSE(k)));
end
sgtitle('Noise-Resolution Trade-off: Resolution Phantom');

figure('Name','Phase 2 - Noise Resolution Quantification');
yyaxis left;
plot(1:4,noiseStd,'o-','LineWidth',1.5);
ylabel('Noise SD in uniform ROI');
yyaxis right;
plot(1:4,resolutionMSE,'s-','LineWidth',1.5);
ylabel('MSE to resolution phantom');
grid on;
xticks(1:4); xticklabels(filters);
xlabel('FBP filter');
title('Quantitative Noise-Resolution Trade-off');

fprintf('\nNoise-resolution trade-off:\n');
for k=1:4
    fprintf('%-12s Noise SD %.6g | MSE %.6g | PSNR %.3f dB\n',...
        filters{k},noiseStd(k),resolutionMSE(k),resolutionPSNR(k));
end

%% 4. ANGULAR UNDERSAMPLING
underMSE = zeros(4,numel(views));
underPSNR = zeros(4,numel(views));

for im=1:4
    f=images{im};
    figure('Name',sprintf('Phase 2 - Angular Undersampling - %s',names{im}));

    for j=1:numel(views)
        nv=views(j);
        idx=round(linspace(1,numViews,nv));
        th=theta(idx);
        ss=sinograms{im}(:,idx);

        fs=filterSinogram(ss,'Ram-Lak');
        r=normalizeImage(myBackprojection(fs,th,N,s));

        underMSE(im,j)=imageMSE(f,r);
        underPSNR(im,j)=imagePSNR(f,r);

        subplot(2,3,j);
        imagesc(r); axis image off; colormap gray;
        title(sprintf('%d views\nMSE %.5g',nv,underMSE(im,j)));
    end
    sgtitle(sprintf('Angular Undersampling - %s',names{im}));

    figure('Name',sprintf('Phase 2 - MSE vs Views - %s',names{im}));
    plot(views,underMSE(im,:),'o-','LineWidth',1.5);
    set(gca,'XDir','reverse'); grid on;
    xlabel('Number of projection views');
    ylabel('MSE');
    title(sprintf('Quantitative Streak/Undersampling Study - %s',names{im}));
end

%% 5. STREAK ARTIFACT ONSET: MEAN ABSOLUTE ERROR
% MSE is the primary quantitative metric. MAE is also reported as a
% direct measure of streak-related image error.
underMAE=zeros(4,numel(views));
for im=1:4
    f=images{im};
    for j=1:numel(views)
        nv=views(j);
        idx=round(linspace(1,numViews,nv));
        th=theta(idx);
        ss=sinograms{im}(:,idx);
        r=normalizeImage(myBackprojection(filterSinogram(ss,'Ram-Lak'),th,N,s));
        underMAE(im,j)=mean(abs(f(:)-r(:)));
    end
end

%% 6. LIMITED-ANGLE STUDY
limitedMSE=zeros(4,numel(limitedAngles));
limitedPSNR=zeros(4,numel(limitedAngles));

for im=1:4
    f=images{im};
    figure('Name',sprintf('Phase 2 - Limited Angle - %s',names{im}));

    for j=1:numel(limitedAngles)
        ang=limitedAngles(j);

        if ang==180
            idx=theta<180;
        else
            idx=theta<ang;
        end

        th=theta(idx);
        ss=sinograms{im}(:,idx);
        fs=filterSinogram(ss,'Ram-Lak');
        r=normalizeImage(myBackprojection(fs,th,N,s));

        limitedMSE(im,j)=imageMSE(f,r);
        limitedPSNR(im,j)=imagePSNR(f,r);

        subplot(1,3,j);
        imagesc(r); axis image off; colormap gray;
        title(sprintf('%d degrees\nMSE %.5g',ang,limitedMSE(im,j)));
    end
    sgtitle(sprintf('Limited-Angle Reconstruction - %s',names{im}));

    figure('Name',sprintf('Phase 2 - MSE vs Angle - %s',names{im}));
    plot(limitedAngles,limitedMSE(im,:),'o-','LineWidth',1.5);
    grid on;
    xlabel('Angular coverage (degrees)');
    ylabel('MSE');
    title(sprintf('Directional Information Loss - %s',names{im}));
end

%% 7. REFERENCE COMPARISON FOR THE THREE REAL CT SLICES
% Ground-truth slices are the full-reference images available in the
% supplied six-file set. If separate reference reconstruction files
% exist later, they can be substituted here without changing FBP.
referenceMSE = filterMSE(2:4,:);
referencePSNR = filterPSNR(2:4,:);

figure('Name','Phase 2 - Ground Truth Reference Comparison');
for i=1:3
    subplot(1,3,i);
    bar(filterMSE(i+1,:));
    xticks(1:4); xticklabels(filters);
    ylabel('MSE'); title(sprintf('GT %03d',i-1));
    grid on;
end
sgtitle('FBP vs Ground-Truth Reference Images');

fprintf('\n===== PHASE 2 SUMMARY =====\n');
fprintf('Filter comparison, noise-resolution trade-off, 6-view study,\n');
fprintf('limited-angle study, and 3-slice reference comparison completed.\n');

%% 8. SAVE EVERYTHING
save('Project3_Phase2_Final.mat',...
    'N','D','theta','views','limitedAngles','filters',...
    'images','names','sinograms','allRecon',...
    'filterMSE','filterPSNR',...
    'resolutionPhantom','resolutionSino','noisyResolutionSino',...
    'noiseStd','resolutionMSE','resolutionPSNR',...
    'underMSE','underPSNR','underMAE',...
    'limitedMSE','limitedPSNR',...
    'referenceMSE','referencePSNR','-v7.3');

fprintf('Saved: Project3_Phase2_Final.mat\n');
fprintf('===== PHASE 2 COMPLETED =====\n');

%% ====================== LOCAL FUNCTIONS ======================

function out=normalizeImage(in)
out=double(in); out=out-min(out(:)); m=max(out(:));
if m>0, out=out/m; end
end

function v=imageMSE(a,b)
a=normalizeImage(a); b=normalizeImage(b);
assert(isequal(size(a),size(b)),'MSE size mismatch.');
v=mean((a(:)-b(:)).^2);
end

function v=imagePSNR(a,b)
m=imageMSE(a,b);
if m<=eps, v=Inf; else, v=10*log10(1/m); end
end

function [sino,s]=myRadon(img,theta,D)
img=double(img); [N,M]=size(img);
sMax=0.5*sqrt(N^2+M^2);
s=linspace(-sMax,sMax,D);
sino=zeros(D,numel(theta));
for k=1:numel(theta)
    rotated=imrotate(img,-theta(k),'bilinear','loose');
    p=sum(rotated,1);
    x=linspace(-size(rotated,2)/2,size(rotated,2)/2,numel(p));
    sino(:,k)=interp1(x,p,s,'linear',0).';
end
sino=sino/max(N,1);
end

function recon=myBackprojection(sino,theta,N,s)
recon=zeros(N,N); x=((1:N)-(N+1)/2); [X,Y]=meshgrid(x,x);
if numel(theta)>1, dtheta=mean(diff(theta))*pi/180; else, dtheta=pi; end
for k=1:numel(theta)
    q=X*cosd(theta(k))+Y*sind(theta(k));
    v=interp1(s,sino(:,k),q,'linear',0);
    recon=recon+v*dtheta;
end
recon=recon/pi;
end

function filtered=filterSinogram(sino,name)
[D,A]=size(sino);
n=2^nextpow2(2*D);
freq=(-n/2:n/2-1)'/n;
ramp=abs(freq);

switch lower(name)
    case 'ram-lak'
        w=ones(size(freq));
    case 'shepp-logan'
        w=ones(size(freq)); z=abs(freq)>eps;
        w(z)=sin(pi*freq(z))./(pi*freq(z));
    case 'cosine'
        w=cos(pi*freq/2); w(abs(freq)>1)=0;
    case 'hamming'
        w=0.54+0.46*cos(pi*freq); w(abs(freq)>1)=0;
    otherwise
        error('Unknown filter %s',name);
end

H=ramp.*max(w,0);
filtered=zeros(size(sino));
st=floor((n-D)/2)+1;
for k=1:A
    P=fftshift(fft(sino(:,k),n));
    q=real(ifft(ifftshift(P.*H)));
    filtered(:,k)=q(st:st+D-1);
end
end

function p=makeResolutionPhantom(N)
% Controlled educational resolution phantom:
% a uniform circular background plus line-pair groups.
p=zeros(N,N);
[X,Y]=meshgrid(1:N,1:N);
cx=(N+1)/2; cy=cx;
p(((X-cx).^2+(Y-cy).^2)<=140^2)=0.35;

% Four bar groups with increasing spatial frequency.
groups=[-95 -60 3; -35 -60 4; 25 -60 5; 85 -60 7];
for g=1:size(groups,1)
    x0=groups(g,1); y0=groups(g,2); w=groups(g,3);
    for b=0:3
        x1=cx+x0+b*2*w;
        p((abs(X-x1)<=w/2) & (abs(Y-(cy+y0))<=18))=1;
    end
end

% Horizontal group.
for b=0:3
    y1=cy+15+b*10;
    p((abs(Y-y1)<=2) & (X>cx-45 & X<cx+45))=1;
end
p=normalizeImage(p);
end
