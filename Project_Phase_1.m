%% PROJECT 3 - PHASE 1
% Forward Projection, Radon Transform and Unfiltered Backprojection
%
% REQUIREMENTS COVERED:
% 1) Modified Shepp-Logan phantom
% 2) At least three LoDoPaB ground-truth CT slices
% 3) Verify the forward model against supplied sinograms
% 4) Point source, centered disc, off-centre object
% 5) Sinusoidal point-source trace
% 6) Unfiltered backprojection
% 7) Analytical + empirical 1/r blur demonstration
%
% FROM SCRATCH:
% myRadon and myBackprojection are implemented below.
% MATLAB radon() and iradon() are NOT used.
%
% DATA EXPECTED IN THE CURRENT MATLAB DRIVE FOLDER:
% ground_truth_test_000.hdf5
% ground_truth_test_001.hdf5
% ground_truth_test_002.hdf5
% observation_test_000.hdf5
% observation_test_001.hdf5
% observation_test_002.hdf5

clear; clc; close all;
rng(1);

N = 362;
D = 513;
numViews = 1000;
theta = linspace(0,179,numViews);

gtFiles = {'ground_truth_test_000.hdf5',...
           'ground_truth_test_001.hdf5',...
           'ground_truth_test_002.hdf5'};
obsFiles = {'observation_test_000.hdf5',...
            'observation_test_001.hdf5',...
            'observation_test_002.hdf5'};

fprintf('\n===== PROJECT 3 - PHASE 1 =====\n');
fprintf('Image size: %d x %d | Detectors: %d | Views: %d\n',N,N,D,numViews);

%% 1. LOAD THE REQUIRED THREE GROUND-TRUTH SLICES
GT = cell(3,1);
OBS = cell(3,1);

for i = 1:3
    if ~isfile(gtFiles{i})
        error('Missing %s. Put the three ground-truth HDF5 files in the current folder.',gtFiles{i});
    end
    if ~isfile(obsFiles{i})
        error('Missing %s. Put the three observation HDF5 files in the current folder.',obsFiles{i});
    end

    GT{i} = readGroundTruth(gtFiles{i},N);
    OBS{i} = readObservation(obsFiles{i},D,numViews);

    fprintf('Loaded GT %d: %s\n',i,gtFiles{i});
    fprintf('Loaded observation %d: %s\n',i,obsFiles{i});
end

%% 2. SHEPP-LOGAN PHANTOM: FORWARD MODEL
phantomImg = phantom('Modified Shepp-Logan',N);
phantomImg = normalizeImage(phantomImg);

[sinoPhantom,s] = myRadon(phantomImg,theta,D);
bpPhantom = normalizeImage(myBackprojection(sinoPhantom,theta,N,s));

figure('Name','Phase 1 - Phantom');
subplot(1,3,1); imagesc(phantomImg); axis image off; colormap gray; colorbar;
title('Modified Shepp-Logan Phantom');
subplot(1,3,2); imagesc(theta,s,sinoPhantom); axis xy; colormap gray; colorbar;
xlabel('\theta (degrees)'); ylabel('Detector position');
title('Forward Projection / Sinogram');
subplot(1,3,3); imagesc(bpPhantom); axis image off; colormap gray; colorbar;
title(sprintf('Unfiltered Backprojection\nMSE = %.5g',imageMSE(phantomImg,bpPhantom)));

%% 3. POINT SOURCE, DISC AND OFF-CENTRE OBJECT
pointX = 70; pointY = -40;
pointImg = createPoint(N,pointX,pointY);
centerDisc = createDisc(N,0,0,45);
offDisc = createDisc(N,pointX,pointY,45);

[pointSino,~] = myRadon(pointImg,theta,D);
[centerSino,~] = myRadon(centerDisc,theta,D);
[offSino,~] = myRadon(offDisc,theta,D);

% Theoretical point trace in the same coordinate system as s.
x0 = pointX; y0 = pointY;
theoreticalTrace = x0*cosd(theta) + y0*sind(theta);

figure('Name','Phase 1 - Sinogram Signatures');
subplot(2,4,1); imagesc(pointImg); axis image off; colormap gray; title('Point source');
subplot(2,4,2); imagesc(theta,s,pointSino); axis xy; colormap gray;
title('Point sinogram');
subplot(2,4,3); imagesc(theta,s,pointSino); axis xy; colormap gray; hold on;
plot(theta,theoreticalTrace,'r','LineWidth',1.5); hold off;
title('Point + theoretical trace');
subplot(2,4,4); imagesc(theta,s,centerSino); axis xy; colormap gray;
title('Centered disc sinogram');
subplot(2,4,5); imagesc(centerDisc); axis image off; colormap gray; title('Centered disc');
subplot(2,4,6); imagesc(offDisc); axis image off; colormap gray; title('Off-centre disc');
subplot(2,4,7); imagesc(theta,s,offSino); axis xy; colormap gray;
title('Off-centre disc sinogram');
subplot(2,4,8); plot(theta,theoreticalTrace,'LineWidth',1.5); grid on;
xlabel('\theta'); ylabel('Detector coordinate');
title('s(\theta)=x_0 cos\theta+y_0 sin\theta');

%% 4. EMPIRICAL 1/r BLUR
% A point-source projection is backprojected without filtering.
% Its radial profile is compared with a 1/r reference.
centerPoint = createPoint(N,0,0);
[centerPointSino,~] = myRadon(centerPoint,theta,D);
pointBP = normalizeImage(myBackprojection(centerPointSino,theta,N,s));

[radius,profile] = radialProfile(pointBP);
valid = radius >= 3 & radius <= floor(N/3) & profile > 0;
p = polyfit(log(radius(valid)),log(profile(valid)),1);
empiricalSlope = p(1);

rRef = radius(valid);
ref = 1./rRef;
ref = ref/max(ref);
prof = profile(valid);
prof = prof/max(prof);

figure('Name','Phase 1 - 1 over r');
loglog(rRef,prof,'LineWidth',1.5); hold on;
loglog(rRef,ref,'--','LineWidth',1.5); grid on;
xlabel('Radius r'); ylabel('Normalized intensity');
legend('Empirical point-source BP','1/r reference','Location','southwest');
title(sprintf('Unfiltered Backprojection: 1/r Blur (fitted slope = %.3f)',empiricalSlope));

fprintf('\n1/r empirical log-log slope = %.4f (ideal value = -1)\n',empiricalSlope);

figure('Name','Phase 1 - Point Source BP');
subplot(1,3,1); imagesc(centerPoint); axis image off; colormap gray; title('Point source');
subplot(1,3,2); imagesc(pointBP); axis image off; colormap gray;
title('Unfiltered backprojection');
subplot(1,3,3); plot(radius,profile,'LineWidth',1.5); grid on;
xlabel('Radius'); ylabel('Normalized profile'); title('Radial profile');

%% 5. THREE LoDoPaB GROUND-TRUTH SLICES
computedSino = cell(3,1);
unfilteredBP = cell(3,1);
bpMSE = zeros(3,1);
forwardMSE = zeros(3,1);
forwardCorr = zeros(3,1);

for i = 1:3
    fprintf('\nProcessing ground-truth slice %d...\n',i);

    [computedSino{i},s] = myRadon(GT{i},theta,D);
    unfilteredBP{i} = normalizeImage(myBackprojection(computedSino{i},theta,N,s));
    bpMSE(i) = imageMSE(GT{i},unfilteredBP{i});

    % The supplied observation is a low-dose/noisy projection measurement,
    % so comparison is reported after normalization and correlation.
    forwardMSE(i) = imageMSE(computedSino{i},OBS{i});
    forwardCorr(i) = normalizedCorrelation(computedSino{i},OBS{i});

    figure('Name',sprintf('Phase 1 - Ground Truth %d',i));
    subplot(2,2,1); imagesc(GT{i}); axis image off; colormap gray; colorbar;
    title(sprintf('Ground Truth Slice %d',i));
    subplot(2,2,2); imagesc(theta,s,computedSino{i}); axis xy; colormap gray; colorbar;
    xlabel('\theta'); ylabel('Detector'); title('Computed sinogram');
    subplot(2,2,3); imagesc(unfilteredBP{i}); axis image off; colormap gray; colorbar;
    title(sprintf('Unfiltered BP | MSE %.5g',bpMSE(i)));
    subplot(2,2,4); imagesc(theta,s,OBS{i}); axis xy; colormap gray; colorbar;
    title(sprintf('Supplied observation\nMSE %.5g, corr %.4f',forwardMSE(i),forwardCorr(i)));

    fprintf('  BP MSE = %.6g\n',bpMSE(i));
    fprintf('  Forward-model normalized MSE vs supplied observation = %.6g\n',forwardMSE(i));
    fprintf('  Forward-model correlation = %.5f\n',forwardCorr(i));
end

%% 6. SAVE PHASE 1 RESULTS FOR PHASE 2
save('Project3_Phase1_Final.mat','phantomImg','sinoPhantom','s','theta',...
    'GT','OBS','computedSino','unfilteredBP','bpMSE','forwardMSE',...
    'forwardCorr','pointImg','pointSino','centerDisc','centerSino',...
    'offDisc','offSino','empiricalSlope','-v7.3');

fprintf('\n===== PHASE 1 COMPLETED =====\n');
fprintf('Saved: Project3_Phase1_Final.mat\n');

%% ====================== LOCAL FUNCTIONS ======================

function out = normalizeImage(in)
out = double(in);
out = out - min(out(:));
m = max(out(:));
if m > 0, out = out/m; end
end

function v = imageMSE(a,b)
a=normalizeImage(a); b=normalizeImage(b);
assert(isequal(size(a),size(b)),'MSE size mismatch.');
v=mean((a(:)-b(:)).^2);
end

function [sino,s] = myRadon(img,theta,D)
% Discrete parallel-beam Radon transform from scratch.
img=double(img);
[N,M]=size(img);
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
recon=zeros(N,N);
x=((1:N)-(N+1)/2);
[X,Y]=meshgrid(x,x);
if numel(theta)>1, dtheta=mean(diff(theta))*pi/180; else, dtheta=pi; end

for k=1:numel(theta)
    q=X*cosd(theta(k))+Y*sind(theta(k));
    v=interp1(s,sino(:,k),q,'linear',0);
    recon=recon+v*dtheta;
end
recon=recon/pi;
end

function p=createPoint(N,x0,y0)
p=zeros(N,N);
cx=round((N+1)/2+x0);
cy=round((N+1)/2+y0);
if cx>=1 && cx<=N && cy>=1 && cy<=N
    p(cy,cx)=1;
end
end

function d=createDisc(N,x0,y0,radius)
[X,Y]=meshgrid(1:N,1:N);
cx=(N+1)/2+x0; cy=(N+1)/2+y0;
d=double((X-cx).^2+(Y-cy).^2<=radius^2);
end

function [r,profile]=radialProfile(img)
img=normalizeImage(img);
[N,M]=size(img);
cx=(M+1)/2; cy=(N+1)/2;
[X,Y]=meshgrid(1:M,1:N);
R=round(sqrt((X-cx).^2+(Y-cy).^2));
maxR=floor(min(N,M)/2);
r=(1:maxR).';
profile=zeros(maxR,1);
for k=1:maxR
    v=img(R==k);
    if ~isempty(v), profile(k)=mean(v); end
end
profile=profile/max(profile+eps);
end

function path=findDatasetPath(info,prefix,targetRows,targetCols)
path='';
for k=1:numel(info.Datasets)
    if isempty(prefix), p=['/' info.Datasets(k).Name];
    else, p=[prefix '/' info.Datasets(k).Name]; end
    dims=double(info.Datasets(k).Dataspace.Size);
    if numel(dims)>=2 && any(dims==targetRows) && any(dims==targetCols)
        path=p; return;
    end
end
for k=1:numel(info.Groups)
    if isempty(prefix), gp=['/' info.Groups(k).Name];
    else, gp=[prefix '/' info.Groups(k).Name]; end
    path=findDatasetPath(info.Groups(k),gp,targetRows,targetCols);
    if ~isempty(path), return; end
end
end

function data=readH5FirstMatching(filename,targetRows,targetCols)
info=h5info(filename);
path=findDatasetPath(info,'',targetRows,targetCols);
if isempty(path)
    error('No HDF5 dataset with dimensions containing %d x %d found in %s.',...
        targetRows,targetCols,filename);
end
di=h5info(filename,path);
dims=double(di.Dataspace.Size);
start=ones(1,numel(dims)); count=ones(1,numel(dims));
r=find(dims==targetRows,1,'first');
c=find(dims==targetCols,1,'last');
count(r)=targetRows; count(c)=targetCols;
raw=h5read(filename,path,start,count);
data=squeeze(double(raw));
if ~isequal(size(data),[targetRows targetCols])
    data=reshape(data,targetRows,targetCols);
end
end

function img=readGroundTruth(filename,N)
img=normalizeImage(readH5FirstMatching(filename,N,N));
end

function sino=readObservation(filename,D,A)
sino=readH5FirstMatching(filename,D,A);
end

function c=normalizedCorrelation(a,b)
a=double(a(:)); b=double(b(:));
a=a-mean(a); b=b-mean(b);
den=sqrt(sum(a.^2)*sum(b.^2));
if den<=eps, c=0; else, c=sum(a.*b)/den; end
end
