clear; clc; close all;

%% =========================================================
% PHASE 1 - FORWARD PROJECTION AND RADON TRANSFORM
% FULL PROJECT REQUIREMENTS - FROM SCRATCH
% =========================================================

N = 362;
numAngles = 1000;
numDetectors = 513;

% Number of samples along each ray
numSamples = 256;

theta = linspace(0,180,numAngles+1);
theta(end) = [];

x = linspace(-(N-1)/2,(N-1)/2,N);
[X,Y] = meshgrid(x,x);

rhoMax = sqrt(2)*(N-1)/2;
rho = linspace(-rhoMax,rhoMax,numDetectors);

fprintf('\n========================================\n');
fprintf('PHASE 1 STARTED\n');
fprintf('Image: %d x %d\n',N,N);
fprintf('Views: %d\n',numAngles);
fprintf('Detectors: %d\n',numDetectors);
fprintf('========================================\n');


%% =========================================================
% 1. SHEPP-LOGAN PHANTOM
% =========================================================

phantom = sheppLoganScratch(N);

figure;
imagesc(phantom);
axis image off;
colormap gray;
colorbar;
title('Shepp-Logan Phantom');


%% =========================================================
% 2. DISCRETE RADON TRANSFORM - SHEPP LOGAN
% =========================================================

fprintf('\nGenerating Shepp-Logan sinogram...\n');

tic;

S_phantom = radonScratchFast( ...
    phantom,theta,rho,numSamples);

t = toc;

fprintf('Completed in %.2f seconds.\n',t);
fprintf('Sinogram size = %d x %d\n', ...
    size(S_phantom,1),size(S_phantom,2));


figure;
imagesc(theta,rho,S_phantom);
axis xy;
axis tight;
colormap gray;
colorbar;
xlabel('Projection angle (degrees)');
ylabel('Detector position');
title('Shepp-Logan Sinogram');


%% =========================================================
% 3. POINT SOURCE SIGNATURE
% =========================================================

fprintf('\nPoint-source experiment...\n');

point = zeros(N,N);
point(round(N/2),round(N/2)) = 1;

S_point = radonScratchFast( ...
    point,theta,rho,numSamples);

figure;

subplot(1,2,1);
imagesc(point);
axis image off;
colormap gray;
title('Point Source');

subplot(1,2,2);
imagesc(theta,rho,S_point);
axis xy;
title('Point Source Sinogram');
xlabel('Angle');
ylabel('Detector');


%% =========================================================
% 4. DISC SIGNATURE
% =========================================================

fprintf('\nDisc experiment...\n');

disc = double(X.^2 + Y.^2 <= 50^2);

S_disc = radonScratchFast( ...
    disc,theta,rho,numSamples);

figure;

subplot(1,2,1);
imagesc(disc);
axis image off;
colormap gray;
title('Disc');

subplot(1,2,2);
imagesc(theta,rho,S_disc);
axis xy;
title('Disc Sinogram');
xlabel('Angle');
ylabel('Detector');


%% =========================================================
% 5. OFF-CENTRE OBJECT
% =========================================================

fprintf('\nOff-centre experiment...\n');

x0 = 70;
y0 = -50;
R = 40;

offCentre = double( ...
    (X-x0).^2 + (Y-y0).^2 <= R^2);

S_off = radonScratchFast( ...
    offCentre,theta,rho,numSamples);

figure;

subplot(1,2,1);
imagesc(offCentre);
axis image off;
colormap gray;
title('Off-Centre Disc');

subplot(1,2,2);
imagesc(theta,rho,S_off);
axis xy;
title('Off-Centre Sinogram');
xlabel('Angle');
ylabel('Detector');


%% =========================================================
% 6. SINUSOIDAL TRACE
% =========================================================

fprintf('\n========================================\n');
fprintf('SINUSOIDAL TRACE\n');
fprintf('========================================\n');

fprintf('\nFor a point at (x0,y0):\n');
fprintf('rho(theta) = x0*cos(theta) + y0*sin(theta)\n');

fprintf('\nTherefore an off-centre point produces a\n');
fprintf('sinusoidal trace in the sinogram.\n');

fprintf('\nFor a point at the centre:\n');
fprintf('x0 = 0, y0 = 0\n');
fprintf('rho(theta) = 0\n');
fprintf('Therefore the trace is vertical.\n');


%% =========================================================
% 7. LOAD THREE LODOPAB GROUND TRUTH SLICES
% =========================================================

fprintf('\n========================================\n');
fprintf('LODOPAB DATA\n');
fprintf('========================================\n');

files = dir('ground_truth_test_*.hdf5');

if numel(files) < 3
    error(['At least THREE ground_truth_test HDF5 ' ...
           'files are required.']);
end

[~,idx] = sort({files.name});
files = files(idx);

GT = cell(3,1);
S_GT = cell(3,1);


%% =========================================================
% 8. PROCESS THREE SLICES
% =========================================================

for k = 1:3

    fprintf('\n----------------------------------------\n');
    fprintf('LoDoPaB slice %d / 3\n',k);
    fprintf('File: %s\n',files(k).name);
    fprintf('----------------------------------------\n');

    data = h5read( ...
        fullfile(files(k).folder,files(k).name), ...
        '/data');

    I = getSlice(data,N);

    I = double(I);

    I = I - min(I(:));

    if max(I(:)) > 0
        I = I ./ max(I(:));
    end

    GT{k} = I;

    figure;
    imagesc(I);
    axis image off;
    colormap gray;
    colorbar;
    title(sprintf('LoDoPaB Ground Truth %d',k));


    fprintf('Generating scratch sinogram...\n');

    tic;

    S_GT{k} = radonScratchFast( ...
        I,theta,rho,numSamples);

    elapsed = toc;

    fprintf('Completed in %.2f seconds.\n',elapsed);

    fprintf('Sinogram size: %d x %d\n', ...
        size(S_GT{k},1),size(S_GT{k},2));


    figure;
    imagesc(theta,rho,S_GT{k});
    axis xy;
    axis tight;
    colormap gray;
    colorbar;
    xlabel('Projection angle');
    ylabel('Detector position');
    title(sprintf('LoDoPaB Scratch Sinogram %d',k));

end


%% =========================================================
% 9. THREE SINOGRAMS TOGETHER
% =========================================================

figure;

for k = 1:3

    subplot(1,3,k);

    imagesc(theta,rho,S_GT{k});

    axis xy;
    axis tight;

    colormap gray;

    xlabel('Angle');

    if k == 1
        ylabel('Detector');
    end

    title(sprintf('LoDoPaB %d',k));

end

sgtitle('Three LoDoPaB Scratch Sinograms');


%% =========================================================
% 10. VERIFY AGAINST SUPPLIED SINOGRAMS
% =========================================================

fprintf('\n========================================\n');
fprintf('SUPPLIED SINOGRAM VERIFICATION\n');
fprintf('========================================\n');

obs = dir('observation_test_*.hdf5');

if isempty(obs)

    fprintf('\nObservation files were not found.\n');
    fprintf('Verification will be completed after upload.\n');

else

    [~,idx] = sort({obs.name});
    obs = obs(idx);

    nCompare = min(3,numel(obs));

    for k = 1:nCompare

        fprintf('\nComparing sample %d...\n',k);

        data = h5read( ...
            fullfile(obs(k).folder,obs(k).name), ...
            '/data');

        supplied = getSinogram( ...
            data,numDetectors,numAngles);

        generated = S_GT{k};

        A = double(generated);
        B = double(supplied);

        A = (A-mean(A(:))) / std(A(:));
        B = (B-mean(B(:))) / std(B(:));

        correlation = corr(A(:),B(:));

        relativeRMSE = ...
            norm(A(:)-B(:)) / norm(B(:));

        fprintf('Correlation   = %.5f\n',correlation);
        fprintf('Relative RMSE = %.5f\n',relativeRMSE);


        figure;

        subplot(1,2,1);
        imagesc(generated);
        axis image;
        colormap gray;
        colorbar;
        title('Scratch Radon');


        subplot(1,2,2);
        imagesc(supplied);
        axis image;
        colorbar;
        title('Supplied Sinogram');

        sgtitle(sprintf( ...
            'Verification Sample %d',k));

    end
end


%% =========================================================
% 11. UNFILTERED BACKPROJECTION
% =========================================================

fprintf('\n========================================\n');
fprintf('UNFILTERED BACKPROJECTION\n');
fprintf('========================================\n');

tic;

BP = backprojectScratchFast( ...
    S_GT{1},theta,rho,N);

elapsed = toc;

fprintf('Backprojection completed in %.2f seconds.\n',elapsed);


BP = BP-min(BP(:));

if max(BP(:)) > 0
    BP = BP/max(BP(:));
end


figure;

subplot(1,2,1);
imagesc(GT{1});
axis image off;
colormap gray;
title('Ground Truth');


subplot(1,2,2);
imagesc(BP);
axis image off;
title('Unfiltered Backprojection');

sgtitle('Ground Truth vs Unfiltered Backprojection');


MSE = mean((GT{1}(:)-BP(:)).^2);

fprintf('\nUnfiltered Backprojection MSE = %.6f\n',MSE);


%% =========================================================
% 12. ANALYTICAL 1/r BLUR
% =========================================================

fprintf('\n========================================\n');
fprintf('1/r BLUR ANALYSIS\n');
fprintf('========================================\n');

fprintf('\nFor a point source:\n');
fprintf('rho(theta) = x0*cos(theta) + y0*sin(theta)\n');

fprintf('\nAt the origin:\n');
fprintf('rho(theta) = 0 for all theta.\n');

fprintf('\nBackprojection spreads the contribution along\n');
fprintf('all lines passing through the point.\n');

fprintf('\nThe resulting blur behaves approximately as:\n');
fprintf('B(r) proportional to 1/r\n');


%% =========================================================
% 13. EMPIRICAL 1/r DEMONSTRATION
% =========================================================

fprintf('\nEmpirical 1/r experiment...\n');

BP_point = backprojectScratchFast( ...
    S_point,theta,rho,N);

Rr = sqrt(X.^2 + Y.^2);

r = (5:floor(N/2)-5)';

profile = zeros(size(r));


for k = 1:length(r)

    mask = Rr >= r(k)-0.5 & ...
           Rr < r(k)+0.5;

    profile(k) = mean(BP_point(mask));

end


valid = profile > 0;

r = r(valid);
profile = profile(valid);

profile = profile/max(profile);

theory = 1./r;
theory = theory/max(theory);


p = polyfit(log(r),log(profile),1);

fprintf('\nMeasured log-log slope = %.3f\n',p(1));
fprintf('Ideal 1/r slope        = -1.000\n');


figure;

loglog(r,profile,'LineWidth',2);
hold on;

loglog(r,theory,'--','LineWidth',2);

grid on;

xlabel('Radius r');
ylabel('Normalized intensity');

legend('Measured','Ideal 1/r');

title('Empirical 1/r Blur');


%% =========================================================
% 14. FINAL PHASE 1 STATUS
% =========================================================

fprintf('\n========================================\n');
fprintf('PHASE 1 STATUS\n');
fprintf('========================================\n');

fprintf('Shepp-Logan phantom          : DONE\n');
fprintf('Discrete Radon transform     : DONE\n');
fprintf('Point source                 : DONE\n');
fprintf('Disc                         : DONE\n');
fprintf('Off-centre object            : DONE\n');
fprintf('Three LoDoPaB slices         : DONE\n');
fprintf('Unfiltered backprojection    : DONE\n');
fprintf('Analytical 1/r               : DONE\n');
fprintf('Empirical 1/r                : DONE\n');

if isempty(obs)
    fprintf('Supplied sinogram comparison : PENDING\n');
else
    fprintf('Supplied sinogram comparison : DONE\n');
end

fprintf('========================================\n');


%% =========================================================
% FUNCTION 1 - FAST DISCRETE RADON
% =========================================================

function S = radonScratchFast(I,theta,rho,numSamples)

    I = double(I);

    N = size(I,1);

    c = linspace(-(N-1)/2,(N-1)/2,N);

    [X,Y] = meshgrid(c,c);

    tMax = sqrt(2)*(N-1)/2;

    t = linspace(-tMax,tMax,numSamples);

    [R,T] = meshgrid(rho,t);

    S = zeros(length(rho),length(theta));

    for a = 1:length(theta)

        angle = theta(a);

        rayX = R*cosd(angle) - T*sind(angle);

        rayY = R*sind(angle) + T*cosd(angle);

        values = interp2( ...
            X,Y,I,rayX,rayY,'linear',0);

        S(:,a) = trapz(t,values,1)';

    end

end


%% =========================================================
% FUNCTION 2 - BACKPROJECTION
% =========================================================

function BP = backprojectScratchFast(S,theta,rho,N)

    c = linspace(-(N-1)/2,(N-1)/2,N);

    [X,Y] = meshgrid(c,c);

    BP = zeros(N,N);

    dtheta = abs(theta(2)-theta(1))*pi/180;

    for a = 1:length(theta)

        angle = theta(a);

        detectorCoordinate = ...
            X*cosd(angle) + ...
            Y*sind(angle);

        values = interp1( ...
            rho,S(:,a), ...
            detectorCoordinate, ...
            'linear',0);

        BP = BP + values*dtheta;

    end

    BP = BP/pi;

end


%% =========================================================
% FUNCTION 3 - SHEPP LOGAN PHANTOM
% =========================================================

function P = sheppLoganScratch(N)

    x = linspace(-1,1,N);

    [X,Y] = meshgrid(x,x);

    P = zeros(N,N);


    E = [
        1.0   .6900 .9200  0       0        0
       -.8   .6624 .8740  0      -.0184     0
       -.2   .1100 .3100  .2200   0       -18
       -.2   .1600 .4100 -.2200   0        18
        .1   .2100 .2500  0       .3500     0
        .1   .0460 .0460  0       .1000     0
        .1   .0460 .0460  0      -.1000     0
        .1   .0460 .0230 -.0800  -.6050     0
        .1   .0230 .0230  0      -.6060     0
        .1   .0230 .0460  .0600  -.6050     0
    ];


    for k = 1:size(E,1)

        A = E(k,1);
        a = E(k,2);
        b = E(k,3);
        x0 = E(k,4);
        y0 = E(k,5);
        phi = E(k,6);

        Xc = X-x0;
        Yc = Y-y0;

        Xr = Xc*cosd(phi)+Yc*sind(phi);

        Yr = -Xc*sind(phi)+Yc*cosd(phi);

        mask = ...
            (Xr/a).^2 + ...
            (Yr/b).^2 <= 1;

        P(mask) = P(mask)+A;

    end


    P = P-min(P(:));

    P = P/max(P(:));

end


%% =========================================================
% FUNCTION 4 - READ GROUND TRUTH
% =========================================================

function I = getSlice(data,N)

    s = size(data);

    if numel(s)==3 && ...
            s(1)==N && s(2)==N

        I = data(:,:,1);

    elseif numel(s)==3 && ...
            s(2)==N && s(3)==N

        I = squeeze(data(1,:,:));

    elseif numel(s)==2 && ...
            s(1)==N && s(2)==N

        I = data;

    else

        error('Unexpected ground-truth HDF5 dimensions.');

    end

end


%% =========================================================
% FUNCTION 5 - READ SUPPLIED SINOGRAM
% =========================================================

function S = getSinogram(data,nDet,nAngles)

    s = size(data);

    if numel(s)==2

        if s(1)==nDet && s(2)==nAngles

            S = data;

        elseif s(2)==nDet && s(1)==nAngles

            S = data';

        else

            error('Unexpected observation dimensions.');

        end

    elseif numel(s)==3

        if s(2)==nDet && s(3)==nAngles

            S = squeeze(data(1,:,:));

        elseif s(2)==nAngles && s(3)==nDet

            S = squeeze(data(1,:,:))';

        else

            error('Unexpected observation dimensions.');

        end

    else

        error('Unexpected observation dimensions.');

    end

    S = double(S);

end
