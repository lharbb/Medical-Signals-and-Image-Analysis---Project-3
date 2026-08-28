%% PROJECT 3 - BONUS STRETCH OBJECTIVE
% SIRT iterative reconstruction vs FBP
% Required comparison: 50 views and 25 views
%
% IMPORTANT:
% - This bonus uses SIRT (Simultaneous Iterative Reconstruction Technique).
% - No radon(), iradon(), deconvwnr(), or deconvreg() are used.
% - The forward/backprojection operators are implemented here.
% - The script uses the same 362 x 362 phantom and 513 detector samples
%   used in Phase 2.
%
% Run this file after your Phase 2 file, OR run it independently.

clear; clc; close all;

%% PARAMETERS
N = 362;
numDetectors = 513;
allViews = [1000 500 250 100 50 25];

% SIRT parameters
numIterations = 15;
relaxation = 0.8;

%% CREATE TEST IMAGE
f = phantom('Modified Shepp-Logan',N);
f = mat2gray(double(f));

%% CREATE 1000-view PROJECTIONS
fprintf('Creating 1000-view projections...\n');

theta1000 = linspace(0,179,1000);
[sino1000,s] = myRadon(f,theta1000,numDetectors);

fprintf('Projection data ready.\n');

%% ============================================================
% BONUS: COMPARE SIRT WITH FBP AT 50 AND 25 VIEWS
%% ============================================================

viewNumbers = [50 25];

FBP = cell(2,1);
SIRT = cell(2,1);
mseFBP = zeros(2,1);
mseSIRT = zeros(2,1);
psnrFBP = zeros(2,1);
psnrSIRT = zeros(2,1);

for a = 1:2

    nv = viewNumbers(a);

    fprintf('\n=============================================\n');
    fprintf('BONUS: %d VIEWS\n',nv);
    fprintf('=============================================\n');

    % Select equally spaced projections
    idx = round(linspace(1,1000,nv));
    idx = unique(idx);

    theta = theta1000(idx);
    sino = sino1000(:,idx);

    %% ---------------- FBP ----------------

    fprintf('Running FBP...\n');

    filtered = filterSinogram(sino,'Ram-Lak');

    rFBP = myBackprojection(filtered,theta,N,s);
    rFBP = mat2gray(rFBP);

    FBP{a} = rFBP;

    mseFBP(a) = mean((f(:)-rFBP(:)).^2);
    psnrFBP(a) = 10*log10(1/mseFBP(a));

    %% ---------------- SIRT ----------------

    fprintf('Running SIRT (%d iterations)...\n',numIterations);

    rSIRT = sirtReconstruction( ...
        sino,theta,N,numDetectors,s,...
        numIterations,relaxation);

    rSIRT = mat2gray(rSIRT);

    SIRT{a} = rSIRT;

    mseSIRT(a) = mean((f(:)-rSIRT(:)).^2);
    psnrSIRT(a) = 10*log10(1/mseSIRT(a));

    fprintf('\n%d views results:\n',nv);
    fprintf('FBP  : MSE = %.6f, PSNR = %.3f dB\n', ...
        mseFBP(a),psnrFBP(a));
    fprintf('SIRT : MSE = %.6f, PSNR = %.3f dB\n', ...
        mseSIRT(a),psnrSIRT(a));

    %% ---------------- IMAGE COMPARISON ----------------

    figure('Name',sprintf('SIRT vs FBP - %d Views',nv));

    subplot(1,3,1);
    imagesc(f);
    axis image off;
    colormap gray;
    title('Ground Truth');

    subplot(1,3,2);
    imagesc(rFBP);
    axis image off;
    title(sprintf('FBP - %d Views',nv));

    subplot(1,3,3);
    imagesc(rSIRT);
    axis image off;
    title(sprintf('SIRT - %d Views',nv));

    sgtitle(sprintf('Bonus Comparison - %d Views',nv));

end

%% ============================================================
% NUMERICAL COMPARISON
%% ============================================================

fprintf('\n=============================================\n');
fprintf('SIRT vs FBP - BONUS SUMMARY\n');
fprintf('=============================================\n');

fprintf('\n        FBP MSE       SIRT MSE      FBP PSNR      SIRT PSNR\n');

for a = 1:2

    fprintf('%2d     %.6f      %.6f      %.3f dB      %.3f dB\n', ...
        viewNumbers(a), ...
        mseFBP(a), ...
        mseSIRT(a), ...
        psnrFBP(a), ...
        psnrSIRT(a));

end

%% Improvement relative to FBP
improvementMSE = 100*(mseFBP-mseSIRT)./mseFBP;

fprintf('\nMSE improvement of SIRT relative to FBP:\n');

for a = 1:2

    fprintf('%d views: %.2f %%\n', ...
        viewNumbers(a),improvementMSE(a));

end

%% BAR CHART
figure('Name','Bonus Quantitative Comparison');

bar([mseFBP mseSIRT]);
grid on;

xlabel('Number of Views');
ylabel('MSE');
title('FBP vs SIRT: MSE at 50 and 25 Views');

set(gca,'XTickLabel',{'50 Views','25 Views'});
legend('FBP','SIRT','Location','best');

fprintf('\n=============================================\n');
fprintf('BONUS STRETCH OBJECTIVE COMPLETED\n');
fprintf('SIRT compared with FBP at 50 and 25 views.\n');
fprintf('=============================================\n');


%% ============================================================
% FUNCTION: SIRT
%% ============================================================

function recon = sirtReconstruction( ...
    sino,theta,N,numDetectors,s,numIterations,relaxation)

    % Initial image
    recon = zeros(N,N);

    % Detector coordinate spacing
    ds = s(2)-s(1);

    % Number of projections
    A = length(theta);

    % SIRT normalization.
    %
    % The forward model uses line sums. The correction is
    % backprojected and normalized by the number of views.
    %
    % A small positive value prevents division by zero.
    backNorm = zeros(N,N);

    for k = 1:A

        q = zeros(N,N);

        x = ((1:N)-(N+1)/2)/N;
        [X,Y] = meshgrid(x,x);

        detectorCoordinate = ...
            X*cosd(theta(k)) + Y*sind(theta(k));

        mask = abs(detectorCoordinate) <= max(abs(s));

        q(mask) = 1;

        backNorm = backNorm + q;

    end

    backNorm = backNorm/A;
    backNorm(backNorm < 1e-8) = 1;

    %% ITERATIONS

    for it = 1:numIterations

        correction = zeros(N,N);

        for k = 1:A

            %% Forward projection of current estimate

            projection = forwardProjection( ...
                recon,theta(k),numDetectors,s);

            %% Projection residual

            residual = sino(:,k) - projection;

            %% Backproject residual

            correction = correction + ...
                backprojectResidual( ...
                    residual,theta(k),N,s);

        end

        correction = correction/A;

        %% Normalized simultaneous update

        recon = recon + ...
            relaxation * correction ./ backNorm;

        %% Physical image constraint

        recon(recon < 0) = 0;

        if mod(it,5)==0 || it==1 || it==numIterations

            fprintf('Iteration %d / %d\n', ...
                it,numIterations);

        end

    end

end


%% ============================================================
% FUNCTION: FORWARD PROJECTION
% Implemented from scratch using bilinear interpolation.
%% ============================================================

function projection = forwardProjection( ...
    img,theta,numDetectors,s)

    N = size(img,1);

    x = ((1:N)-(N+1)/2)/N;
    y = x;

    [X,Y] = meshgrid(x,y);

    % Rotate coordinates into detector frame
    U = X*cosd(theta) + Y*sind(theta);
    V = -X*sind(theta) + Y*cosd(theta);

    % Sample image at rotated coordinates
    rotated = interp2( ...
        X,Y,img,U,V,'linear',0);

    % Sum along rays
    projection = zeros(numDetectors,1);

    ds = s(2)-s(1);

    for i = 1:numDetectors

        mask = abs(U-s(i)) <= ds/2;

        if any(mask(:))
            projection(i) = sum(rotated(mask))/N;
        end

    end

end


%% ============================================================
% FUNCTION: BACKPROJECT RESIDUAL
%% ============================================================

function correction = backprojectResidual( ...
    residual,theta,N,s)

    x = ((1:N)-(N+1)/2)/N;
    y = x;

    [X,Y] = meshgrid(x,y);

    detectorCoordinate = ...
        X*cosd(theta) + Y*sind(theta);

    values = interp1( ...
        s,residual,detectorCoordinate, ...
        'linear',0);

    correction = values;

end


%% ============================================================
% FUNCTION: DISCRETE RADON TRANSFORM
%% ============================================================

function [sino,s] = myRadon(img,theta,numDetectors)

    [N,M] = size(img);

    x = ((1:M)-(M+1)/2)/M;
    y = ((1:N)-(N+1)/2)/N;

    [X,Y] = meshgrid(x,y);

    sMax = sqrt(.5^2+.5^2);

    s = linspace(-sMax,sMax,numDetectors);
    ds = s(2)-s(1);

    sino = zeros(numDetectors,length(theta));

    for k = 1:length(theta)

        q = X*cosd(theta(k)) + ...
            Y*sind(theta(k));

        for i = 1:numDetectors

            mask = abs(q-s(i)) <= ds/2;

            if any(mask(:))
                sino(i,k) = sum(img(mask));
            end

        end

    end

    sino = sino/N;

end


%% ============================================================
% FUNCTION: FBP FILTER
%% ============================================================

function filtered = filterSinogram(sino,name)

    [D,A] = size(sino);

    n = 2^nextpow2(2*D);

    freq = (-n/2:n/2-1)'/n;

    ramp = abs(freq);

    switch lower(name)

        case 'ram-lak'
            w = ones(size(freq));

        case 'shepp-logan'
            w = ones(size(freq));
            z = freq~=0;
            w(z) = sin(pi*freq(z))./(pi*freq(z));

        case 'cosine'
            w = cos(pi*freq/2);

        case 'hamming'
            w = .54+.46*cos(pi*freq);

        otherwise
            error('Unknown filter.');

    end

    w = max(w,0);
    H = ramp.*w;

    filtered = zeros(size(sino));

    for k=1:A

        P = fftshift(fft(sino(:,k),n));

        q = real(ifft(ifftshift(P.*H)));

        st = floor((n-D)/2)+1;

        filtered(:,k) = q(st:st+D-1);

    end

end


%% ============================================================
% FUNCTION: FBP BACKPROJECTION
%% ============================================================

function recon = myBackprojection(sino,theta,N,s)

    recon = zeros(N,N);

    x = ((1:N)-(N+1)/2)/N;

    [X,Y] = meshgrid(x,x);

    dtheta = abs(theta(2)-theta(1))*pi/180;

    for k=1:length(theta)

        q = X*cosd(theta(k)) + ...
            Y*sind(theta(k));

        v = interp1( ...
            s,sino(:,k),q,'linear',0);

        recon = recon + v*dtheta;

    end

    recon = recon/pi;

end
