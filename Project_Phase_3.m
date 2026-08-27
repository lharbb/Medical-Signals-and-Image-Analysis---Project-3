%% ============================================================
% PROJECT 3 - PHASE 3
% IMAGE DEGRADATION AND RESTORATION
%% ============================================================

clear; clc; close all;

%% LOAD PHASE 2 RESULTS

if ~isfile('Phase2_Results.mat')
    error('Run Project_Phase_2.m first.');
end

load('Phase2_Results.mat','allResults');

fprintf('\n=============================================\n');
fprintf('PROJECT 3 - PHASE 3\n');
fprintf('DEGRADATION AND RESTORATION\n');
fprintf('=============================================\n');

%% PARAMETERS

motionLength = 21;
motionAngle = 20;

outFocusRadius = 9;

gaussianSigma = 0.03;

radialCutoffs = [0.02 0.05 0.10];

NSRvalues = [0.0001 0.001 0.01 0.1];

gammaValues = [0.0001 0.001 0.01 0.1];

%% PROCESS THREE RECONSTRUCTED SLICES

for im = 1:3

    fprintf('\n=============================================\n');
    fprintf('IMAGE %d\n',im);
    fprintf('=============================================\n');

    f = allResults{im}.recon{1};

    f = mat2gray(f);

    %% PSFs

    hMotion = motionPSF(motionLength,motionAngle);

    hDefocus = defocusPSF(outFocusRadius);

    %% ========================================================
    % DEGRADATION MODEL
    % g = h*f + n
    %% ========================================================

    blurMotion = conv2(f,hMotion,'same');

    blurDefocus = conv2(f,hDefocus,'same');

    %% GAUSSIAN NOISE

    rng(100+im);

    nGaussian = gaussianSigma*randn(size(f));

    gGaussian = blurMotion + nGaussian;

    gGaussian = min(max(gGaussian,0),1);

    %% POISSON NOISE

    scale = 255;

    gPoisson = imnoise(blurMotion,'poisson');

    %% COMBINED DEGRADATION

    motionGaussian = gGaussian;

    defocusGaussian = ...
        min(max(blurDefocus + ...
        gaussianSigma*randn(size(f)),0),1);

    motionPoisson = imnoise(blurMotion,'poisson');

    defocusPoisson = imnoise(blurDefocus,'poisson');

    %% DISPLAY DEGRADATION

    figure;

    subplot(3,4,1);
    imagesc(f);
    axis image off;
    colormap gray;
    title('Original');

    subplot(3,4,2);
    imagesc(blurMotion);
    axis image off;
    title('Motion Blur');

    subplot(3,4,3);
    imagesc(blurDefocus);
    axis image off;
    title('Out-of-Focus');

    subplot(3,4,4);
    imagesc(gGaussian);
    axis image off;
    title('Motion + Gaussian');

    subplot(3,4,5);
    imagesc(defocusGaussian);
    axis image off;
    title('Defocus + Gaussian');

    subplot(3,4,6);
    imagesc(gPoisson);
    axis image off;
    title('Motion + Poisson');

    subplot(3,4,7);
    imagesc(defocusPoisson);
    axis image off;
    title('Defocus + Poisson');

    subplot(3,4,8);
    imagesc(motionPoisson);
    axis image off;
    title('Poisson');

    sgtitle(sprintf('Image %d - Degradation',im));

    %% ========================================================
    % INVERSE FILTERING
    %% ========================================================

    fprintf('\nInverse filtering...\n');

    invMotion = inverseFilter(gGaussian,hMotion,0.001);

    invMotion = mat2gray(invMotion);

    figure;

    subplot(1,3,1);
    imagesc(f);
    axis image off;
    title('Original');

    subplot(1,3,2);
    imagesc(gGaussian);
    axis image off;
    title('Degraded');

    subplot(1,3,3);
    imagesc(invMotion);
    axis image off;
    title('Inverse Filter');

    sgtitle(sprintf('Image %d - Inverse Filtering Failure',im));

    inverseMSE = mean((f(:)-invMotion(:)).^2);

    fprintf('Inverse filtering MSE = %.6f\n',inverseMSE);

    fprintf(['Failure occurs because inverse filtering divides ', ...
        'by small values of H, amplifying noise.\n']);

    %% ========================================================
    % RADIAL LIMITED INVERSE FILTER
    %% ========================================================

    radialMSE = zeros(size(radialCutoffs));

    figure;

    for k = 1:length(radialCutoffs)

        r = radialInverse( ...
            gGaussian,hMotion,radialCutoffs(k));

        r = mat2gray(r);

        radialMSE(k) = mean((f(:)-r(:)).^2);

        subplot(1,3,k);

        imagesc(r);
        axis image off;

        title(sprintf('Cutoff %.2f', ...
            radialCutoffs(k)));

    end

    sgtitle(sprintf('Image %d - Radially Limited Inverse',im));

    %% ========================================================
    % WIENER FILTER
    %% ========================================================

    wienerMSE = zeros(size(NSRvalues));

    figure;

    for k = 1:length(NSRvalues)

        r = wienerRestore( ...
            gGaussian,hMotion,NSRvalues(k));

        r = mat2gray(r);

        wienerMSE(k) = mean((f(:)-r(:)).^2);

        subplot(2,2,k);

        imagesc(r);
        axis image off;

        title(sprintf('NSR = %.4f', ...
            NSRvalues(k)));

    end

    sgtitle(sprintf('Image %d - Wiener Filtering',im));

    %% ========================================================
    % CLS FILTER
    %% ========================================================

    clsMSE = zeros(size(gammaValues));

    figure;

    for k = 1:length(gammaValues)

        r = clsRestore( ...
            gGaussian,hMotion,gammaValues(k));

        r = mat2gray(r);

        clsMSE(k) = mean((f(:)-r(:)).^2);

        subplot(2,2,k);

        imagesc(r);
        axis image off;

        title(sprintf('\\gamma = %.4f', ...
            gammaValues(k)));

    end

    sgtitle(sprintf('Image %d - CLS Filtering',im));

    %% ========================================================
    % NOISE ESTIMATION FROM IMAGE
    %% ========================================================

    estimatedNoise = estimateNoise(gGaussian);

    estimatedNSR = estimatedNoise^2 / ...
        (var(gGaussian(:))+eps);

    estimatedGamma = estimatedNoise^2;

    estimatedWiener = wienerRestore( ...
        gGaussian,hMotion,estimatedNSR);

    estimatedCLS = clsRestore( ...
        gGaussian,hMotion,estimatedGamma);

    estimatedWiener = mat2gray(estimatedWiener);
    estimatedCLS = mat2gray(estimatedCLS);

    estimatedWienerMSE = ...
        mean((f(:)-estimatedWiener(:)).^2);

    estimatedCLSMSE = ...
        mean((f(:)-estimatedCLS(:)).^2);

    %% BEST VALUES

    [bestWienerMSE,bw] = min(wienerMSE);

    [bestCLSMSE,bc] = min(clsMSE);

    %% ACCURACY LOSS

    wienerLoss = ...
        100*(estimatedWienerMSE-bestWienerMSE) / ...
        max(bestWienerMSE,eps);

    clsLoss = ...
        100*(estimatedCLSMSE-bestCLSMSE) / ...
        max(bestCLSMSE,eps);

    %% DISPLAY ESTIMATED RESULTS

    figure;

    subplot(1,3,1);
    imagesc(f);
    axis image off;
    title('Original');

    subplot(1,3,2);
    imagesc(estimatedWiener);
    axis image off;

    title(sprintf('Estimated Wiener\nMSE %.6f', ...
        estimatedWienerMSE));

    subplot(1,3,3);
    imagesc(estimatedCLS);
    axis image off;

    title(sprintf('Estimated CLS\nMSE %.6f', ...
        estimatedCLSMSE));

    sgtitle(sprintf('Image %d - Estimated Noise Parameters',im));

    %% PRINT RESULTS

    fprintf('\n---------------------------------------------\n');
    fprintf('PHASE 3 RESULTS - IMAGE %d\n',im);
    fprintf('---------------------------------------------\n');

    fprintf('Motion blur length = %d pixels\n', ...
        motionLength);

    fprintf('Out-of-focus radius = %d pixels\n', ...
        outFocusRadius);

    fprintf('\nInverse filtering MSE = %.6f\n', ...
        inverseMSE);

    fprintf('\nRadial inverse:\n');

    for k = 1:length(radialCutoffs)

        fprintf('Cutoff %.2f -> MSE %.6f\n', ...
            radialCutoffs(k),radialMSE(k));

    end

    fprintf('\nWiener filtering:\n');

    for k = 1:length(NSRvalues)

        fprintf('NSR %.4f -> MSE %.6f\n', ...
            NSRvalues(k),wienerMSE(k));

    end

    fprintf('Best Wiener NSR = %.4f\n', ...
        NSRvalues(bw));

    fprintf('Best Wiener MSE = %.6f\n', ...
        bestWienerMSE);

    fprintf('\nCLS filtering:\n');

    for k = 1:length(gammaValues)

        fprintf('Gamma %.4f -> MSE %.6f\n', ...
            gammaValues(k),clsMSE(k));

    end

    fprintf('Best CLS gamma = %.4f\n', ...
        gammaValues(bc));

    fprintf('Best CLS MSE = %.6f\n', ...
        bestCLSMSE);

    fprintf('\nEstimated noise STD = %.6f\n', ...
        estimatedNoise);

    fprintf('Estimated Wiener NSR = %.6f\n', ...
        estimatedNSR);

    fprintf('Estimated Wiener MSE = %.6f\n', ...
        estimatedWienerMSE);

    fprintf('Wiener accuracy loss = %.2f %%\n', ...
        wienerLoss);

    fprintf('Estimated CLS MSE = %.6f\n', ...
        estimatedCLSMSE);

    fprintf('CLS accuracy loss = %.2f %%\n', ...
        clsLoss);

end

fprintf('\n=============================================\n');
fprintf('PHASE 3 COMPLETED\n');
fprintf('=============================================\n');


%% ============================================================
% MOTION BLUR PSF
%% ============================================================

function h = motionPSF(L,angle)

    h = zeros(L,L);

    c = (L+1)/2;

    for x = 1:L

        y = round(c + ...
            (x-c)*tand(angle));

        if y >= 1 && y <= L
            h(y,x) = 1;
        end

    end

    h = h/sum(h(:));

end


%% ============================================================
% OUT-OF-FOCUS PSF
%% ============================================================

function h = defocusPSF(radius)

    n = 2*radius+1;

    [X,Y] = meshgrid( ...
        -radius:radius, ...
        -radius:radius);

    h = double(X.^2+Y.^2 <= radius^2);

    h = h/sum(h(:));

end


%% ============================================================
% INVERSE FILTER
%% ============================================================

function out = inverseFilter(g,h,epsilon)

    [M,N] = size(g);

    H = psf2otf(h,[M N]);

    G = fft2(g);

    F = G ./ (H + epsilon);

    out = real(ifft2(F));

    out = min(max(out,0),1);

end


%% ============================================================
% RADIAL LIMITED INVERSE FILTER
%% ============================================================

function out = radialInverse(g,h,cutoff)

    [M,N] = size(g);

    H = psf2otf(h,[M N]);

    G = fft2(g);

    [X,Y] = meshgrid( ...
        (-floor(N/2):ceil(N/2)-1)/N, ...
        (-floor(M/2):ceil(M/2)-1)/M);

    R = sqrt(X.^2+Y.^2);

    mask = R <= cutoff;

    Hs = fftshift(H);

    Fs = zeros(size(H));

    Fs(mask) = fftshift(G(mask))./ ...
        (Hs(mask)+1e-3);

    F = ifftshift(Fs);

    out = real(ifft2(F));

    out = min(max(out,0),1);

end


%% ============================================================
% WIENER FILTER
%% ============================================================

function out = wienerRestore(g,h,NSR)

    [M,N] = size(g);

    H = psf2otf(h,[M N]);

    G = fft2(g);

    W = conj(H) ./ ...
        (abs(H).^2 + NSR);

    F = W.*G;

    out = real(ifft2(F));

    out = min(max(out,0),1);

end


%% ============================================================
% CONSTRAINED LEAST SQUARES
%% ============================================================

function out = clsRestore(g,h,gamma)

    [M,N] = size(g);

    H = psf2otf(h,[M N]);

    G = fft2(g);

    p = [0 -1 0;
        -1 4 -1;
         0 -1 0];

    P = psf2otf(p,[M N]);

    F = conj(H).*G ./ ...
        (abs(H).^2 + gamma*abs(P).^2);

    out = real(ifft2(F));

    out = min(max(out,0),1);

end


%% ============================================================
% NOISE ESTIMATION FROM IMAGE
%% ============================================================

function sigma = estimateNoise(img)

    kernel = [1 -2 1;
              -2 4 -2;
               1 -2 1];

    residual = conv2(img,kernel,'same');

    sigma = median(abs(residual(:))) / 0.6745;

end
