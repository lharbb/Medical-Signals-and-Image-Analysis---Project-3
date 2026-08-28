% PROJECT 3 - PHASE 3
% Degradation and Restoration
%
% REQUIREMENTS COVERED:
% 1) Explicit degradation model g = h*f + n
% 2) Linear motion blur
% 3) Out-of-focus blur
% 4) Additive Gaussian noise
% 5) Poisson noise
% 6) Direct inverse filtering and explicit failure demonstration
% 7) Radially limited inverse filtering
% 8) Wiener MMSE filtering with swept noise-to-signal ratio
% 9) Constrained least-squares filtering with swept regularization
% 10) Estimate noise parameters from the images themselves
% 11) Quantify accuracy loss from estimated instead of known parameters
%
% Run Phase 2 first so Project3_Phase2_Final.mat exists.
% No deconvwnr(), deconvreg(), or other restoration shortcut is used.

clear; clc; close all;
rng(3);

%% SETTINGS
motionLength = 15;
motionAngle = 20;
defocusRadius = 7;
gaussianSigma = 0.03;

radialRadii = [20 35 50 65 80 100 125];
wienerNSR = [1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2];
clsLambda = [1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2];

fprintf('\n===== PROJECT 3 - PHASE 3 =====\n');

%% 1. LOAD PHASE 2 RECONSTRUCTIONS
if ~isfile('Project3_Phase2_Final.mat')
    error('Project3_Phase2_Final.mat not found. Run Project_Phase_2.m first.');
end

load('Project3_Phase2_Final.mat','images','names','allRecon',...
    'filterMSE','filters');

%% 2. PROCESS THE THREE REAL GROUND-TRUTH RECONSTRUCTIONS
results = cell(3,1);

for i=1:3

    % Select the FBP reconstruction with the lowest full-reference MSE.
    [~,bestK] = min(filterMSE(i+1,:));
    f = normalizeImage(allRecon{i+1,bestK});
    bestFilter = filters{bestK};

    fprintf('\n--------------------------------------------\n');
    fprintf('Ground-truth slice %d | FBP input = %s\n',i-1,bestFilter);
    fprintf('--------------------------------------------\n');

    %% 3. DEGRADATION FUNCTIONS
    hMotion = makeMotionPSF(motionLength,motionAngle);
    hDefocus = makeDiskPSF(defocusRadius);

    %% 4. EXPLICIT MODEL g = h*f+n
    blurredMotion = conv2(f,hMotion,'same');
    blurredDefocus = conv2(f,hDefocus,'same');

    % Additive Gaussian noise.
    nGaussianMotion = gaussianSigma*randn(size(f));
    gMotionGaussian = blurredMotion + nGaussianMotion;

    nGaussianDefocus = gaussianSigma*randn(size(f));
    gDefocusGaussian = blurredDefocus + nGaussianDefocus;

    % Poisson observations. n = g - h*f makes the additive form explicit.
    poissonScale = 4096;
    gMotionPoisson = poissonObservation(blurredMotion,poissonScale);
    gDefocusPoisson = poissonObservation(blurredDefocus,poissonScale);

    nPoissonMotion = gMotionPoisson - blurredMotion;
    nPoissonDefocus = gDefocusPoisson - blurredDefocus;

    figure('Name',sprintf('Phase 3 - Degradation Slice %d',i-1));
    subplot(2,4,1); imagesc(f); axis image off; colormap gray; title('FBP reconstructed f');
    subplot(2,4,2); imagesc(blurredMotion); axis image off; colormap gray; title('Motion blur h*f');
    subplot(2,4,3); imagesc(blurredDefocus); axis image off; colormap gray; title('Out-of-focus h*f');
    subplot(2,4,4); imagesc(normalizeImage(gMotionGaussian)); axis image off; colormap gray;
    title('Motion + Gaussian');
    subplot(2,4,5); imagesc(normalizeImage(gDefocusGaussian)); axis image off; colormap gray;
    title('Defocus + Gaussian');
    subplot(2,4,6); imagesc(normalizeImage(gMotionPoisson)); axis image off; colormap gray;
    title('Motion + Poisson');
    subplot(2,4,7); imagesc(normalizeImage(gDefocusPoisson)); axis image off; colormap gray;
    title('Defocus + Poisson');
    subplot(2,4,8); imagesc(nGaussianMotion); axis image off; colormap gray;
    title('Example additive n');
    sgtitle(sprintf('Explicit Degradation Model: g = h*f + n | Slice %d',i-1));

    %% 5. DIRECT INVERSE FILTER: SHOW FAILURE
    % Main failure test uses motion blur + additive Gaussian noise.
    g = gMotionGaussian;
    h = hMotion;

    H = psf2otfCustom(h,size(g));
    Hshift = fftshift(H);

    % Direct inverse: G/H. A tiny threshold is used only to avoid Inf.
    invRaw = inverseFilter(g,h,0);
    invDirect = normalizeImage(invRaw);
    inverseMSE = imageMSE(f,invDirect);

    smallH = abs(Hshift) < 1e-3;
    inverseGain = mean(1./max(abs(Hshift(smallH)),eps));

    figure('Name',sprintf('Phase 3 - Inverse Failure Slice %d',i-1));
    subplot(1,4,1); imagesc(f); axis image off; colormap gray; title('Original f');
    subplot(1,4,2); imagesc(normalizeImage(g)); axis image off; colormap gray;
    title('Degraded g');
    subplot(1,4,3); imagesc(log(1+abs(Hshift))); axis image off; colormap gray;
    title('|H| spectrum');
    subplot(1,4,4); imagesc(invDirect); axis image off; colormap gray;
    title(sprintf('Direct inverse\nMSE %.5g',inverseMSE));
    sgtitle('Why Direct Inverse Filtering Fails');

    fprintf('Direct inverse MSE = %.6g\n',inverseMSE);
    fprintf('Small-|H| samples (<1e-3): %d\n',nnz(smallH));
    fprintf('Mean inverse gain in small-|H| region: %.4g\n',inverseGain);
    fprintf(['Interpretation: F_hat=(G/H)=(F+N/H). When |H| is small, ',...
        'the N/H term becomes very large, so noise dominates.\n']);

    %% 6. RADIALLY LIMITED INVERSE FILTERING
    radialMSE=zeros(size(radialRadii));
    radialPSNR=zeros(size(radialRadii));
    radialRecon=cell(size(radialRadii));

    for k=1:numel(radialRadii)
        r=normalizeImage(inverseFilter(g,h,radialRadii(k)));
        radialRecon{k}=r;
        radialMSE(k)=imageMSE(f,r);
        radialPSNR(k)=imagePSNR(f,r);
    end

    [bestRadialMSE,kr]=min(radialMSE);
    bestRadialRadius=radialRadii(kr);
    bestRadial=radialRecon{kr};

    figure('Name',sprintf('Phase 3 - Radial Inverse Slice %d',i-1));
    for k=1:numel(radialRadii)
        subplot(2,4,k);
        imagesc(radialRecon{k}); axis image off; colormap gray;
        title(sprintf('R=%d | MSE %.4g',radialRadii(k),radialMSE(k)));
    end
    sgtitle(sprintf('Radially Limited Inverse - Slice %d',i-1));

    figure('Name',sprintf('Phase 3 - Radial Sweep Slice %d',i-1));
    plot(radialRadii,radialMSE,'o-','LineWidth',1.5); grid on;
    xlabel('Radial cutoff'); ylabel('MSE');
    title(sprintf('Radially Limited Inverse Sweep - Slice %d',i-1));

    %% 7. WIENER MMSE SWEEP
    signalVar=max(var(blurredMotion(:)),eps);
    knownGaussianNSR=(gaussianSigma^2)/signalVar;

    wienerMSE=zeros(size(wienerNSR));
    wienerPSNR=zeros(size(wienerNSR));
    wienerRecon=cell(size(wienerNSR));

    for k=1:numel(wienerNSR)
        r=normalizeImage(wienerFilterFromScratch(g,h,wienerNSR(k)));
        wienerRecon{k}=r;
        wienerMSE(k)=imageMSE(f,r);
        wienerPSNR(k)=imagePSNR(f,r);
    end

    [bestWienerMSE,kw]=min(wienerMSE);
    bestWienerNSR=wienerNSR(kw);
    bestWiener=wienerRecon{kw};

    figure('Name',sprintf('Phase 3 - Wiener Sweep Slice %d',i-1));
    semilogx(wienerNSR,wienerMSE,'o-','LineWidth',1.5); grid on;
    xlabel('Noise-to-signal ratio'); ylabel('MSE');
    title(sprintf('Wiener NSR Sweep - Slice %d',i-1));

    %% 8. CONSTRAINED LEAST-SQUARES SWEEP
    clsMSE=zeros(size(clsLambda));
    clsPSNR=zeros(size(clsLambda));
    clsRecon=cell(size(clsLambda));

    for k=1:numel(clsLambda)
        r=normalizeImage(clsFilterFromScratch(g,h,clsLambda(k)));
        clsRecon{k}=r;
        clsMSE(k)=imageMSE(f,r);
        clsPSNR(k)=imagePSNR(f,r);
    end

    [bestCLSMSE,kc]=min(clsMSE);
    bestCLSLambda=clsLambda(kc);
    bestCLS=clsRecon{kc};

    figure('Name',sprintf('Phase 3 - CLS Sweep Slice %d',i-1));
    semilogx(clsLambda,clsMSE,'o-','LineWidth',1.5); grid on;
    xlabel('Regularization parameter \lambda'); ylabel('MSE');
    title(sprintf('CLS Regularization Sweep - Slice %d',i-1));

    %% 9. KNOWN VS IMAGE-ESTIMATED GAUSSIAN NOISE
    sigmaEstimatedG=estimateGaussianSigma(g);
    estimatedGaussianNSR=(sigmaEstimatedG^2)/signalVar;

    restoredKnownG=normalizeImage(wienerFilterFromScratch(g,h,knownGaussianNSR));
    restoredEstimatedG=normalizeImage(wienerFilterFromScratch(g,h,estimatedGaussianNSR));

    knownGMSE=imageMSE(f,restoredKnownG);
    estimatedGMSE=imageMSE(f,restoredEstimatedG);

    gaussianAccuracyLoss=100*(estimatedGMSE-knownGMSE)/max(knownGMSE,eps);

    %% 10. KNOWN VS IMAGE-ESTIMATED POISSON NOISE
    signalVarP=max(var(blurredMotion(:)),eps);

    knownPoissonVar=var(nPoissonMotion(:));
    knownPoissonNSR=knownPoissonVar/signalVarP;

    estimatedPoissonVar=estimatePoissonVariance(gMotionPoisson);
    estimatedPoissonNSR=estimatedPoissonVar/signalVarP;

    restoredKnownP=normalizeImage(wienerFilterFromScratch(...
        gMotionPoisson,h,knownPoissonNSR));
    restoredEstimatedP=normalizeImage(wienerFilterFromScratch(...
        gMotionPoisson,h,estimatedPoissonNSR));

    knownPMSE=imageMSE(f,restoredKnownP);
    estimatedPMSE=imageMSE(f,restoredEstimatedP);

    poissonAccuracyLoss=100*(estimatedPMSE-knownPMSE)/max(knownPMSE,eps);

    %% 11. BEST-METHOD COMPARISON
    figure('Name',sprintf('Phase 3 - Restoration Comparison Slice %d',i-1));
    subplot(2,3,1); imagesc(f); axis image off; colormap gray; title('Reference f');
    subplot(2,3,2); imagesc(normalizeImage(g)); axis image off; colormap gray;
    title('Degraded g');
    subplot(2,3,3); imagesc(invDirect); axis image off; colormap gray;
    title(sprintf('Direct inverse\nMSE %.4g',inverseMSE));
    subplot(2,3,4); imagesc(bestRadial); axis image off; colormap gray;
    title(sprintf('Radial inverse R=%d\nMSE %.4g',bestRadialRadius,bestRadialMSE));
    subplot(2,3,5); imagesc(bestWiener); axis image off; colormap gray;
    title(sprintf('Wiener NSR %.2e\nMSE %.4g',bestWienerNSR,bestWienerMSE));
    subplot(2,3,6); imagesc(bestCLS); axis image off; colormap gray;
    title(sprintf('CLS lambda %.2e\nMSE %.4g',bestCLSLambda,bestCLSMSE));
    sgtitle(sprintf('Restoration Comparison - Slice %d',i-1));

    %% 12. REPORT NUMERICAL RESULTS
    fprintf('\nSlice %d results:\n',i-1);
    fprintf('  Best FBP filter           : %s\n',bestFilter);
    fprintf('  Direct inverse MSE        : %.6g\n',inverseMSE);
    fprintf('  Best radial inverse MSE   : %.6g (R=%d)\n',bestRadialMSE,bestRadialRadius);
    fprintf('  Best Wiener MSE           : %.6g (NSR=%.3e)\n',bestWienerMSE,bestWienerNSR);
    fprintf('  Best CLS MSE              : %.6g (lambda=%.3e)\n',bestCLSMSE,bestCLSLambda);
    fprintf('  Gaussian sigma known      : %.6g\n',gaussianSigma);
    fprintf('  Gaussian sigma estimated  : %.6g\n',sigmaEstimatedG);
    fprintf('  Gaussian known-param MSE  : %.6g\n',knownGMSE);
    fprintf('  Gaussian estimated MSE    : %.6g\n',estimatedGMSE);
    fprintf('  Gaussian accuracy loss    : %.3f %%\n',gaussianAccuracyLoss);
    fprintf('  Poisson NSR known         : %.6g\n',knownPoissonNSR);
    fprintf('  Poisson NSR estimated     : %.6g\n',estimatedPoissonNSR);
    fprintf('  Poisson known-param MSE   : %.6g\n',knownPMSE);
    fprintf('  Poisson estimated MSE     : %.6g\n',estimatedPMSE);
    fprintf('  Poisson accuracy loss     : %.3f %%\n',poissonAccuracyLoss);

    results{i}=struct(...
        'slice',i-1,...
        'bestFBPFilter',bestFilter,...
        'f',f,...
        'hMotion',hMotion,...
        'hDefocus',hDefocus,...
        'motionBlur',blurredMotion,...
        'defocusBlur',blurredDefocus,...
        'motionGaussian',gMotionGaussian,...
        'defocusGaussian',gDefocusGaussian,...
        'motionPoisson',gMotionPoisson,...
        'defocusPoisson',gDefocusPoisson,...
        'inverseMSE',inverseMSE,...
        'radialRadii',radialRadii,...
        'radialMSE',radialMSE,...
        'radialPSNR',radialPSNR,...
        'bestRadialRadius',bestRadialRadius,...
        'bestRadialMSE',bestRadialMSE,...
        'wienerNSR',wienerNSR,...
        'wienerMSE',wienerMSE,...
        'wienerPSNR',wienerPSNR,...
        'bestWienerNSR',bestWienerNSR,...
        'bestWienerMSE',bestWienerMSE,...
        'clsLambda',clsLambda,...
        'clsMSE',clsMSE,...
        'clsPSNR',clsPSNR,...
        'bestCLSLambda',bestCLSLambda,...
        'bestCLSMSE',bestCLSMSE,...
        'knownGaussianSigma',gaussianSigma,...
        'estimatedGaussianSigma',sigmaEstimatedG,...
        'knownGaussianNSR',knownGaussianNSR,...
        'estimatedGaussianNSR',estimatedGaussianNSR,...
        'knownGaussianMSE',knownGMSE,...
        'estimatedGaussianMSE',estimatedGMSE,...
        'gaussianAccuracyLossPercent',gaussianAccuracyLoss,...
        'knownPoissonVariance',knownPoissonVar,...
        'estimatedPoissonVariance',estimatedPoissonVar,...
        'knownPoissonNSR',knownPoissonNSR,...
        'estimatedPoissonNSR',estimatedPoissonNSR,...
        'knownPoissonMSE',knownPMSE,...
        'estimatedPoissonMSE',estimatedPMSE,...
        'poissonAccuracyLossPercent',poissonAccuracyLoss);
end

%% 13. SAVE
save('Project3_Phase3_Final.mat','results','motionLength','motionAngle',...
    'defocusRadius','gaussianSigma','radialRadii','wienerNSR','clsLambda','-v7.3');

fprintf('\n===== PHASE 3 COMPLETED =====\n');
fprintf('Saved: Project3_Phase3_Final.mat\n');

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

function h=makeMotionPSF(len,angle)
len=max(3,round(len));
if mod(len,2)==0, len=len+1; end
x=(-(len-1)/2:(len-1)/2);
[X,Y]=meshgrid(x,x);
xr=X*cosd(angle)+Y*sind(angle);
yr=-X*sind(angle)+Y*cosd(angle);
h=double(abs(xr)<=len/2 & abs(yr)<=0.5);
h=h/max(sum(h(:)),eps);
end

function h=makeDiskPSF(radius)
radius=max(1,round(radius));
[X,Y]=meshgrid(-radius:radius,-radius:radius);
h=double(X.^2+Y.^2<=radius^2);
h=h/max(sum(h(:)),eps);
end

function g=poissonObservation(f,scale)
f=normalizeImage(f);
lambda=max(f,0)*scale;
if exist('poissrnd','file')==2
    counts=poissrnd(lambda);
    g=counts/scale;
else
    % MATLAB fallback: Poisson noise generated by Image Processing Toolbox.
    g=imnoise(f,'poisson');
end
end

function out=inverseFilter(g,h,radius)
[M,N]=size(g);
H=psf2otfCustom(h,[M N]);
G=fft2(g);
if radius<=0
    mask=ones(M,N);
else
    [U,V]=meshgrid((-floor(N/2)):(ceil(N/2)-1),...
                   (-floor(M/2)):(ceil(M/2)-1));
    mask=sqrt(U.^2+V.^2)<=radius;
end
Hs=fftshift(H);
invH=zeros(size(Hs));
stable=abs(Hs)>1e-8;
invH(stable)=1./Hs(stable);
Fshift=fftshift(G).*invH.*mask;
out=real(ifft2(ifftshift(Fshift)));
end

function out=wienerFilterFromScratch(g,h,NSR)
H=psf2otfCustom(h,size(g));
G=fft2(g);
W=conj(H)./(abs(H).^2+max(NSR,eps));
out=real(ifft2(G.*W));
end

function out=clsFilterFromScratch(g,h,lambda)
[M,N]=size(g);
H=psf2otfCustom(h,[M N]);
G=fft2(g);

% Laplacian constraint P.
P=[0 1 0;1 -4 1;0 1 0];
P=psf2otfCustom(P,[M N]);

den=abs(H).^2+lambda*abs(P).^2;
out=real(ifft2((conj(H).*G)./max(den,eps)));
end

function H=psf2otfCustom(psf,outSize)
psf=double(psf);
H=zeros(outSize);
[m,n]=size(psf);
H(1:m,1:n)=psf;
H=circshift(H,-floor([m n]/2));
H=fft2(H);
end

function sigma=estimateGaussianSigma(img)
% Estimate additive Gaussian noise from the image itself using a
% high-pass residual and robust MAD.
localMean=conv2(img,ones(3)/9,'same');
residual=img-localMean;
sigma=median(abs(residual(:)-median(residual(:))))/0.6745;
if ~isfinite(sigma) || sigma<=0
    sigma=std(residual(:));
end
end

function variance=estimatePoissonVariance(img)
% Estimate Poisson noise variance from the image itself using a
% high-pass residual. The known synthetic Poisson parameter is NOT used.
localMean=conv2(img,ones(3)/9,'same');
residual=img-localMean;
variance=var(residual(:));
if ~isfinite(variance) || variance<=0
    variance=eps;
end
end
