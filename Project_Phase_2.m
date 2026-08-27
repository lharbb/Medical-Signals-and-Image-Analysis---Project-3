%% ============================================================
% PROJECT 3 - PHASE 2
% FILTERED BACKPROJECTION FROM SCRATCH
% 3 LoDoPaB GROUND-TRUTH SLICES
%% ============================================================

clear; clc; close all;

N = 362;
D = 513;

views = [1000 500 250 100 50 25];
angles = [180 120 90];

filters = {'Ram-Lak','Shepp-Logan','Cosine','Hamming'};

files = {
    'ground_truth_test_000.hdf5'
    'ground_truth_test_001.hdf5'
    'ground_truth_test_002.hdf5'
};

%% LOAD 3 IMAGES
imgs = cell(3,1);

for i = 1:3
    imgs{i} = loadH5(files{i},N);
end

%% PROCESS THREE IMAGES

allResults = cell(3,1);

for im = 1:3

    f = imgs{im};

    fprintf('\n====================================\n');
    fprintf('IMAGE %d\n',im);
    fprintf('====================================\n');

    %% 1000 VIEW SINOGRAM

    theta = linspace(0,179,1000);

    [sino,s] = myRadon(f,theta,D);

    figure;
    imagesc(theta,s,sino);
    axis xy image;
    colormap gray;
    colorbar;
    xlabel('\theta (degrees)');
    ylabel('Detector');
    title(sprintf('Image %d - Sinogram',im));

    %% FOUR FILTERS

    recon = cell(4,1);
    filterMSE = zeros(4,1);

    for k = 1:4

        fs = filterSinogram(sino,filters{k});

        r = myBackprojection(fs,theta,N,s);

        r = mat2gray(r);

        recon{k} = r;

        filterMSE(k) = mean((f(:)-r(:)).^2);

    end

    figure;

    for k = 1:4

        subplot(2,2,k);
        imagesc(recon{k});
        axis image off;
        colormap gray;
        colorbar;

        title(sprintf('%s\nMSE = %.6f', ...
            filters{k},filterMSE(k)));

    end

    sgtitle(sprintf('Image %d - Four Filters',im));

    %% ANGULAR UNDERSAMPLING

    underRecon = cell(length(views),1);
    underMSE = zeros(length(views),1);

    figure;

    for k = 1:length(views)

        idx = round(linspace(1,1000,views(k)));

        th = theta(idx);
        ss = sino(:,idx);

        fs = filterSinogram(ss,'Ram-Lak');

        r = myBackprojection(fs,th,N,s);
        r = mat2gray(r);

        underRecon{k} = r;

        underMSE(k) = mean((f(:)-r(:)).^2);

        subplot(2,3,k);
        imagesc(r);
        axis image off;
        colormap gray;

        title(sprintf('%d Views',views(k)));

    end

    sgtitle(sprintf('Image %d - Angular Undersampling',im));

    %% LIMITED ANGLE

    limitedRecon = cell(3,1);
    limitedMSE = zeros(3,1);

    figure;

    for k = 1:3

        th = linspace(0,angles(k),180);

        [sl,sd] = myRadon(f,th,D);

        fs = filterSinogram(sl,'Ram-Lak');

        r = myBackprojection(fs,th,N,sd);

        r = mat2gray(r);

        limitedRecon{k} = r;

        limitedMSE(k) = mean((f(:)-r(:)).^2);

        subplot(1,3,k);

        imagesc(r);
        axis image off;
        colormap gray;

        title(sprintf('%d Degrees',angles(k)));

    end

    sgtitle(sprintf('Image %d - Limited Angle',im));

    %% PRINT RESULTS

    fprintf('\nFILTER COMPARISON\n');

    for k = 1:4
        fprintf('%s: MSE = %.6f\n', ...
            filters{k},filterMSE(k));
    end

    fprintf('\nANGULAR UNDERSAMPLING\n');

    for k = 1:length(views)
        fprintf('%d views: MSE = %.6f\n', ...
            views(k),underMSE(k));
    end

    fprintf('\nLIMITED ANGLE\n');

    for k = 1:3
        fprintf('%d degrees: MSE = %.6f\n', ...
            angles(k),limitedMSE(k));
    end

    %% SAVE RESULTS FOR PHASE 3

    allResults{im}.original = f;
    allResults{im}.sinogram = sino;
    allResults{im}.theta = theta;
    allResults{im}.s = s;

    allResults{im}.filters = filters;
    allResults{im}.recon = recon;
    allResults{im}.filterMSE = filterMSE;

    allResults{im}.views = views;
    allResults{im}.underRecon = underRecon;
    allResults{im}.underMSE = underMSE;

    allResults{im}.limitedAngles = angles;
    allResults{im}.limitedRecon = limitedRecon;
    allResults{im}.limitedMSE = limitedMSE;

end

save('Phase2_Results.mat','allResults','-v7.3');

fprintf('\n====================================\n');
fprintf('PHASE 2 COMPLETED\n');
fprintf('Results saved in Phase2_Results.mat\n');
fprintf('====================================\n');


%% ============================================================
% HDF5 LOADER
%% ============================================================

function img = loadH5(filename,N)

    info = h5info(filename);

    dataset = findDataset(info);

    fprintf('Loading: %s\n',filename);
    fprintf('Dataset: %s\n',dataset);

    img = h5read(filename,dataset);

    img = double(img);

    while ndims(img) > 2
        img = img(:,:,1);
    end

    img = imresize(img,[N N]);

    img = mat2gray(img);

end


function name = findDataset(info)

    if ~isempty(info.Datasets)

        name = [info.Name '/' info.Datasets(1).Name];

        if startsWith(name,'//')
            name = name(2:end);
        end

        return;
    end

    for k = 1:length(info.Groups)

        try
            name = findDataset(info.Groups(k));
            return;
        catch
        end

    end

    error('No numeric dataset found in HDF5 file.');

end


%% ============================================================
% RADON FROM SCRATCH
%% ============================================================

function [sino,s] = myRadon(img,theta,D)

    [N,M] = size(img);

    x = ((1:M)-(M+1)/2)/M;
    y = ((1:N)-(N+1)/2)/N;

    [X,Y] = meshgrid(x,y);

    sMax = sqrt(0.5^2+0.5^2);

    s = linspace(-sMax,sMax,D);

    ds = s(2)-s(1);

    sino = zeros(D,length(theta));

    for k = 1:length(theta)

        q = X*cosd(theta(k)) + Y*sind(theta(k));

        for i = 1:D

            mask = abs(q-s(i)) <= ds/2;

            if any(mask(:))
                sino(i,k) = sum(img(mask));
            end

        end

    end

    sino = sino/N;

end


%% ============================================================
% FILTER
%% ============================================================

function out = filterSinogram(sino,name)

    [D,A] = size(sino);

    n = 2^nextpow2(2*D);

    freq = (-n/2:n/2-1)'/n;

    ramp = abs(freq);

    switch lower(name)

        case 'ram-lak'
            w = ones(size(freq));

        case 'shepp-logan'

            w = ones(size(freq));
            z = freq ~= 0;

            w(z) = sin(pi*freq(z))./(pi*freq(z));

        case 'cosine'

            w = cos(pi*freq/2);

        case 'hamming'

            w = 0.54 + 0.46*cos(pi*freq);

        otherwise

            error('Unknown filter.');

    end

    w = max(w,0);

    H = ramp.*w;

    out = zeros(size(sino));

    st = floor((n-D)/2)+1;

    for k = 1:A

        P = fftshift(fft(sino(:,k),n));

        Q = P.*H;

        q = real(ifft(ifftshift(Q)));

        out(:,k) = q(st:st+D-1);

    end

end


%% ============================================================
% BACKPROJECTION FROM SCRATCH
%% ============================================================

function recon = myBackprojection(sino,theta,N,s)

    recon = zeros(N,N);

    x = ((1:N)-(N+1)/2)/N;

    [X,Y] = meshgrid(x,x);

    dtheta = abs(theta(2)-theta(1))*pi/180;

    for k = 1:length(theta)

        q = X*cosd(theta(k)) + Y*sind(theta(k));

        v = interp1(s,sino(:,k),q,'linear',0);

        recon = recon + v*dtheta;

    end

    recon = recon/pi;

end
