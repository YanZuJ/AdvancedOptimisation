%% max_ensemble_denoise_full.m
clear; close all; clc;

%% ---------------- Parameters ----------------
noisyFileName = 'noisyboat.png';
useColor = true;
lambda = 1e-4;
lambda_higher = 5e-5;
rho = 2;
maxIter = 3000;
tol = 1e-5;
huberDelta = 0.01;
showEvery = 25;

%% ---------------- Read image ----------------
Iin = im2double(imread(noisyFileName));
if useColor && size(Iin,3)==3
    I_ycbcr = rgb2ycbcr(Iin);
    Ychan = I_ycbcr(:,:,1);
    Cb = I_ycbcr(:,:,2);
    Cr = I_ycbcr(:,:,3);
    y = Ychan;
else
    if size(Iin,3)==3
        y = rgb2gray(Iin);
    else
        y = Iin;
    end
    Cb=[]; Cr=[];
end
[Ny, Nx] = size(y);

%% ---------------- Gradient operators ----------------
grad = @(x) deal(circshift(x,[0 -1])-x, circshift(x,[-1 0])-x);
grad2 = @(x) deal(circshift(x,[0 -2])-2*circshift(x,[0 -1])+x, ...
                   circshift(x,[-2 0])-2*circshift(x,[-1 0])+x);
divergence = @(gx,gy) gx - circshift(gx,[0 1]) + gy - circshift(gy,[1 0]);

%% ---------------- FFT denominators ----------------
[ux, uy] = meshgrid(0:(Nx-1),0:(Ny-1));
wx = 2*pi*ux/Nx; wy = 2*pi*uy/Ny;
eigDtD = (2-2*cos(wx)) + (2-2*cos(wy));
eigDtD2 = (4-4*cos(wx)) + (4-4*cos(wy));
denomFFT = 1 + rho*eigDtD;
denomFFT2 = 1 + rho*eigDtD2;

%% ---------------- ADMM Initialization ----------------
x = y;
[z1x,z1y] = deal(zeros(Ny,Nx)); % Aniso TV
[z2x,z2y] = deal(zeros(Ny,Nx)); % Iso TV
[z3x,z3y] = deal(zeros(Ny,Nx)); % Higher-order TV
[z4x,z4y] = deal(zeros(Ny,Nx)); % Huber TV

[u1x,u1y] = deal(zeros(Ny,Nx));
[u2x,u2y] = deal(zeros(Ny,Nx));
[u3x,u3y] = deal(zeros(Ny,Nx));
[u4x,u4y] = deal(zeros(Ny,Nx));

prevx = x;

%% ---------------- ADMM Iterations ----------------
for k=1:maxIter
    % ---------------- x-update ----------------
    rhs = y + rho*(divergence(z1x-u1x,z1y-u1y) + ...
                   divergence(z2x-u2x,z2y-u2y) + ...
                   divergence(z3x-u3x,z3y-u3y) + ...
                   divergence(z4x-u4x,z4y-u4y));
    % simple FFT solve approximation
    x = real(ifft2(fft2(rhs)./(1+4*rho)));

    % ---------------- Update auxiliary variables ----------------
    % 1) Anisotropic TV
    [dx1, dy1] = grad(x);
    thresh1 = lambda/rho;
    z1x = max(abs(dx1+u1x)-thresh1,0).*sign(dx1+u1x);
    z1y = max(abs(dy1+u1y)-thresh1,0).*sign(dy1+u1y);

    % 2) Isotropic TV
    [dx2, dy2] = grad(x);
    v1 = dx2+u2x; v2 = dy2+u2y;
    mag = sqrt(v1.^2 + v2.^2);
    scale = max(zeros(size(mag)), 1 - lambda./(rho*(mag+1e-12))); % FIX: element-wise
    z2x = scale.*v1; z2y = scale.*v2;

    % 3) Higher-order TV
    [dx3, dy3] = grad2(x);
    thresh3 = lambda_higher/(2*rho);
    z3x = max(abs(dx3+u3x)-thresh3,0).*sign(dx3+u3x);
    z3y = max(abs(dy3+u3y)-thresh3,0).*sign(dy3+u3y);

    % 4) Huber TV
    [dx4, dy4] = grad(x);
    v1 = dx4+u4x; v2 = dy4+u4y;
    mag = sqrt(v1.^2 + v2.^2);
    scale = ones(size(mag));
    mask = mag > huberDelta;
    scale(mask) = max(0, 1 - lambda./(rho*mag(mask))); % FIX: element-wise
    z4x = scale.*v1; z4y = scale.*v2;

    % ---------------- Dual updates ----------------
    u1x = u1x + (dx1-z1x); u1y = u1y + (dy1-z1y);
    u2x = u2x + (dx2-z2x); u2y = u2y + (dy2-z2y);
    u3x = u3x + (dx3-z3x); u3y = u3y + (dy3-z3y);
    u4x = u4x + (dx4-z4x); u4y = u4y + (dy4-z4y);

    % ---------------- Check convergence ----------------
    relchg = norm(x(:)-prevx(:))/max(1e-8,norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || k==maxIter || relchg<tol
        fprintf('Iter %4d: relchg=%.3e\n',k,relchg);
    end
    if relchg<tol, break; end
    prevx = x;
end

%% ---------------- Ensemble result ----------------
x_maxreg = max(cat(3, z1x+z1y, z2x+z2y, z3x+z3y, z4x+z4y), [], 3);

%% ---------------- Color conversion ----------------
if useColor && ~isempty(Cb) && ~isempty(Cr)
    if ~isequal(size(Cb), [Ny Nx])
        Cb = imresize(Cb,[Ny Nx]);
        Cr = imresize(Cr,[Ny Nx]);
    end
    output_rgb = ycbcr2rgb(cat(3, x_maxreg, Cb, Cr));
else
    output_rgb = x_maxreg;
end

%% ---------------- Display ----------------
figure;
imshow(output_rgb); title('Ensemble Max-Regularization Output');

%% ---------------- Save ----------------
imwrite(output_rgb,'ensemble_maxreg.png');
fprintf('Saved ensemble max-regularization image.\n');
