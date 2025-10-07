% tv_denoise_admm.m
% Denoise a noisy image using anisotropic Total Variation (TV) minimization
% solved by ADMM. Uses FFT-based solver for the x-update (periodic BC).
%
% Minimizes: 0.5 * ||x - y||_2^2 + lambda * ||D x||_1
% where D computes forward finite differences (horizontal and vertical).
%
% Usage: edit noisyFileName, lambda, rho, maxIter below and run.

clear; close all; clc;

%% Parameters (edit these)
noisyFileName = 'Lena256.png';  % your noisy image file
useColor = false;                 % true -> process luminance channel, false -> grayscale
lambda = 0.01;                   % TV regularization weight
rho = 5;                          % ADMM penalty parameter (tweakable)
maxIter = 10000;                    % ADMM iterations
tol = 1e-5;                       % stopping tolerance (relative change)
showEvery = 25;                   % show progress every this many iterations

%% Read image (force double in [0,1])
Iin = im2double(imread(noisyFileName));
if useColor && size(Iin,3) == 3
    % Convert to YCbCr and denoise luminance channel
    I_ycbcr = rgb2ycbcr(Iin);
    Ychan = I_ycbcr(:,:,1);
    Cb = I_ycbcr(:,:,2);
    Cr = I_ycbcr(:,:,3);
    mY = Ychan;
else
    % Convert to grayscale if RGB, otherwise keep
    if size(Iin,3) == 3
        mY = rgb2gray(Iin);
    else
        mY = Iin;
    end
end

[Ny, Nx] = size(mY);
n = Ny * Nx;
y = mY;                 % noisy image in 2D
yvec = y(:);            % vectorized (not used in ops below except PSNR calc)

%% Helper functions (forward differences with periodic boundary conditions)
grad = @(x) deal( ...
    circshift(x, [0 -1]) - x, ...    % Dx: differences along columns (horizontal)
    circshift(x, [-1 0]) - x ...     % Dy: differences along rows (vertical)
    );

divergence = @(gx, gy) ( ...
    gx - circshift(gx, [0 1]) + ...     % -D_x^T gx  (adjoint for periodic FD)
    gy - circshift(gy, [1 0]) ...       % -D_y^T gy
    );

%% Precompute FFT denominator for x-update
% Frequencies:
[ux, uy] = meshgrid(0:(Nx-1), 0:(Ny-1));
wx = 2*pi*ux / Nx;
wy = 2*pi*uy / Ny;
% eigenvalue of D^T D for forward differences under periodic BC:
eigDtD = (2 - 2*cos(wx)) + (2 - 2*cos(wy));   % = 4*sin^2(wx/2) + 4*sin^2(wy/2)
denomFFT = 1 + rho * eigDtD;

%% Initialize ADMM variables
x = y;                          % primal variable (2D)
[zx, zy] = deal(zeros(Ny, Nx)); % auxiliary variables for gradients
[ux_d, uy_d] = deal(zeros(Ny, Nx)); % scaled dual variables (u = dual)

prevx = x;

%% ADMM iterations
fprintf('ADMM TV denoising: lambda=%.4g, rho=%.4g, maxIter=%d\n', lambda, rho, maxIter);
for k = 1:maxIter
    % x-update: solve (I + rho D^T D) x = y + rho div(z - u)
    rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
    rhsFFT = fft2(rhs);
    x = real(ifft2(rhsFFT ./ denomFFT));   % periodic BC solution via FFT

    % compute gradients of x
    [dx, dy] = grad(x);

    % z-update: soft-thresholding (anisotropic TV -> shrink each component)
    v1 = dx + ux_d;
    v2 = dy + uy_d;
    % soft threshold elementwise (anisotropic)
    zx = max(abs(v1) - lambda / rho, 0) .* sign(v1);
    zy = max(abs(v2) - lambda / rho, 0) .* sign(v2);

    % dual update: u = u + (D x - z)
    ux_d = ux_d + (dx - zx);
    uy_d = uy_d + (dy - zy);

    % stopping criterion (relative change)
    relchg = norm(x(:) - prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k, showEvery) == 0 || k == 1 || k == maxIter || relchg < tol
        % compute objective (optional)
        tvTerm = sum(abs(dx(:))) + sum(abs(dy(:)));
        obj = 0.5 * sum((x(:) - yvec).^2) + lambda * tvTerm;
        fprintf('iter %3d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    prevx = x;

    if relchg < tol
        fprintf('Converged (relchg < tol) at iter %d\n', k);
        break;
    end
end

%% Postprocess and results
% Clip to [0,1]
x = min(max(x, 0), 1);

% PSNR helper
psnrval = @(ref, rec) 10*log10(1 ./ mean((ref(:)-rec(:)).^2));

% If we had original noisy image stored in Iin, compute PSNRs:
noisy_psnr = psnrval(mY, mY); % trivial (will be Inf/NaN if zero MSE) - skip
% If you have a clean reference image to compare (optional), set it above as 'mI'
% e.g. mI = im2double(imread('cleanLena.png'));
haveRef = false;  % set true if you also have a clean reference 'mI'
if haveRef
    % ensure mI is grayscale and same size
    % mI = im2double(imread('cleanLena.png'));
    ref = mI; % replace with actual variable
    fprintf('PSNR(noisy vs ref) = %.2f dB\n', psnrval(ref, mY));
    fprintf('PSNR(denoised vs ref) = %.2f dB\n', psnrval(ref, x));
end

%% Display results
figure('Name','TV Denoise ADMM','NumberTitle','off','Position',[100 100 900 380]);
subplot(1,3,1);
imshow(Iin); title('Input file (original read)');

subplot(1,3,2);
imshow(mY); title('Noisy input (grayscale)');

subplot(1,3,3);
imshow(x); title(sprintf('Denoised (ADMM) \\lambda=%.3g', lambda));

% Optionally save result
imwrite(x, 'denoised_admm.png');

fprintf('Done. Denoised image saved as denoised_admm.png\n');