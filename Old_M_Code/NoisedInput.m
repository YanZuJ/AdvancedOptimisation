% tv_denoise_admm_variants.m
% TV / Tikhonov denoising with selectable regulariser.
% Usage: edit parameters below and run.

clear; close all; clc;

%% Parameters (edit these)
noisyFileName = 'noisyboat.png';  % input noisy image file
useColor = false;               % true -> process luminance channel, false -> grayscale
regType = 'anisotropic';          % 'anisotropic' | 'isotropic' | 'tikhonov'
lambda = 0.00001;                  % regularization weight
rho = 5;                        % ADMM penalty parameter (used for TV variants)
maxIter = 10000;                % ADMM iterations (for TV) or max iters for other methods
tol = 1e-5;                     % stopping tolerance (relative change)
showEvery = 25;                 % display progress every this many iterations

%% Read image (force double in [0,1])
Iin = im2double(imread(noisyFileName));
if useColor && size(Iin,3) == 3
    I_ycbcr = rgb2ycbcr(Iin);
    Ychan = I_ycbcr(:,:,1);
    Cb = I_ycbcr(:,:,2);
    Cr = I_ycbcr(:,:,3);
    mY = Ychan;
else
    if size(Iin,3) == 3
        mY = rgb2gray(Iin);
    else
        mY = Iin;
    end
end

[Ny, Nx] = size(mY);
n = Ny * Nx;
y = mY;                 % noisy image in 2D
yvec = y(:);

%% forward difference (periodic) and adjoint
grad = @(x) deal( ...
    circshift(x, [0 -1]) - x, ...    % Dx (horizontal differences)
    circshift(x, [-1 0]) - x ...     % Dy (vertical differences)
    );

divergence = @(gx, gy) ( ...
    gx - circshift(gx, [0 1]) + ...     % -D_x^T gx  (adjoint)
    gy - circshift(gy, [1 0]) ...
    );

%% Precompute FFT denominator eigenvalues for periodic BC
[ux, uy] = meshgrid(0:(Nx-1), 0:(Ny-1));
wx = 2*pi*ux / Nx;
wy = 2*pi*uy / Ny;
eigDtD = (2 - 2*cos(wx)) + (2 - 2*cos(wy));   % eigenvalues of D^T D
% denom for ADMM x-update: (1 + rho * eigDtD)
denomFFT_admm = 1 + rho * eigDtD;
% denom for direct Tikhonov solve: (1 + lambda * eigDtD)
denomFFT_tikh = 1 + lambda * eigDtD;

%% Handle chosen regulariser
regType = lower(regType);
switch regType
    case 'anisotropic'
        fprintf('Using ANISOTROPIC TV (component-wise L1 on gradients)\n');
        useADMM = true;
    case 'isotropic'
        fprintf('Using ISOTROPIC TV (L2,1 total variation)\n');
        useADMM = true;
    case 'tikhonov'
        fprintf('Using TIKHONOV (quadratic gradient penalty)\n');
        useADMM = false; % solved directly (no ADMM with z)
    otherwise
        error('Unknown regType. Choose ''anisotropic'', ''isotropic'', or ''tikhonov''.');
end

if ~useADMM
    % Direct Tikhonov solver via FFT:
    rhsFFT = fft2(y);                % RHS is y (periodic BC)
    x = real(ifft2(rhsFFT ./ denomFFT_tikh));
    % Optionally iterate for refinement (but closed-form under periodic BC)
    % Clip and display below
else
    % ADMM variables (TV)
    x = y;
    [zx, zy] = deal(zeros(Ny, Nx));
    [ux_d, uy_d] = deal(zeros(Ny, Nx)); % scaled duals
    prevx = x;

    fprintf('ADMM TV denoising: lambda=%.4g, rho=%.4g, maxIter=%d\n', lambda, rho, maxIter);
    for k = 1:maxIter
        % x-update: solve (I + rho D^T D) x = y + rho div(z - u)
        rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
        rhsFFT = fft2(rhs);
        x = real(ifft2(rhsFFT ./ denomFFT_admm));

        % gradients of x
        [dx, dy] = grad(x);

        % z-update: proximal operator for regulariser
        v1 = dx + ux_d;
        v2 = dy + uy_d;

        switch regType
            case 'anisotropic'
                % component-wise soft-thresholding (anisotropic TV)
                thresh = lambda / rho;
                zx = max(abs(v1) - thresh, 0) .* sign(v1);
                zy = max(abs(v2) - thresh, 0) .* sign(v2);

            case 'isotropic'
                % vector (isotropic) shrinkage per pixel:
                % magnitude per pixel
                mag = sqrt(v1.^2 + v2.^2);
                thresh = lambda / rho;
                % avoid division by zero
                scale = max(0, 1 - thresh ./ (mag + 1e-12));
                zx = scale .* v1;
                zy = scale .* v2;

            otherwise
                error('Unexpected branch in z-update');
        end

        % dual update (scaled)
        ux_d = ux_d + (dx - zx);
        uy_d = uy_d + (dy - zy);

        % stopping criterion
        relchg = norm(x(:) - prevx(:)) / max(1e-8, norm(prevx(:)));
        if mod(k, showEvery) == 0 || k == 1 || k == maxIter || relchg < tol
            % objective (compute depending on reg)
            switch regType
                case 'anisotropic'
                    tvTerm = sum(abs(dx(:))) + sum(abs(dy(:)));
                    obj = 0.5 * sum((x(:) - yvec).^2) + lambda * tvTerm;
                case 'isotropic'
                    tvTerm = sum( sqrt(dx(:).^2 + dy(:).^2) );
                    obj = 0.5 * sum((x(:) - yvec).^2) + lambda * tvTerm;
            end
            fprintf('iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
        end
        prevx = x;

        if relchg < tol
            fprintf('Converged (relchg < tol) at iter %d\n', k);
            break;
        end
    end
end

%% Postprocess and results
x = min(max(x, 0), 1);

% PSNR helper (if reference available, set haveRef true and provide mI)
psnrval = @(ref, rec) 10*log10(1 ./ mean((ref(:)-rec(:)).^2));
haveRef = false;
if haveRef
    ref = mI; % ensure variable exists
    fprintf('PSNR(noisy vs ref) = %.2f dB\n', psnrval(ref, y));
    fprintf('PSNR(denoised vs ref) = %.2f dB\n', psnrval(ref, x));
end

%% Display
figure('Name','Denoise result','NumberTitle','off','Position',[100 100 900 380]);
subplot(1,3,1); imshow(Iin); title('Input file (original read)');
subplot(1,3,2); imshow(y); title('Noisy input (grayscale)');
subplot(1,3,3); imshow(x); title(sprintf('Denoised (%s) \\lambda=%.3g', regType, lambda));

imwrite(x, ['denoised_' regType '.png']);
fprintf('Done. Denoised image saved as denoised_%s.png\n', regType);
