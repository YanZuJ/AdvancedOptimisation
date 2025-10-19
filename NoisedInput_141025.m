%% tv_denoise_remove_huber_highorder_vectorized.m
% Ensemble denoising with:
% - Anisotropic TV
% - Isotropic TV
% - Exp-gradient regularizer: sum_{ij} ( exp(sqrt(||g_ij||)) - 1 )
% All ADMM-based, denoise luminance if color, produce ensemble outputs.

clear; close all; clc;

%% ---------------- Parameters ----------------
noisyFileName = 'noisyballoon2.png';
useColor = true;

% regularisation weights (tune these)
lambda_tv = 1e-4;      % first-order TV weight
lambda_exp = 1e-4;      % new exp(sqrt(.))-based regulariser weight
rho = 2;                % ADMM penalty (lower for stability)
maxIter = 200;
tol = 1e-5;
showEvery = 25;

%% ---------------- Read image ----------------
Iin = im2double(imread(noisyFileName));
if useColor && size(Iin,3)==3
    I_ycbcr = rgb2ycbcr(Iin);
    Ychan = I_ycbcr(:,:,1);
    Cb = I_ycbcr(:,:,2);
    Cr = I_ycbcr(:,:,3);
    mY = Ychan;
else
    if size(Iin,3)==3
        mY = rgb2gray(Iin);
    else
        mY = Iin;
    end
    Cb = [];
    Cr = [];
end
[Ny, Nx] = size(mY);
y = mY;

%% ---------------- Operators ----------------
grad = @(x) deal(circshift(x,[0 -1]) - x, circshift(x,[-1 0]) - x);
divergence = @(gx,gy) gx - circshift(gx,[0 1]) + gy - circshift(gy,[1 0]);
grad2 = @(x) deal(circshift(x,[0 -2]) - 2*circshift(x,[0 -1]) + x, ...
                 circshift(x,[-2 0]) - 2*circshift(x,[-1 0]) + x);

%% ---------------- FFT precompute ----------------
[ux, uy] = meshgrid(0:(Nx-1), 0:(Ny-1));
wx = 2*pi*ux/Nx; wy = 2*pi*uy/Ny;
eigDtD = (2 - 2*cos(wx)) + (2 - 2*cos(wy)); % eigenvalues for D^T D
denomFFT_first = 1 + rho * eigDtD; % (I + rho D^T D)

%% ---------------- 1) Anisotropic TV ----------------
x = y;
[zx, zy] = deal(zeros(Ny,Nx)); % aux (z)
[ux_d, uy_d] = deal(zeros(Ny,Nx)); % scaled dual
prevx = x;
for k = 1:maxIter
    rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
    x = real(ifft2( fft2(rhs) ./ denomFFT_first ));
    [dx, dy] = grad(x);
    thresh = lambda_tv / rho;
    zx = max(abs(dx + ux_d) - thresh, 0) .* sign(dx + ux_d);
    zy = max(abs(dy + uy_d) - thresh, 0) .* sign(dy + uy_d);
    ux_d = ux_d + (dx - zx);
    uy_d = uy_d + (dy - zy);
    relchg = norm(x(:)-prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || relchg < tol
        tvTerm = sum(abs(dx(:))) + sum(abs(dy(:)));
        obj = 0.5*sum((x(:)-y(:)).^2) + lambda_tv * tvTerm;
        fprintf('Aniso TV iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    if relchg < tol, break; end
    prevx = x;
end
x_aniso = x;

%% ---------------- 2) Isotropic TV ----------------
x = y;
[zx, zy] = deal(zeros(Ny,Nx));
[ux_d, uy_d] = deal(zeros(Ny,Nx));
prevx = x;
for k = 1:maxIter
    rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
    x = real(ifft2( fft2(rhs) ./ denomFFT_first ));
    [dx, dy] = grad(x);
    v1 = dx + ux_d;
    v2 = dy + uy_d;
    mag = sqrt(v1.^2 + v2.^2);
    scale = max(zeros(size(mag)), 1 - (lambda_tv ./ (rho * (mag + 1e-12))));
    zx = scale .* v1;
    zy = scale .* v2;
    ux_d = ux_d + (dx - zx);
    uy_d = uy_d + (dy - zy);
    relchg = norm(x(:)-prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || relchg < tol
        tvTerm = sum(sqrt(dx(:).^2 + dy(:).^2));
        obj = 0.5*sum((x(:)-y(:)).^2) + lambda_tv * tvTerm;
        fprintf('Iso TV iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    if relchg < tol, break; end
    prevx = x;
end
x_iso = x;

%% ---------------- 3) Exp(sqrt(.)) gradient regulariser (vectorized) ----------------
phi = @(t) exp(sqrt(t)) - 1;
x = y;
[vx, vy] = deal(zeros(Ny,Nx));
[bx, by] = deal(zeros(Ny,Nx));
prevx = x;

for k = 1:maxIter
    % x-update
    rhs = y + rho * divergence(vx - bx, vy - by);
    x = real(ifft2( fft2(rhs) ./ denomFFT_first ));
    
    % compute gradient
    [dx, dy] = grad(x);
    qx = dx + bx; qy = dy + by;
    r = sqrt(qx.^2 + qy.^2); % radial magnitude
    
    % vectorized prox update
    t = max(r - lambda_exp/rho, 0); % initial approximation
    max_newton = 20;
    tol_newton = 1e-12;
    
    for it = 1:max_newton
        t_safe = max(t, 1e-12);
        dphi = 0.5 ./ sqrt(t_safe) .* exp(sqrt(t_safe));
        g = lambda_exp * dphi + rho * (t - r);
        ddphi = -0.25 ./ (t_safe.^(3/2)) .* exp(sqrt(t_safe)) + 0.25 ./ t_safe .* exp(sqrt(t_safe));
        H = lambda_exp * ddphi + rho;
        t_new = max(t - g ./ (H + eps), 0);
        if max(abs(t_new(:) - t(:))) < tol_newton
            t = t_new;
            break;
        end
        t = t_new;
    end
    
    vx = (t ./ max(r, 1e-12)) .* qx;
    vy = (t ./ max(r, 1e-12)) .* qy;
    
    % dual update
    bx = bx + (dx - vx);
    by = by + (dy - vy);
    
    relchg = norm(x(:)-prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || relchg < tol
        gradmag = sqrt(dx.^2 + dy.^2);
        obj = 0.5*sum((x(:)-y(:)).^2) + lambda_exp * sum(phi(gradmag(:)));
        fprintf('ExpReg iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    if relchg < tol, break; end
    prevx = x;
end
x_exp = x;

%% ---------------- Compute reg-values for weighting ----------------
[dx_a, dy_a] = grad(x_aniso);
reg_a = sum(abs(dx_a(:)) + abs(dy_a(:)));
[dx_i, dy_i] = grad(x_iso);
reg_i = sum(sqrt(dx_i(:).^2 + dy_i(:).^2));
[dx_e, dy_e] = grad(x_exp);
reg_e = sum( exp(sqrt( sqrt(dx_e(:).^2 + dy_e(:).^2) )) - 1 );
reg_values = [reg_a, reg_i, reg_e];
weights = 1 ./ (reg_values + eps);
weights = weights / sum(weights);

%% ---------------- Ensemble outputs ----------------
x_list = cat(3, x_aniso, x_iso, x_exp);
x_ensemble_mean = mean(x_list, 3);
x_ensemble_weighted = weights(1)*x_aniso + weights(2)*x_iso + weights(3)*x_exp;
x_ensemble_median = median(x_list, 3);

%% ---------------- Color conversion & display/save ----------------
if useColor && ~isempty(Cb) && ~isempty(Cr)
    if ~isequal(size(Cb), [Ny Nx])
        Cb = imresize(Cb, [Ny Nx]); Cr = imresize(Cr, [Ny Nx]);
    end
    aniso_rgb = ycbcr2rgb(cat(3, x_aniso, Cb, Cr));
    iso_rgb = ycbcr2rgb(cat(3, x_iso, Cb, Cr));
    exp_rgb = ycbcr2rgb(cat(3, x_exp, Cb, Cr));
    mean_rgb = ycbcr2rgb(cat(3, x_ensemble_mean, Cb, Cr));
    weighted_rgb= ycbcr2rgb(cat(3, x_ensemble_weighted,Cb, Cr));
    median_rgb = ycbcr2rgb(cat(3, x_ensemble_median, Cb, Cr));
else
    aniso_rgb = x_aniso;
    iso_rgb = x_iso;
    exp_rgb = x_exp;
    mean_rgb = x_ensemble_mean;
    weighted_rgb = x_ensemble_weighted;
    median_rgb = x_ensemble_median;
end

figure('Name','Reg Outputs','NumberTitle','off','Position',[50 50 1400 600]);
subplot(2,4,1); imshow(Iin); title('Noisy Input');
subplot(2,4,2); imshow(aniso_rgb); title('Aniso TV');
subplot(2,4,3); imshow(iso_rgb); title('Iso TV');
subplot(2,4,4); imshow(exp_rgb); title('Exp(sqrt(.)) Reg');
subplot(2,4,5); imshow(mean_rgb); title('Ensemble Mean');
subplot(2,4,6); imshow(weighted_rgb); title('Ensemble Weighted');
subplot(2,4,7); imshow(median_rgb); title('Ensemble Median');

imwrite(aniso_rgb, 'denoised_aniso.png');
imwrite(iso_rgb, 'denoised_iso.png');
imwrite(exp_rgb, 'denoised_expreg.png');
imwrite(mean_rgb, 'denoised_ensemble_mean.png');
imwrite(weighted_rgb,'denoised_ensemble_weighted.png');
imwrite(median_rgb, 'denoised_ensemble_median.png');
fprintf('All outputs saved.\n');
