%% tv_denoise_with_expreg.m
% Ensemble denoising with:
% - Anisotropic TV
% - Isotropic TV
% - Higher-order TV
% - Huber TV
% - Exp-gradient regularizer: sum_{ij} ( exp(sqrt(||g_ij||)) - 1 )
% All ADMM-based, denoise luminance if color, produce ensemble outputs.

clear; close all; clc;

%% ---------------- Parameters ----------------
noisyFileName = 'noisytrain.png';
useColor = true;

% regularisation weights (tune these)
lambda_tv      = 1e-4;    % first-order TV weight
lambda_higher  = 5e-5;    % higher-order TV weight (smaller)
lambda_exp     = 1e-4;    % new exp(sqrt(.))-based regulariser weight

rho = 2;                  % ADMM penalty (lower for stability)
maxIter = 200;
tol = 1e-5;
showEvery = 25;

huberDelta = 0.01;        % Huber TV delta

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
    Cb = []; Cr = [];
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
eigDtD  = (2 - 2*cos(wx)) + (2 - 2*cos(wy));          % eigenvalues for D^T D
eigDtD2 = (4 - 4*cos(wx)) + (4 - 4*cos(wy));          % adjusted for second-order op
denomFFT_first  = 1 + rho * eigDtD;                   % (I + rho D^T D)
denomFFT_second = 1 + rho * eigDtD2;                  % (I + rho D2^T D2)

%% ---------------- 1) Anisotropic TV ----------------
x = y;
[zx, zy] = deal(zeros(Ny,Nx));  % aux (z)
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
[zx, zy] = deal(zeros(Ny,Nx)); [ux_d, uy_d] = deal(zeros(Ny,Nx));
prevx = x;
for k = 1:maxIter
    rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
    x = real(ifft2( fft2(rhs) ./ denomFFT_first ));

    [dx, dy] = grad(x);
    v1 = dx + ux_d; v2 = dy + uy_d;
    mag = sqrt(v1.^2 + v2.^2);
    scale = max(zeros(size(mag)), 1 - (lambda_tv ./ (rho * (mag + 1e-12))));
    zx = scale .* v1; zy = scale .* v2;
    ux_d = ux_d + (dx - zx); uy_d = uy_d + (dy - zy);

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

%% ---------------- 3) Higher-order TV (2nd order) ----------------
x = y;
[zx, zy] = deal(zeros(Ny,Nx)); [ux_d, uy_d] = deal(zeros(Ny,Nx));
prevx = x;
for k = 1:maxIter
    rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
    x = real(ifft2( fft2(rhs) ./ denomFFT_second ));

    [dx2, dy2] = grad2(x);
    thresh2 = lambda_higher / (2 * rho);
    zx = max(abs(dx2 + ux_d) - thresh2, 0) .* sign(dx2 + ux_d);
    zy = max(abs(dy2 + uy_d) - thresh2, 0) .* sign(dy2 + uy_d);
    ux_d = ux_d + (dx2 - zx); uy_d = uy_d + (dy2 - zy);

    relchg = norm(x(:)-prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || relchg < tol
        tvTerm = sum(abs(dx2(:))) + sum(abs(dy2(:)));
        obj = 0.5*sum((x(:)-y(:)).^2) + lambda_higher * tvTerm;
        fprintf('High-order TV iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    if relchg < tol, break; end
    prevx = x;
end
x_higher = x;

%% ---------------- 4) Huber TV ----------------
x = y;
[zx, zy] = deal(zeros(Ny,Nx)); [ux_d, uy_d] = deal(zeros(Ny,Nx));
prevx = x;
for k = 1:maxIter
    rhs = y + rho * divergence(zx - ux_d, zy - uy_d);
    x = real(ifft2( fft2(rhs) ./ denomFFT_first ));

    [dx, dy] = grad(x);
    v1 = dx + ux_d; v2 = dy + uy_d;
    mag = sqrt(v1.^2 + v2.^2);
    scale = ones(size(mag));
    mask = mag > huberDelta;
    % element-wise safe operation
    scale(mask) = max( zeros(sum(mask(:)),1), 1 - (lambda_tv ./ (rho * mag(mask))) );
    % map back to matrix (we built scale via logical index)
    s = ones(size(mag));
    s(mask) = scale(mask);
    zx = s .* v1; zy = s .* v2;

    ux_d = ux_d + (dx - zx); uy_d = uy_d + (dy - zy);

    relchg = norm(x(:)-prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || relchg < tol
        huberTerm = sum(huberDelta * ( sqrt(1 + (dx(:).^2 + dy(:).^2)/huberDelta^2) - 1 ));
        obj = 0.5*sum((x(:)-y(:)).^2) + lambda_tv * huberTerm;
        fprintf('Huber TV iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    if relchg < tol, break; end
    prevx = x;
end
x_huber = x;

%% ---------------- 5) Exp(sqrt(.)) gradient regulariser ----------------
% Regularizer: R_exp(x) = sum_{i,j} [ exp( sqrt( || (D x)_{ij} || ) ) - 1 ]
% ADMM: v = D x, minimize (over v) lambda * sum phi(||v_ij||) + (rho/2)|| v - q ||^2
phi = @(t) exp(sqrt(t)) - 1;        % t >= 0
% derivative phi'(t) = (1/(2 sqrt(t))) * exp(sqrt(t)), but numerically we solve prox via 1D minimization

x = y;
[vx, vy] = deal(zeros(Ny,Nx));     % auxiliary v (gradient field)
[bx, by] = deal(zeros(Ny,Nx));     % scaled dual (b)
prevx = x;

% Pixel-wise scalar proximal solver for: argmin_{t>=0} lambda * phi(t) + (rho/2) (t - r)^2
% We solve by Newton with safeguarding per pixel.
prox_scalar = @(r, lam, rho) prox_exp_sqrt_scalar(r, lam, rho);

for k = 1:maxIter
    % x-update
    rhs = y + rho * divergence(vx - bx, vy - by);
    x = real(ifft2( fft2(rhs) ./ denomFFT_first ));

    % compute gradient
    [dx, dy] = grad(x);
    qx = dx + bx; qy = dy + by;
    r = sqrt(qx.^2 + qy.^2);    % radius per pixel

    % v-update: radial shrink using prox for scalar t
    t = zeros(Ny, Nx);
    % iterate over pixels — vectorized-ish via linear indexing for speed
    rq = r(:);
    tvec = zeros(size(rq));
    if any(rq > 0)
        % loop only for nonzero r to avoid sqrt(0) issues
        for idx = 1:numel(rq)
            ri = rq(idx);
            % compute scalar prox
            tvec(idx) = prox_scalar(ri, lambda_exp, rho);
        end
    end
    t = reshape(tvec, Ny, Nx);

    % avoid division by zero
    denom = r + (r==0); % when r==0, set denom=1 to avoid NaN; t will be zero anyway
    vx = (t ./ denom) .* qx;
    vy = (t ./ denom) .* qy;

    % dual update
    bx = bx + (dx - vx);
    by = by + (dy - vy);

    relchg = norm(x(:)-prevx(:)) / max(1e-8, norm(prevx(:)));
    if mod(k,showEvery)==0 || k==1 || relchg < tol
        % compute objective-ish (data + lambda * sum phi(r_t))
        % We evaluate phi on gradient magnitudes of x (not q) for monitoring:
        gradmag = sqrt(dx.^2 + dy.^2);
        obj = 0.5*sum((x(:)-y(:)).^2) + lambda_exp * sum(phi(gradmag(:)));
        fprintf('ExpReg iter %4d: obj=%.6f, relchg=%.3e\n', k, obj, relchg);
    end
    if relchg < tol, break; end
    prevx = x;
end
x_exp = x;

%% ---------------- Compute reg-values for weighting ----------------
[dx_a, dy_a] = grad(x_aniso);         reg_a = sum(abs(dx_a(:)) + abs(dy_a(:)));
[dx_i, dy_i] = grad(x_iso);           reg_i = sum(sqrt(dx_i(:).^2 + dy_i(:).^2));
[dx_h2, dy_h2] = grad2(x_higher);     reg_h2 = sum(abs(dx_h2(:)) + abs(dy_h2(:)));
[dx_hu, dy_hu] = grad(x_huber);       reg_hu = sum(sqrt(dx_hu(:).^2 + dy_hu(:).^2));
[dx_e,  dy_e]  = grad(x_exp);         reg_e = sum( exp(sqrt( sqrt(dx_e(:).^2 + dy_e(:).^2) )) - 1 );  % approximate measure (nested sqrt consistent with implementation)

reg_values = [reg_a, reg_i, reg_h2, reg_hu, reg_e];
weights = 1 ./ (reg_values + eps);   % avoid divide-by-zero
weights = weights / sum(weights);

%% ---------------- Ensemble outputs ----------------
x_list = cat(3, x_aniso, x_iso, x_higher, x_huber, x_exp);
x_ensemble_mean = mean(x_list, 3);
x_ensemble_weighted = weights(1)*x_aniso + weights(2)*x_iso + weights(3)*x_higher + weights(4)*x_huber + weights(5)*x_exp;
x_ensemble_median = median(x_list, 3);

%% ---------------- Color conversion & display/save ----------------
if useColor && ~isempty(Cb) && ~isempty(Cr)
    if ~isequal(size(Cb), [Ny Nx])
        Cb = imresize(Cb, [Ny Nx]); Cr = imresize(Cr, [Ny Nx]);
    end
    aniso_rgb   = ycbcr2rgb(cat(3, x_aniso,   Cb, Cr));
    iso_rgb     = ycbcr2rgb(cat(3, x_iso,     Cb, Cr));
    higher_rgb  = ycbcr2rgb(cat(3, x_higher,  Cb, Cr));
    huber_rgb   = ycbcr2rgb(cat(3, x_huber,   Cb, Cr));
    exp_rgb     = ycbcr2rgb(cat(3, x_exp,     Cb, Cr));
    mean_rgb    = ycbcr2rgb(cat(3, x_ensemble_mean,    Cb, Cr));
    weighted_rgb= ycbcr2rgb(cat(3, x_ensemble_weighted,Cb, Cr));
    median_rgb  = ycbcr2rgb(cat(3, x_ensemble_median,  Cb, Cr));
else
    aniso_rgb = x_aniso; iso_rgb = x_iso; higher_rgb = x_higher;
    huber_rgb = x_huber; exp_rgb = x_exp;
    mean_rgb = x_ensemble_mean; weighted_rgb = x_ensemble_weighted; median_rgb = x_ensemble_median;
end

figure('Name','Reg Outputs','NumberTitle','off','Position',[50 50 1800 500]);
subplot(2,4,1); imshow(Iin); title('Noisy Input');
subplot(2,4,2); imshow(aniso_rgb);   title('Aniso TV');
subplot(2,4,3); imshow(iso_rgb);     title('Iso TV');
subplot(2,4,4); imshow(higher_rgb);  title('Higher-order TV');
subplot(2,4,6); imshow(huber_rgb);   title('Huber TV');
subplot(2,4,7); imshow(exp_rgb);     title('Exp(sqrt(.)) Reg');
subplot(2,4,5); imshow(mean_rgb);    title('Ensemble Mean');
subplot(2,4,8); imshow(weighted_rgb);title('Ensemble Weighted');

imwrite(aniso_rgb,   'denoised_aniso.png');
imwrite(iso_rgb,     'denoised_iso.png');
imwrite(higher_rgb,  'denoised_highorder.png');
imwrite(huber_rgb,   'denoised_huber.png');
imwrite(exp_rgb,     'denoised_expreg.png');
imwrite(mean_rgb,    'denoised_ensemble_mean.png');
imwrite(weighted_rgb,'denoised_ensemble_weighted.png');
imwrite(median_rgb,  'denoised_ensemble_median.png');

fprintf('All outputs saved.\n');

%% ----------------- helper: prox solver for scalar t -----------------
function t = prox_exp_sqrt_scalar(r, lam, rho)
% Solve t = argmin_{t>=0} lam * (exp(sqrt(t)) - 1) + (rho/2)*(t - r)^2
% Use Newton with safeguards. r >= 0.
    if r <= 0
        t = 0;
        return;
    end
    % initial guess: shrink towards r
    t = max(r - lam/rho, 0);
    max_newton = 60;
    for it = 1:max_newton
        if t <= 1e-12
            % use small-t expansion to avoid division by zero
            % phi(t) ~ 1 + sqrt(t) + ... but we just use derivative approximation:
            dphi = 0.5 * (1/sqrt(max(t,1e-12))) * exp(sqrt(max(t,1e-12)));
        else
            dphi = 0.5 * (1 / sqrt(t)) * exp(sqrt(t));
        end
        % objective derivative: lam * dphi + rho*(t - r)
        g = lam * dphi + rho * (t - r);
        % second derivative (for Newton)
        if t <= 1e-12
            ddphi = 0.5 * (-0.5) * (1/(max(t,1e-12)^(3/2))) * exp(sqrt(max(t,1e-12))) + ...
                   0.25 * (1/(max(t,1e-12))) * exp(sqrt(max(t,1e-12)));
            % above approximation; keep it safe
        else
            ddphi = lam * ( ( -0.25 / (t^(3/2)) ) * exp(sqrt(t)) + (0.25 / t) * exp(sqrt(t)) );
        end
        H = ddphi + rho;
        % Newton step
        delta = g / (H + eps);
        t_new = t - delta;
        % safeguard: ensure non-negativity and not too big step
        if t_new < 0
            t_new = 0;
        end
        % damping
        if abs(t_new - t) < 1e-12
            t = t_new; break;
        end
        t = t_new;
    end
    % final clamp
    if t < 0, t = 0; end
end
