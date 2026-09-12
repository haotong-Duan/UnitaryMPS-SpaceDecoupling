function [X_new, info] = low_rank_sphere_ls_step(X, gradF, r, alpha0, fhandle, opts)

if nargin < 6, opts = struct(); end
if ~isfield(opts, 'beta'),     opts.beta     = 0.5;   end
if ~isfield(opts, 'c'),        opts.c        = 1e-4;  end
if ~isfield(opts, 'maxback'),  opts.maxback  = 30;    end
if ~isfield(opts, 'tol_rank'), opts.tol_rank = 1e-12; end

F = -(gradF - sum(sum(gradF .* X)) * X);

[U, S, V] = svd(X, 'econ');
sv = diag(S);
if isempty(sv) || sv(1) == 0
    s = 1;
else
    s = max(1, sum(sv > opts.tol_rank * sv(1)));
end
s  = min(s, r);
Us = U(:, 1:s);
Vs = V(:, 1:s);

UtF  = Us' * F;
FV   = F  * Vs;
Pi_s = Us * UtF + FV * Vs' - Us * (UtF * Vs) * Vs';

Xi_s = Pi_s - sum(sum(Pi_s .* X)) * X;

Fp = F - Pi_s;
k  = r - s;
Xi_k = zeros(size(X));
if k > 0
    [Uk, Sk, Vk] = svd(Fp, 'econ');
    kk = min(k, size(Sk, 1));
    if kk > 0
        Xi_k = Uk(:, 1:kk) * Sk(1:kk, 1:kk) * Vk(:, 1:kk)';
    end
end

G     = Xi_s + Xi_k;
g2    = sum(sum(G .* G));
gnorm = sqrt(g2);

info = struct('alpha', 0, 'gnorm', gnorm, 'nback', 0, 'accepted', false);
X_new = X;
if ~isfinite(gnorm) || gnorm <= eps
    return
end

fx    = fhandle(X);
alpha = alpha0;
for j = 0:opts.maxback
    Y  = local_retract(X + alpha * G, r);
    fy = fhandle(Y);
    if isfinite(fy) && fy <= fx - opts.c * alpha * g2
        X_new         = Y;
        info.alpha    = alpha;
        info.nback    = j;
        info.accepted = true;
        return
    end
    alpha = opts.beta * alpha;
end
info.nback = opts.maxback;
end


function Y = local_retract(Z, r)
[Uz, Sz, Vz] = svd(Z, 'econ');
rr = min(r, size(Sz, 1));
Y  = Uz(:, 1:rr) * Sz(1:rr, 1:rr) * Vz(:, 1:rr)';
nY = norm(Y, 'fro');
if nY > 0
    Y = Y / nY;
end
end
