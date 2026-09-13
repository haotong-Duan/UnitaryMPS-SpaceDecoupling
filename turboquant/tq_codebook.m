function [c, dmse] = tq_codebook(b)
% Optimal scalar codebook (Lloyd-Max) for a unit-variance Gaussian, which is
% the limit of the Beta coordinate distribution of a randomly rotated vector
% (Lemma 1 of TurboQuant). dmse is the per-coordinate distortion E|x-c(x)|^2,
% so the end-to-end relative MSE of a d-dimensional block equals dmse.
persistent CB
if isempty(CB), CB = cell(1, 16); end
if b <= 16 && ~isempty(CB{b})
    c = CB{b}.c; dmse = CB{b}.dmse;
    return
end
K = 2^b;
Phi  = @(x) 0.5 * (1 + erf(x / sqrt(2)));
phi  = @(x) exp(-x.^2 / 2) / sqrt(2 * pi);

p = (((1:K)' - 0.5)) / K;
c = sqrt(2) * erfinv(2 * p - 1);

% Lloyd iteration; high bit-widths need many sweeps to converge in the tails
for it = 1:50000
    e  = [-Inf; (c(1:end-1) + c(2:end)) / 2; Inf];
    m0 = Phi(e(2:end)) - Phi(e(1:end-1));
    m1 = phi(e(1:end-1)) - phi(e(2:end));
    cn = c;
    ok = m0 > 1e-300;
    cn(ok) = m1(ok) ./ m0(ok);
    if max(abs(cn - c)) < 1e-14
        c = cn;
        break
    end
    c = cn;
end

e  = [-Inf; (c(1:end-1) + c(2:end)) / 2; Inf];
m0 = Phi(e(2:end)) - Phi(e(1:end-1));
m1 = phi(e(1:end-1)) - phi(e(2:end));
ef = e; ef(~isfinite(ef)) = 0;
m2 = m0 + ef(1:end-1) .* phi(e(1:end-1)) - ef(2:end) .* phi(e(2:end));
dmse = sum(m2 - 2 * c .* m1 + c.^2 .* m0);

if b <= 16
    CB{b} = struct('c', c, 'dmse', dmse);
end
end
