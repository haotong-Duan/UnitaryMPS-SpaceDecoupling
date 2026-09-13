function code = tq_compress(x, b, L, seed)
% TurboQuant_mse (Algorithm 1) on a real vector, block-wise.
%   x    : vector of length d
%   b    : bit-width per coordinate
%   L    : block length, must be a power of two (default 1024)
%   seed : seed of the random sign flips (the rotation is a randomized
%          Hadamard transform, as in Appendix E of the paper)
if nargin < 3 || isempty(L), L = 1024; end
if nargin < 4 || isempty(seed), seed = randi(2^31 - 1); end

x  = x(:);
d  = numel(x);
nb = ceil(d / L);
X  = zeros(L, nb);
X(1:d) = x;

S = tq_signs(L, nb, seed);
Y = tq_fwht(S .* X);

sigma = sqrt(sum(Y.^2, 1) / L);
sigma(sigma == 0) = 1;
Z = Y ./ sigma;

c = tq_codebook(b);
e = [-Inf; (c(1:end-1) + c(2:end)) / 2; Inf];
idx = reshape(discretize(Z(:), e), L, nb);

code = struct('idx', uint16(idx), 'sigma', sigma, 'seed', seed, ...
              'd', d, 'L', L, 'b', b);
end
