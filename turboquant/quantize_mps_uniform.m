function [mps, stats] = quantize_mps_uniform(mps, b, opts)
% Baseline: mode-2 random 2x2 rotation + max-abs uniform k-bit quantization,
% i.e. the scheme currently used in MPS_quant.md.
if nargin < 3, opts = struct(); end
if ~isfield(opts, 'skip_canonical'), opts.skip_canonical = true; end

N = numel(mps.tensors);
rel = zeros(1, N);
nparam = zeros(1, N);

for i = 1:N
    A  = mps.tensors{i};
    sz = [size(A, 1), size(A, 2), size(A, 3)];
    [Q, ~] = qr(randn(2));
    Ar = reshape(permute(A, [1 3 2]), [], 2) * Q;
    maxv  = max(abs(Ar(:)));
    scale = maxv / (2^(b-1) - 1);
    Aq = round(Ar / scale) * scale;
    Aq = permute(reshape(Aq * Q', sz(1), sz(3), 2), [1 3 2]);
    rel(i) = norm(Aq(:) - A(:)) / max(norm(A(:)), realmin);
    nparam(i) = numel(A);
    mps.tensors{i} = Aq;
end

if ~opts.skip_canonical
    mps.current_bond = 1;
    mps.left_canonical();
end

stats = struct('b', b, 'rel_per_core', rel, 'rel_mean', mean(rel), ...
               'rel_max', max(rel), 'n_param', sum(nparam), ...
               'bits_per_par', b + 32 / mean(nparam), ...
               'compression', 64 / (b + 32 / mean(nparam)));
stats.bits_total = stats.bits_per_par * stats.n_param;
stats.mb         = stats.bits_total / 8 / 2^20;
stats.mb_fp64    = stats.n_param * 8 / 2^20;
end
