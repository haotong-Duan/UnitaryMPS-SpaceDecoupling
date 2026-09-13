function [mps, stats] = quantize_mps_turboquant(mps, b, opts)
% Quantize every core tensor of an MPS with TurboQuant_mse and write the
% dequantized cores back. The MPS is re-canonicalized afterwards, which
% restores Z = 1 for the quantized model.
%
%   b     : bit-width per parameter
%   opts  : .L    block length, power of two (default 1024)
%           .seed base seed (default 0)
%           .skip_canonical  do not re-canonicalize (default true)
%           .verbose         progress printing (default true)
%
% stats  : per-core and aggregate relative errors and bit cost.

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'L'),    opts.L = 1024; end
if ~isfield(opts, 'seed'), opts.seed = 0; end
if ~isfield(opts, 'skip_canonical'), opts.skip_canonical = true; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end

N = numel(mps.tensors);
rel = zeros(1, N);
nparam = zeros(1, N);
nbits  = zeros(1, N);

for i = 1:N
    A  = mps.tensors{i};
    sz = [size(A, 1), size(A, 2), size(A, 3)];
    x  = A(:);
    code = tq_compress(x, b, opts.L, opts.seed + i);
    xq   = tq_decompress(code);
    xq   = xq(:);
    rel(i)    = norm(xq - x) / max(norm(x), realmin);
    nparam(i) = numel(x);
    nbits(i)  = numel(code.idx) * b + numel(code.sigma) * 32 + 32;
    mps.tensors{i} = reshape(xq, sz);
    if opts.verbose && mod(i, 100) == 0
        fprintf('  quantized %d/%d cores\n', i, N);
    end
end

if ~opts.skip_canonical
    mps.current_bond = 1;
    mps.left_canonical();
end

stats = struct();
stats.b            = b;
stats.rel_per_core = rel;
stats.rel_mean     = mean(rel);
stats.rel_max      = max(rel);
stats.n_param      = sum(nparam);
stats.bits_total   = sum(nbits);
stats.bits_per_par = sum(nbits) / sum(nparam);
stats.mb           = sum(nbits) / 8 / 2^20;
stats.mb_fp64      = sum(nparam) * 8 / 2^20;
stats.compression  = 64 / stats.bits_per_par;
end
