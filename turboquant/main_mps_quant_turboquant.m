%MAIN_MPS_QUANT_TURBOQUANT
%   Post-training quantization of a trained MPS with TurboQuant, compared
%   against the max-abs uniform baseline.
%   Run 'addpath' for datasets, figure and turboquant first.
%
%   The NLL of a quantized MPS is evaluated as -mean(log|Psi|^2) + log Z with
%   log Z from a transfer-matrix contraction. Re-canonicalizing instead would
%   cost one full SVD sweep over all 784 bonds per bit-width and push every
%   bond dimension to max_bondim, which is both slow and memory-hungry.

clear; clc;
rng(2);

n = 784; k = 392;
Dmax = 200;
n_batches = 1;
BITS = [8];
L = 1024;
N_SHOW = 20;          

load('mnist_images.mat');
load('mnist_test_images.mat');

mps = UMPS_SD(n, train_x_binary, n_batches);
mps.max_bondim = Dmax;
mps.learning_rate = 0.007;
mps.train(3);

A0 = mps.tensors;
r0 = mps.ttrank;
npar = sum(cellfun(@numel, A0));

t = tic;
[nll0, lz0] = mps_nll(mps);
fprintf('\nbaseline: nll = %.4f (compute_nll = %.4f), log Z = %.2e, params = %d, %.1f s\n', ...
        nll0, mps.compute_nll(), lz0, npar, toc(t));
mb0 = npar * 8 / 2^20;
fprintf('memory of one copy of the cores: %.1f MB (fp64)\n\n', mb0);

for b = BITS
    tq_codebook(b);   % fill the codebook cache once
end

nb = numel(BITS);
nll_tq = zeros(1, nb); nll_un = zeros(1, nb);
rel_tq = zeros(1, nb); rel_un = zeros(1, nb);
bpp    = zeros(1, nb); pred   = zeros(1, nb);
mb_tq  = zeros(1, nb); mb_un  = zeros(1, nb);

for j = 1:nb
    b = BITS(j);
    fprintf('--- b = %d ---\n', b);

    t = tic;
    mps.tensors = A0;
    [mps, s_tq] = quantize_mps_turboquant(mps, b, struct('L', L, 'seed', 100 * b, 'verbose', false));
    nll_tq(j) = mps_nll(mps);
    rel_tq(j) = s_tq.rel_mean;
    bpp(j)    = s_tq.bits_per_par;
    mb_tq(j)  = s_tq.mb;
    fprintf('  TurboQuant  nll = %9.4f  rel = %.3e  mem = %6.1f MB (%.1fx)  (%.1f s)\n', ...
            nll_tq(j), rel_tq(j), mb_tq(j), mb0 / mb_tq(j), toc(t));

    t = tic;
    mps.tensors = A0;
    [mps, s_un] = quantize_mps_uniform(mps, b);
    nll_un(j) = mps_nll(mps);
    rel_un(j) = s_un.rel_mean;
    mb_un(j)  = s_un.mb;
    fprintf('  uniform     nll = %9.4f  rel = %.3e  mem = %6.1f MB (%.1fx)  (%.1f s)\n', ...
            nll_un(j), rel_un(j), mb_un(j), mb0 / mb_un(j), toc(t));

    [~, dmse] = tq_codebook(b);
    pred(j) = sqrt(dmse);
end

mps.tensors = A0;
mps.ttrank  = r0;

fprintf('\n%4s %12s %12s %12s %12s %12s %10s %8s\n', 'bits', 'nll(TQ)', 'nll(unif)', ...
        'rel(TQ)', 'rel predicted', 'bits/param', 'memory/MB', 'ratio');
for j = 1:nb
    fprintf('%4d %12.4f %12.4f %12.3e %12.3e %12.2f %10.1f %7.1fx\n', ...
            BITS(j), nll_tq(j), nll_un(j), rel_tq(j), pred(j), bpp(j), ...
            mb_tq(j), mb0 / mb_tq(j));
end
fprintf('%4s %12.4f %12s %12s %12s %12.0f %10.1f %7.1fx\n', ...
        'fp64', nll0, '-', '-', '-', 64, mb0, 1);

figure; set(gcf, 'Color', 'w'); hold on; grid on;
plot(BITS, nll_tq, '-o', 'LineWidth', 1.3);
plot(BITS, nll_un, '-s', 'LineWidth', 1.3);
yline(nll0, '--k');
xlabel('bit-width'); ylabel('NLL'); legend('TurboQuant', 'uniform', 'fp64');
set(gca, 'YScale', 'log');

% completion task with one quantized model (generate_sample_half canonicalizes)
bstar = 6;
mps.tensors = A0; mps.ttrank = r0; mps.current_bond = 1;
mps = quantize_mps_turboquant(mps, bstar, struct('L', L, 'seed', 1, 'verbose', false));
z = test_x_binary(n - k + 1:n, 1:N_SHOW);
s = generate_sample_half(mps, z, N_SHOW) - 1;
figure_mnist(s, n - k);
