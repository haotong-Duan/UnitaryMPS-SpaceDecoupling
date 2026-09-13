function tq_selftest(d, L)
% Check that the measured distortion matches the theory: for a unit vector
% the relative MSE after b-bit TurboQuant should equal tq_codebook's dmse.
if nargin < 1, d = 65536; end
if nargin < 2, L = 1024; end
x = randn(d, 1);
fprintf('%4s %14s %14s %10s\n', 'b', 'measured MSE', 'predicted', 'ratio');
for b = 1:8
    code = tq_compress(x, b, L, 7);
    xq = tq_decompress(code); xq = xq(:);
    meas = sum((xq - x).^2) / sum(x.^2);
    [~, pred] = tq_codebook(b);
    fprintf('%4d %14.3e %14.3e %10.3f\n', b, meas, pred, meas / pred);
end
end
