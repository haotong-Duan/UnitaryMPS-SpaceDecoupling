function x = tq_decompress(code)
c = tq_codebook(code.b);
Y = reshape(c(double(code.idx(:))), code.L, []) .* code.sigma;
S = tq_signs(code.L, size(Y, 2), code.seed);
X = S .* tq_fwht(Y);
x = reshape(X(1:code.d), [], 1);
end
