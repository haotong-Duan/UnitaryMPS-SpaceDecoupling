function y = tq_fwht(x)
% Orthonormal fast Walsh-Hadamard transform applied to every column of x.
% Self-inverse. Number of rows must be a power of two.
[D, nb] = size(x);
y = x;
h = 1;
while h < D
    y = reshape(y, h, 2, D / (2 * h), nb);
    a = y(:, 1, :, :);
    b = y(:, 2, :, :);
    y(:, 1, :, :) = a + b;
    y(:, 2, :, :) = a - b;
    y = reshape(y, D, nb);
    h = 2 * h;
end
y = y / sqrt(D);
end
