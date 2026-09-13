function lz = mps_logZ(mps)
% log of Z = <Psi|Psi> by transfer-matrix contraction, with per-site
% rescaling to avoid overflow. Does not modify the MPS.
N = numel(mps.tensors);
E = 1;
lz = 0;
for i = 1:N
    A  = mps.tensors{i};
    dl = size(A, 1); dr = size(A, 3);
    A1 = reshape(A(:, 1, :), dl, dr);
    A2 = reshape(A(:, 2, :), dl, dr);
    E  = A1' * E * A1 + A2' * E * A2;
    nE = norm(E, 'fro');
    if nE == 0, lz = -Inf; return, end
    E  = E / nE;
    lz = lz + log(nE);
end
lz = lz + log(abs(E));
end
