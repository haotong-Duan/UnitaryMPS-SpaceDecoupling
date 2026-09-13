clear; clc; rng("shuffle");
% ==================== Parameters ====================
N          = 256;   % number of qubits
d          = 2;     % physical dimension
D          = 16;    % bond dimension of MPS
n_samples  = 1000;  % training samples
n_test     = 300;   % test samples
fprintf('Generating dataset using random MPS...\n');
Dims = ones(1, N+1);
for n = 2:N
    Dims(n) = min(Dims(n-1)*d, D);
end
for n = N:-1:2
    Dims(n) = min(Dims(n), Dims(n+1)*d);
end

MPS = cell(1, N);
for n = 1:N
    MPS{n} = randn(Dims(n), d, Dims(n+1));
end

for n = 1:N-1
    [MPS{n}, R] = left_canon_site(MPS{n});
    MPS{n+1} = absorb_left(MPS{n+1}, R);
end
MPS{N} = MPS{N} / norm(MPS{N}(:));
samples_train = sample_mps(MPS, N, d, n_samples);
samples_test  = sample_mps(MPS, N, d, n_test);
fprintf('Done!\n');
fprintf('Train size: %d x %d\n', size(samples_train));
fprintf('Test size : %d x %d\n', size(samples_test));

% ==================== Functions ====================
function [A_new, R_factor] = left_canon_site(A_n)
    [DL, dp, DR] = size(A_n);
    M = reshape(permute(A_n, [2 1 3]), [dp*DL, DR]);
    [Q, R] = qr(M, 0);
    new_D = size(Q, 2);
    A_new = permute(reshape(Q, [dp, DL, new_D]), [2 1 3]);
    R_factor = R;
end

function A_np1 = absorb_left(A_np1, R_factor)
    [DL_old, dp, DR] = size(A_np1);
    M = R_factor * reshape(A_np1, [DL_old, dp*DR]);
    A_np1 = reshape(M, [size(R_factor,1), dp, DR]);
end

function samples = sample_mps(MPS, N, d, K)
    samples = zeros(K, N);
    for k = 1:K
        rho = 1;
        for n = 1:N
            probs = zeros(1, d);
            vecs = cell(1, d);
            for s = 1:d
                A_s = reshape(MPS{n}(:,s,:), [], size(MPS{n},3));
                vecs{s} = rho * A_s;
                probs(s) = sum(vecs{s}.^2);
            end
            probs = max(probs, 0);
            probs = probs / sum(probs);
            r = rand();
            cp = cumsum(probs);
            s_chosen = find(cp >= r, 1);
            samples(k, n) = s_chosen;
            nm = norm(vecs{s_chosen});
            if nm > 1e-15
                rho = vecs{s_chosen} / nm;
            else
                rho = vecs{s_chosen};
            end
        end
    end
end