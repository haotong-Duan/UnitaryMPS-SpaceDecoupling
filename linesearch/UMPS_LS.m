classdef UMPS_LS < UMPS_SD
    properties
        ls_alpha = 1;
        ls_alpha_max = 10;
        ls_beta = 0.5;
        ls_c = 1e-4;
        ls_maxback = 30;
        ls_stats = [];
    end

    methods
        function schro = UMPS_LS(n, data, n_batches)
            schro@UMPS_SD(n, data, n_batches);
        end

        function gradient_descent(schro, batch)
            [Dl, Dr] = schro.bond_dims();
            B = schro.batch_size;
            data_idx = schro.batch_idx(:, batch);
            sk  = schro.data(schro.current_bond, data_idx);
            sk1 = schro.data(schro.current_bond + 1, data_idx);
            L = schro.cumulants{schro.current_bond}(data_idx, :);
            R = schro.cumulants{schro.current_bond + 1}(data_idx, :);

            P = zeros(B, 2 * Dl);
            Q = zeros(B, 2 * Dr);
            colsQ = 2 * ((1:Dr) - 1);
            for i = 1:B
                P(i, (1:Dl) + Dl * (sk(i) - 1)) = L(i, :);
                Q(i, sk1(i) + colsQ)            = R(i, :);
            end

            X = reshape(schro.merged_tensor, 2 * Dl, 2 * Dr);
            r = min([schro.max_bondim, 2 * Dl, 2 * Dr]);

            psi_cur = sum((P * X) .* Q, 2);
            gradF = -(2 / B) * (P' * (Q ./ psi_cur));

            fobj = @(Y) UMPS_LS.batch_nll(Y, P, Q);
            opts = struct('beta', schro.ls_beta, 'c', schro.ls_c, ...
                          'maxback', schro.ls_maxback);
            alpha0 = min(schro.ls_alpha_max, max(1, schro.ls_alpha / schro.ls_beta^2));

            [X, info] = low_rank_sphere_ls_step(X, gradF, r, alpha0, fobj, opts);
            if info.accepted
                schro.ls_alpha = info.alpha;
            end
            schro.ls_stats(end + 1, :) = [info.alpha, info.gnorm, info.nback];

            X = X / norm(X, 'fro');
            schro.psi(data_idx) = sum((P * X) .* Q, 2);
            schro.merged_tensor = reshape(X, [Dl, 2, 2, Dr]);
        end
    end

    methods (Static)
        function val = batch_nll(X, P, Q)
            p = sum((P * X) .* Q, 2);
            if any(~isfinite(p)) || any(abs(p) < 1e-300)
                val = Inf;
                return
            end
            val = -mean(log(p .^ 2));
        end
    end
end
