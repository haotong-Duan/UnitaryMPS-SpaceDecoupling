function S = tq_signs(L, nb, seed)
% Reproducible random sign flips. The global stream is saved and restored so
% the caller's rng state is untouched; only the seed has to be stored.
st = rng;
rng(seed, 'twister');
S = 2 * double(rand(L, nb) > 0.5) - 1;
rng(st);
end
