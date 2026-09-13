function [nll, lz] = mps_nll(mps)
% NLL of a non-normalized MPS: -mean(log |Psi|^2) + log Z.
% Equivalent to mps.compute_nll() when the MPS is canonical (log Z = 0),
% but works without re-canonicalizing, which is what makes post-training
% quantization affordable here.
mps.merged_tensor = [];
psi = mps.recompute_psi();
lz  = mps_logZ(mps);
nll = -mean(log(psi(:).^2)) + lz;
end
