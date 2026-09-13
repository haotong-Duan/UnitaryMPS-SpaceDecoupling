clear,clc
rng("shuffle");clear mps;
n=784;m=100;k=392;% half part
Dmax=200; % max bond dimension
n_batches=1;
% Run 'addpath' for datasets, figure and linesearch (addpath(genpath('..'));)
load('mnist_images_ls.mat');     %emnist
load('mnist_test_images_ls.mat');   %emnist
mps=UMPS_LS(n,train_x_binary,n_batches);
mps.max_bondim=Dmax;
% no learning rate: the step size is chosen by Armijo backtracking
mps.ls_alpha=1;
mps.ls_beta=0.5;
mps.ls_c=1e-4;
mps.ls_maxback=30;
mps.train(5); %loops
%
% Generate directly
%---
% gener=generate_sample(mps,20);
% figure_generate(gener)
%---
% Given the right side, generate the left side
%---
z=test_x_binary(n-k+1:n,:);
s=generate_sample_half(mps,z,size(test_x_binary,2))-1;

figure_mnist(s,n-k);

% line search diagnostics: g^-(X) -> 0 (Theorem 3.9)
figure;
subplot(2,1,1); semilogy(mps.ls_stats(:,2)); ylabel('g^-(X)'); grid on;
subplot(2,1,2); plot(mps.ls_stats(:,1)); ylabel('\alpha'); xlabel('subproblem'); grid on;
