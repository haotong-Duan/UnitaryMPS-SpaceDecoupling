clear,clc
% initial MPS. Run 'addpath' for datasets, figure and linesearch.
n=784;Dmax=200;n_batches=1;n_loops=3;n_train=100;seed=2024;
lr=0.1; 
load('mnist_images.mat');
data=train_x_binary(:,1:n_train);

rng(seed); a=UMPS_SD(n,data,n_batches); a.max_bondim=Dmax; a.learning_rate=lr;
t=tic; a.train(n_loops); ta=toc(t);

rng(seed); b=UMPS_LS(n,data,n_batches); b.max_bondim=Dmax;
t=tic; b.train(n_loops); tb=toc(t);

fprintf('\n%-8s %10s %10s %8s %10s\n','method','NLL','time/s','r_max','r_mean');
fprintf('%-8s %10.3f %10.2f %8d %10.2f\n','sd',a.nll_history(end),ta,max(a.ttrank),mean(a.ttrank));
fprintf('%-8s %10.3f %10.2f %8d %10.2f\n','ls',b.nll_history(end),tb,max(b.ttrank),mean(b.ttrank));

figure;
subplot(1,2,1); hold on; grid on;
plot(a.nll_history,'-o'); plot(b.nll_history,'-s');
xlabel('loop'); ylabel('NLL'); legend('sd','ls');
subplot(1,2,2); hold on; grid on;
plot(cumsum(a.time_history),a.nll_history,'-o');
plot(cumsum(b.time_history),b.nll_history,'-s');
xlabel('time [s]'); ylabel('NLL'); legend('sd','ls');
