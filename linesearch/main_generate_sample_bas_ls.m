clear,clc
rng("shuffle");clear mps;
n=256;k=128;
Dmax=500;
n_batches=1;
% Run 'generate_bas(16,16)', then randomly save 1000 for the experiment.
load('bars_stripes_16_1000.mat');
mps=UMPS_LS(n,train_x_binary(:,1:400),n_batches);
mps.max_bondim=Dmax;
mps.ls_alpha=1;
mps.train(5);
% gener=generate_sample(mps,20);
% figure_bas(gener,sqrt(n),size(gener,2));
