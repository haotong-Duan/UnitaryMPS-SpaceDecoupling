clear,clc
rng("shuffle");clear mps;
n=256;k=128;
Dmax=500; 
n_batches=1; 
% Run 'generate_bas(16,16)', then randomly save 1000 for the experiment.
load('bars_stripes_16_1000.mat'); 
mps=UMPS_SD(n,train_x_binary(:,1:400),n_batches);
mps.max_bondim=Dmax;
mps.learning_rate=0.007;
mps.train(5); 
% gener=generate_sample(mps,20);
% figure_bas(gener,sqrt(n),size(gener,2));
% idx = randperm(300,20);
% test_x_binary = train_x_binary(:,idx);
% z=test_x_binary(n-k+1:n,:);
% s=generate_sample_half(mps,z,size(test_x_binary,2))-1;
% figure_bas(s,n-k,sqrt(n));


