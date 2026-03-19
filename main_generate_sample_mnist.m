clear,clc
rng("shuffle");clear mps;
n=784;m=100;k=392;% half part 
Dmax=400; % max bond dimension
n_batches=1;
% Run 'addpath' for datasets
load('mnist_images.mat');     %emnist    
load('mnist_test_images.mat');   %emnist
mps=UMPS_SD(n,train_x_binary,n_batches);
mps.max_bondim=Dmax;
mps.learning_rate=0.001; % often 0.001
mps.train(4); %loops
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
