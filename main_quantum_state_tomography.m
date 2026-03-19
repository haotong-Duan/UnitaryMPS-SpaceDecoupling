% First run the file '/datasets/generate_qubits_N'.
clear mps
rng("shuffle");
samples = samples_train;
n=size(samples,2);
Dmax=400; 
n_batches=1; 
mps=MPS_train_riemann2_dec_end_test(n,samples',n_batches);
mps.max_bondim=Dmax;
mps.learning_rate=0.001; 
mps.train(8); 
