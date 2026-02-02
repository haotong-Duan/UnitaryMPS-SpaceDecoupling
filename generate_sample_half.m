function [samples] = generate_sample_half(schro,samples_q,n_samples)
    schro.left_canonical();
    q=size(samples_q,1);
    samples = ones(schro.n,n_samples); 
     for a = 1:n_samples
        for i = 1:q
           samples(i+schro.n-q,a) = samples_q(i, a);
        end  
     end
c=cell(n_samples,1);
for k=1:n_samples
c{k}(1)=1;
end
     for a = 1:n_samples
         for i=schro.n:-1:schro.n-q+1
              if samples(i,a)==2
                  c{a} = reshape(schro.tensors{i}(:,2,:), size(schro.tensors{i},1), size(schro.tensors{i},3))*c{a};
              else 
                  c{a} = reshape(schro.tensors{i}(:,1,:), size(schro.tensors{i},1), size(schro.tensors{i},3))*c{a};
              end
         end
     end                                                                  
    for a = 1:n_samples
        rvec = c{a}; 
        for i = (schro.n - q):-1:1
            vec1 = reshape(schro.tensors{i}(:,1,:), size(schro.tensors{i},1), size(schro.tensors{i},3)) * rvec;
            vec2 = reshape(schro.tensors{i}(:,2,:), size(schro.tensors{i},1), size(schro.tensors{i},3)) * rvec;            
            n1 = vec1' * vec1;
            n2 = vec2' * vec2;            
            if rand() < (n2 / (n1 + n2))
                state = 2; 
                rvec = vec2; 
            else
                state = 1; 
                rvec = vec1; 
            end            
            samples(i, a) = state;
        end
    end
end