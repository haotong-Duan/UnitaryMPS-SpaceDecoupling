classdef UMPS_SD < handle
    properties
        n = 20;             
        current_bond = 1;   
        m = 1;                
        data;               % data (n*m)
        batched_data;       
        batch_idx;          
        n_batches = 1;      
        batch_size = 1;     
        max_bondim = 80;    % r_{max}
        min_bondim = 2;     
        ttrank = [];     
        tensors = {};       
        going_right = true;
        merged_tensor = []; 
        learning_rate = 0.001; % learning_rate
        cumulants = {};     
        psi = [];           
        log_file = '';      
        converge_crit = 0.002; 
        nll_history = [];   
    end
    
    methods
        function schro=UMPS_SD(n,data,n_batches)
            schro.n=n;
            schro.data=data;
            schro.n_batches=n_batches;
            %initialize tensors
            schro.tensors{1}=randn(1,2,schro.min_bondim);
            schro.tensors{n}=randn(schro.min_bondim,2,1);
            for i=2:n-1
                schro.tensors{i}=randn(schro.min_bondim,2,schro.min_bondim);
                schro.ttrank(i-1)=schro.min_bondim;
            end
            for i = 1:n-1
                A = schro.tensors{i};
                [d1, d2, d3] = size(A);
                A_mat = reshape(A, d1*d2, d3);
                [Q, ~] = qr(A_mat, 0);
                schro.tensors{i} = reshape(Q, d1, d2, d3);
            end
            A = schro.tensors{n};
            [d1, d2, d3] = size(A);
            A_mat = reshape(A, d1, d2*d3);
            [Q, ~] = qr(A_mat', 0);
            schro.tensors{n} = reshape(Q', d1, d2, d3);
            schro.ttrank(schro.n)=1;
            schro.ttrank(n-1)=schro.min_bondim;
            schro.tensors{n} = schro.tensors{n} / norm(schro.tensors{n},'fro');
            schro.make_batches();
            schro.current_bond=1;
            schro.left_canonical();           
        end
        function make_batches(schro)
            assert(size(schro.data,1)==schro.n);
            schro.m=size(schro.data,2);
            assert(mod(schro.m,schro.n_batches)==0);
            schro.batch_size=schro.m/schro.n_batches;
            schro.batch_idx = reshape(1:schro.m,schro.batch_size,schro.n_batches);
        end 
        function left_canonical(schro)
            schro.going_right=true;
            for bond=schro.current_bond:schro.n-1
                schro.current_bond=bond;
                schro.merge_bond();
                schro.rebuild_bond(true);
            end
        end
        function merge_bond(schro)
            assert(schro.current_bond>0 && schro.current_bond<schro.n);
            schro.merged_tensor = schro.tensor_product(schro.tensors{schro.current_bond},'ijk',schro.tensors{schro.current_bond+1},'kmn');
            return
        end
        function rebuild_bond(schro,fix_ttrank)
            dl=size(schro.tensors{schro.current_bond},1);
            dr=size(schro.tensors{schro.current_bond+1},3);
            D=size(schro.tensors{schro.current_bond},3);
            schro.merged_tensor=reshape(schro.merged_tensor,2*dl,2*dr);
            [U,S,V]=svd(schro.merged_tensor);
            if(~fix_ttrank)
                s=diag(S)';
                [~,idx]=find(s>=0);
                D=max(idx);
            end
            D=min(D,schro.max_bondim);
            D=min(D,dl*2);
            D=min(D,dr*2);
            U=U(:,1:D);
            S=S(1:D,1:D);
            V=V(:,1:D)';
            if(schro.going_right)
                V=S*V;
                V=V./norm(V,'fro');
            else
                U=U*S;
                U=U./norm(U,'fro');
            end
            schro.tensors{schro.current_bond}=reshape(U,[dl,2,D]);
            schro.ttrank(schro.current_bond)=D;
            schro.tensors{schro.current_bond+1}=reshape(V,[D,2,dr]);
            schro.merged_tensor=[];
        end
        
        function train(schro,n_loops)
            schro.init_cumulants();
            assert(schro.current_bond==schro.n-1); 
            nll_new=Inf;
            for loop=1:n_loops
                tic;
                for batch=1:schro.n_batches
                    %%
                    schro.going_right=false; 
                    for bond=schro.n-1:-1:2
                        schro.current_bond=bond;
                        schro.merge_bond();
                        schro.gradient_descent(batch);
                        schro.rebuild_bond(false);
                        schro.update_cumulants();
                    end
                                    
                    schro.going_right=true; % going right
                    for bond=1:schro.n-2
                        schro.current_bond=bond;
                        schro.merge_bond();
                        schro.gradient_descent(batch);
                        schro.rebuild_bond(false);
                        schro.update_cumulants();
                    end
                end
                nll_old=nll_new;
                nll_new=schro.compute_nll();
                fprintf('/%d <nll>=%.3f, <max_ttrank>=%d, <mean of ttranks>=%.4f\n',loop,nll_new,max(schro.ttrank),mean(schro.ttrank));
                toc;
                schro.nll_history(loop)=nll_new;
                if(~isempty(schro.log_file))
                    fprintf('saving to %s\n',schro.log_file);
                    mpsf=sprintf('%s.mps.mat',schro.log_file);
                    mps=schro;
                    save(mpsf,'mps');
                    s=schro.generate_sample(100)-1;
                    matf=sprintf('%s.sample.mat',schro.log_file);
                    save(matf,'s');
                end
                if(nll_new > nll_old || abs(nll_new-nll_old)<schro.converge_crit)   
                    break;
                end
            end
        end
        
        function init_cumulants(schro)
           schro.cumulants = {};
           schro.cumulants{1} = ones(schro.m,1);
           for i=2:schro.n-1
               schro.cumulants{i} = ones(schro.m,schro.ttrank(i-1));
               for a=1:schro.m
                   schro.cumulants{i}(a,:)=schro.cumulants{i-1}(a,:)*reshape(schro.tensors{i-1}(:,schro.data(i-1,a),:),size(schro.tensors{i-1},1),size(schro.tensors{i-1},3));
               end
           end
           i=schro.n; % for computing psi
           for a=1:schro.m
               schro.psi(a)=schro.cumulants{i-1}(a,:)*squeeze(schro.tensors{i-1}(:,schro.data(i-1,a),:))*squeeze(schro.tensors{i}(:,schro.data(i,a),:));
           end
           assert(schro.current_bond==schro.n-1);
           schro.cumulants{schro.n}=ones(schro.m,1);
        end
                
        function gradient_descent(schro, batch)               
            if (schro.current_bond == 1)
                Dl = 1;
                Dr = schro.ttrank(schro.current_bond + 1);
            elseif (schro.current_bond == schro.n - 1)
                Dr = 1;
                Dl = schro.ttrank(schro.current_bond - 1);
            else
                Dl = schro.ttrank(schro.current_bond - 1);
                Dr = schro.ttrank(schro.current_bond + 1);
            end
            nominator = zeros(Dl, Dr, schro.batch_size);
            for i = 1:schro.batch_size
                idx = schro.batch_idx(i, batch);  
                sample = schro.data(:, idx);
                lvec = schro.cumulants{schro.current_bond}(idx, :);  % 1*Dl
                rvec = schro.cumulants{schro.current_bond + 1}(idx, :); % 1*Dr
                nominator(:, :, i) = lvec' * rvec;
                tmp = reshape(schro.merged_tensor(:, sample(schro.current_bond), sample(schro.current_bond + 1), :), Dl, Dr);
                schro.psi(idx) = lvec * tmp * rvec';  
            end
            
            data_idx = schro.batch_idx(:, batch);  
            samples = schro.data(:, data_idx);
            gradient = zeros(Dl,2,2,Dr);
            grad_mid = cell(2,2);
            for si=1:2
                for sj=1:2
                    idx = logical( (samples(schro.current_bond,:) == si ) .* (samples(schro.current_bond+1,:)==sj) ); 
                    a=nominator(:,:,idx);
                    b=schro.psi(data_idx(idx));
                    grad_mid{si,sj}=zeros(Dl,Dr);
                    if(size(a,3)==0) 
                        grad_mid{si,sj}=-2*schro.merged_tensor(:,si,sj,:);
                    else
                        for i=1:size(a,3)
                            grad_mid{si,sj}=grad_mid{si,sj}+a(:,:,i)./b(i)*2;
                        end
                    end
                    gradient(:,si,sj,:) = grad_mid{si,sj};
                end
            end
            gradient = reshape(gradient,2*Dl,2*Dr);
            schro.merged_tensor=reshape(schro.merged_tensor,2*Dl,2*Dr);
            min1 = min([schro.max_bondim,2*Dl,2*Dr]);
            schro.merged_tensor = low_rank_frobenius_step(schro.merged_tensor,gradient,min1,schro.learning_rate);
            schro.merged_tensor = schro.merged_tensor/norm(reshape(schro.merged_tensor,Dl*2,Dr*2),'fro');
            schro.merged_tensor = reshape(schro.merged_tensor,[Dl,2,2,Dr]);
        end

        
        function update_cumulants(schro)
            if((schro.current_bond==1 && (~schro.going_right)) || (schro.current_bond==schro.n-1 && schro.going_right))
                return
            end
            if(schro.going_right)
                schro.cumulants{schro.current_bond+1} = ones(schro.m,schro.ttrank(schro.current_bond));
                for a=1:schro.m
                    schro.cumulants{schro.current_bond+1}(a,:)=schro.cumulants{schro.current_bond}(a,:)*squeeze(schro.tensors{schro.current_bond}(:,schro.data(schro.current_bond,a),:));
                end
            else
                schro.cumulants{schro.current_bond} = ones(schro.m,schro.ttrank(schro.current_bond));
                for a=1:schro.m
                    if(schro.current_bond==schro.n-1)
                        tmp=reshape(schro.tensors{schro.current_bond+1}(:,schro.data(schro.current_bond+1,a),:),schro.ttrank(schro.current_bond),1);
                    else
                        tmp=reshape(schro.tensors{schro.current_bond+1}(:,schro.data(schro.current_bond+1,a),:),schro.ttrank(schro.current_bond),schro.ttrank(schro.current_bond+1));
                    end
                    schro.cumulants{schro.current_bond}(a,:)=tmp* (schro.cumulants{schro.current_bond+1}(a,:))' ; % Dl*Dr * Dr*1 ->  Dl*1
                end
            end
        end
        
        function nll=compute_nll(schro)
            nll=-mean(log(abs(schro.recompute_psi()).^2));
            return
        end
            
        function [samples]=generate_sample(schro,n_samples)
            schro.left_canonical();
            samples=ones(schro.n,n_samples);
            for a=1:n_samples
                rvec=1;
                for i=schro.n:-1:1
                    vec1=reshape(schro.tensors{i}(:,1,:),size(schro.tensors{i},1),size(schro.tensors{i},3))*rvec; %D*1
                    vec2=reshape(schro.tensors{i}(:,2,:),size(schro.tensors{i},1),size(schro.tensors{i},3))*rvec; %D*1
                    n1=vec1'*vec1;
                    n2=vec2'*vec2;
                    if(rand()< (n2/(n1+n2)))
                        samples(i,a)=2;
                        rvec=vec2;
                    else
                        rvec=vec1;
                    end
                end
            end
            return;
        end
        
        function [psi]=recompute_psi(schro) 
            if(isempty(schro.merged_tensor))
                for idata=1:schro.m
                    sample=schro.data(:,idata);
                    psi=1;
                    for i=1:schro.n
                        psi = psi* reshape(schro.tensors{i}(:,sample(i),:),size(schro.tensors{i},1),size(schro.tensors{i},3));%1*Dl
                    end
                    schro.psi(idata)=psi;
                end
            else
               for idata=1:schro.m
                   sample=schro.data(:,idata);
                    lvec=1;
                    for i=1:schro.current_bond-1
                        lvec = lvec* reshape(schro.tensors{i}(:,sample(i),:),size(schro.tensors{i},1),size(schro.tensors{i},3));%1*Dl
                    end
                    rvec=1;
                    for i=schro.n:-1:schro.current_bond+2
                        rvec = reshape(schro.tensors{i}(:,sample(i),:),size(schro.tensors{i},1),size(schro.tensors{i},3))*rvec; %Dr*1
                    end
                    psi=squeeze(lvec*reshape( schro.merged_tensor(:,sample(schro.current_bond),sample(schro.current_bond+1),:),size(lvec,2),size(rvec,1) )*rvec);
                    schro.psi(idata)=psi;
               end
            end
            psi=schro.psi;
            return
        end
        function [C,cindex] = tensor_product(varargin)
            if nargin == 4
                A = varargin{1};
                aindex = varargin{2};
                B = varargin{3};
                bindex = varargin{4};
            elseif nargin == 5
                cindex = varargin{1};
                A = varargin{2};
                aindex = varargin{3};
                B = varargin{4};
                bindex = varargin{5};
            end
            a_length = length ( aindex );
            b_length = length ( bindex );
            
            size_a = size(A);
            size_a(end+1:a_length) = 1;
            size_b = size(B);
            size_b(end+1:b_length) = 1;
            
            [com_in_a, com_in_b ] = find_common ( aindex, bindex );
            
            if ~all(size_a(com_in_a)==size_b(com_in_b))
                error('The dimention doesnot match!');
            end
            
            diff_in_a = 1:a_length;
            diff_in_a ( com_in_a ) = [];
            diff_in_b = 1:b_length;
            diff_in_b ( com_in_b ) = [];
            temp_idx = [ aindex(diff_in_a) , bindex(diff_in_b) ];
            
            if nargin ==5
                [ ix1,ix2 ] = find_common ( temp_idx , cindex );
                ix_temp (ix2) = ix1 ;
            else
                cindex = temp_idx;
            end
            c_length = length(cindex);
            % mutiply
            if any([ com_in_a diff_in_a ] ~= 1:a_length)
                A = permute( A, [ com_in_a diff_in_a ] );
            end
            if any([ com_in_b diff_in_b ] ~= 1:b_length)
                B = permute( B, [ com_in_b diff_in_b ] );
            end
            
            sda = prod(size_a(diff_in_a));
            sc = prod(size_a(com_in_a));
            sdb = prod(size_b(diff_in_b));
            
            A = reshape(A,[sc,sda,1]);
            B = reshape(B,[sc,sdb,1]);
            
            C = A.' * B ;
            
            C = reshape(C,[size_a(diff_in_a),size_b(diff_in_b),1,1]);
            
            if c_length > 1 && nargin == 5 && any(ix_temp ~= 1:c_length)
                C = permute(C,ix_temp);
            end
            
            function [com_a, com_b] = find_common ( a, b)
            % find the common elements
            a = a.';
            a_len = length( a );
            b_len = length( b );
            a = a(:,ones (1,b_len) );
            b = b( ones(a_len ,1),:);
            %[b a] = meshgrid(b,a);
            [ com_a ,com_b ] = find ( a == b );
            com_a = com_a.';
            com_b = com_b.';
            
                        end
                    end
                end
end




