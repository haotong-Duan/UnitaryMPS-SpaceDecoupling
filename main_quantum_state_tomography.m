clear mps
rng(4);
samples = samples_train;
n=size(samples,2);
Dmax=400; 
n_batches=1; 
mps=MPS_train_riemann2_dec_end_test(n,samples',n_batches);
mps.max_bondim=Dmax;
mps.learning_rate=0.001; 
mps.train(8 ); 


%% ==================== 关联函数计算与绘图 ====================
fprintf('\n--- Computing Spin-Spin Correlations (Sampling Method) ---\n');

% 设定
i_ref = 1; 
r_range = 1:2:20; % 每隔一个点取一个，图像更清爽
n_test_samples = 5000; % 使用较多样本以减小统计误差

% 1. 从 Ground Truth (DMRG MPS) 生成样本并计算关联
% 假设你之前的 PART 2 已经生成了 samples_test (来自 MPS_gs)
% samples_test 应该是 (n_samples x N)，值为 1 或 2
s_gt = samples_test(1:n_test_samples, :); 
s_gt_spin = 3 - 2*s_gt; % 将 {1,2} 映射到 {1, -1} (即 Pauli-Z 特征值)

corr_gt = zeros(1, length(r_range));
for r_idx = 1:length(r_range)
    r = r_range(r_idx);
    target_site = i_ref + r;
    % 计算 <Z_i * Z_j>
    corr_gt(r_idx) = mean(s_gt_spin(:, i_ref) .* s_gt_spin(:, target_site));
end

% 2. 从你训练好的模型 (schro 对象) 生成样本并计算关联
fprintf('Generating samples from learned UMPS-SD model...\n');
s_learned = mps.generate_sample(n_test_samples)'; % 注意转置对齐 (samples x N)
s_learned_spin = 3 - 2*s_learned; 

corr_learned = zeros(1, length(r_range));
for r_idx = 1:length(r_range)
    r = r_range(r_idx);
    target_site = i_ref + r;
    corr_learned(r_idx) = mean(s_learned_spin(:, i_ref) .* s_learned_spin(:, target_site));
end

%% ==================== Plotting (JMLR Style) ====================
figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 5, 4]);
hold on; box on; grid on;

% 绘制 GT 曲线（黑实线）
plot(r_range, corr_gt, 'k-o', 'LineWidth', 1.2, 'MarkerSize', 5, ...
    'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth (DMRG)');

% 绘制算法训练出的曲线（红虚线）
plot(r_range, corr_learned, 'r--s', 'LineWidth', 1.2, 'MarkerSize', 6, ...
    'DisplayName', 'UMPS-SD (Ours)');

% 细节微调
xlabel('Distance $|i - j|$', 'FontSize', 12, 'Interpreter', 'latex');
ylabel('Correlation $\langle \sigma^z_i \sigma^z_j \rangle$', 'FontSize', 12, 'Interpreter', 'latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10);
legend('Location', 'northeast', 'Interpreter', 'latex');
title('Spin-Spin Correlation Recovery', 'FontSize', 12, 'Interpreter', 'latex');

% 打印前几个点的数据供论文表格参考
fprintf('\nTable Data (First 5 points):\n');
fprintf('Dist\tGT\t\tLearned\n');
for k = 1:min(5, length(r_range))
    fprintf('%d\t\t%.6f\t%.6f\n', r_range(k), corr_gt(k), corr_learned(k));
end





% % 5. 计算保真度 (Fidelity)
% psi_learned = reconstruct_psi_from_class(mps);
% prob_gt = abs(psi_gt).^2;
% prob_learned = abs(psi_learned).^2;
% fidelity_classical = (sum(sqrt(prob_gt .* prob_learned)))^2;
% fprintf('概率分布保真度 (Classical Fidelity): %.4f\n', fidelity_classical);



% fprintf('--- 开始实验结果分析 ---\n');
% 
% %% 1. 训练轨迹分析 (NLL History)
% figure('Color', 'w', 'Name', 'Training Convergence');
% plot(mps.nll_history, '-o', 'LineWidth', 2, 'MarkerSize', 4);
% xlabel('Epoch (Loops)');
% ylabel('Negative Log-Likelihood (NLL)');
% title('UMPS-SD 训练收敛曲线');
% grid on;
% 
% %% 2. 计算保真度 (Fidelity)
% % 由于 N=20，我们可以重建完整的波函数向量进行对比
% fprintf('正在重建 UMPS 完整波函数以计算保真度...\n');
% 
% % 将 MPS 转化为 2^N 维向量
% % 注意：这需要一定的内存 (2^20 约 8MB)
% psi_model = ones(1, 1);
% for i = 1:mps.n
%     % 这里演示简单的收缩逻辑，实际大规模可以使用专门的 contract 函数
%     A = mps.tensors{i}; % [dl, 2, dr]
%     [dl, ~, dr] = size(A);
%     % 这里的重建逻辑基于张量积收缩
%     if i == 1
%         psi_model = reshape(A, 2, dr);
%     else
%         % psi_model: [2^(i-1), dl]
%         % A: [dl, 2, dr]
%         % 结果: [2^(i-1), 2, dr] -> [2^i, dr]
%         A_reshaped = reshape(A, dl, 2 * dr);
%         psi_model = psi_model * A_reshaped; 
%         psi_model = reshape(psi_model, 2^i, []);
%     end
% end
% psi_model = psi_model(:); % 展开为 2^N 维向量
% psi_model = psi_model / norm(psi_model); % 强制归一化
% 
% % 计算 Fidelity: F = |<psi_model | psi0>|^2
% % 这里的 psi0 是你之前生成数据时保存的真实基态
% if exist('psi0', 'var')
%     fidelity = abs(psi_model' * psi0)^2;
%     fprintf('>>> 量子态保真度 (Fidelity): %.6f\n', fidelity);
% else
%     fprintf('提示: 未找到原始基态 psi0，跳过保真度计算。\n');
% end
% 
% %% 3. 物理观测量对比：磁化强度 (Magnetization) <Zi>
% fprintf('正在计算物理观测量对比...\n');
% 
% % 从模型生成新样本进行统计分析
% n_gen = 5000;
% gen_samples = mps.generate_sample(n_gen) - 1; % 转换为 0,1
% 
% % 计算每个位点的平均磁化强度 (0/1 映射到 1/-1)
% mag_model = mean(1 - 2*gen_samples, 2); 
% mag_true = mean(1 - 2*(mps.data - 1), 2); % 训练数据的统计值
% 
% figure('Color', 'w');
% subplot(2,1,1);
% plot(1:mps.n, mag_true, 'k--', 'LineWidth', 1.5); hold on;
% plot(1:mps.n, mag_model, 'ro', 'MarkerSize', 6);
% legend('Ground Truth (Data)', 'UMPS Model');
% xlabel('Site Index'); ylabel('<Z_i>');
% title('各站点磁化强度对比');
% grid on;
% 
% %% 4. 关联函数对比 (Spin-Spin Correlation) <Zi Zj>
% % 计算 C(r) = <Z_mid * Z_{mid+r}>
% mid_idx = floor(mps.n / 2);
% r_range = 0:(mps.n - mid_idx - 1);
% corr_model = zeros(size(r_range));
% corr_true = zeros(size(r_range));
% 
% S_model = 1 - 2*gen_samples;
% S_true = 1 - 2*(mps.data - 1);
% 
% for i = 1:length(r_range)
%     r = r_range(i);
%     corr_model(i) = mean(S_model(mid_idx, :) .* S_model(mid_idx + r, :));
%     corr_true(i) = mean(S_true(mid_idx, :) .* S_true(mid_idx + r, :));
% end
% 
% subplot(2,1,2);
% plot(r_range, corr_true, 'k-s', 'LineWidth', 1.5); hold on;
% plot(r_range, corr_model, 'r-d', 'LineWidth', 1.5);
% legend('Ground Truth (Data)', 'UMPS Model');
% xlabel('Distance r'); ylabel('<Z_j Z_{j+r}>');
% title('自旋关联函数衰减曲线');
% grid on;
% 
% %% 5. 采样分布可视化
% % 选取前 200 个样本看结构
% figure('Color', 'w');
% subplot(1,2,1);
% imagesc(mps.data(:, 1:200)'); colormap gray;
% title('原始训练数据 (Samples)');
% xlabel('Qubit Index'); ylabel('Sample ID');
% 
% subplot(1,2,2);
% imagesc(gen_samples(:, 1:200)'); colormap gray;
% title('UMPS 生成数据 (Generated)');
% xlabel('Qubit Index'); ylabel('Sample ID');


% %% 学习率遍历实验（带保真度自动停止功能）
% rng(1);Dmax=20;
% learning_rates = [0.0001:0.0001:0.5]; % 遍历的学习率范围
% target_fidelity = 0.8; % 设定的自动停止阈值
% max_loops = 2;        % 每种学习率允许的最大迭代次数
% 
% % 存储最终结果
% best_lr = 0;
% best_fid = 0;
% 
% for i = 1:length(learning_rates)
%     lr = learning_rates(i);
%     fprintf('\n===== 正在测试学习率 LR = %.3f =====\n', lr);
%     
%     % 1. 初始化模型
%     mps_test = MPS_train_riemann2_dec_end(n, samples' + 1, n_batches);
%     mps_test.max_bondim = Dmax;
%     mps_test.learning_rate = lr;
%     
%     % 2. 手动控制训练循环，以便每轮检查保真度
%     for loop = 1:max_loops
%         % 执行一轮训练 (调用你类里的核心训练逻辑)
%         % 注意：这里为了能中途检查，建议把 train 方法里的逻辑拆解或者确保能访问中间状态
%         if mps_test.current_bond ~= mps_test.n - 1
%             mps_test.left_canonical();
%         end
%         mps_test.train(1); 
%         
%         % 3. 计算当前轮次的经典保真度
%         psi_l = reconstruct_psi_from_class(mps_test);
%         p_gt = abs(psi_gt).^2;
%         p_l = abs(psi_l).^2;
%         current_fid = (sum(sqrt(p_gt .* p_l)))^2;
%         
%         fprintf('Loop %d: Classical Fidelity = %.4f, NLL = %.4f\n', ...
%                 loop, current_fid, mps_test.nll_history(end));
%         
%         % 4. 自动停止判断
%         if current_fid >= target_fidelity
%             fprintf('成功！保真度达到 %.4f，在学习率 %.3f 下提前停止训练。\n', current_fid, lr);
%             
%             % 记录表现最好的结果
%             if current_fid > best_fid
%                 best_fid = current_fid;
%                 best_lr = lr;
%             end
%             break; % 跳出当前的 max_loops 循环
%         end
%     end
% end
% 
% fprintf('\n实验结束。最佳学习率: %.3f, 最高保真度: %.4f\n', best_lr, best_fid);