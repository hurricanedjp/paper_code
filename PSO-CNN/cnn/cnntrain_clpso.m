function net = cnntrain_clpso(net, x, y, opts)
% %%个体极值与全局极值。对于个体极值和全体极值是不太好初始化的，因为会影响后面的更新，如果初始的极值过小，训练都达不到。
% [bestfitness bestindex]=min(net.fitness);
% gbest=net.par{bestindex};   %全局第一次最佳位置
% pbest=net.par;    %个体第一次最佳位置
% net.fitnesspbest=net.fitness;   %个体最佳适应度值
% net.fitnessgbest=bestfitness;   %全局最佳适应度值

result=[];

for i = 1 : opts.numepochs
    % disp(X) 打印数组元素。如果X是个字符串，那就打印这个字符串
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    % tic 和 toc 是用来计时的，计算这两条语句之间所耗的时间
    tic;
    %粒子更新
    for num=1:opts.sizepar
        %赋值函数，与cnnff分开
        net = cnnassign(net,num);
        % 在当前的网络权值和网络输入下计算网络的输出
        net = cnnff(net,x,y,num); % Feedforward
        
    end
    %适应度更新,个体极值与全体极值更新；速度和位置更新
    %         net = cnnupdate(net,opts);
    net = cnnupdate_clpso(net,i,opts);
    %将每次迭代的适应度值赋给result
    if isempty(result)
        result=net.fitnessgbest;
    else
        result(end+1)=net.fitnessgbest;
    end
    
    toc;
end
%画出最优适应度的变化过程，即收敛过程
% disp(['本次实验共迭代',num2str(opts.numepochs),'次，其中PSO占' ,num2str(PSO_count), '次，SGD占' ,num2str(SGD_count), '次。'])
figure('Name','第四十六次实验');
plot(result);
title('适应度曲线');
xlabel('计算适应度次数');ylabel('适应度');

save fitnessgbest_each_iteration46 result;
end
