function net = cnntrain(net, x, y, opts)
m = size(x, 3); % m 保存的是 训练样本个数
numbatches = m / opts.batchsize;
% 	% rem: Remainder after division. rem(x,y) is x - n.*y 相当于求余
% 	% rem(numbatches, 1) 就相当于取其小数部分，如果为0，就是整数
%     if rem(numbatches, 1) ~= 0
%         error('numbatches not integer');
%     end



% %%个体极值与全局极值。对于个体极值和全体极值是不太好初始化的，因为会影响后面的更新，如果初始的极值过小，训练都达不到。
% [bestfitness bestindex]=min(net.fitness);
% gbest=net.par{bestindex};   %全局第一次最佳位置
% pbest=net.par;    %个体第一次最佳位置
% net.fitnesspbest=net.fitness;   %个体最佳适应度值
% net.fitnessgbest=bestfitness;   %全局最佳适应度值

net.rL = [];
% err = [];
result=[];
num_pso=[];%pso误差在result中的索引，方便画图
num_sgd=[];%sgd误差在result中的索引，方便画图
PSO_count=0;
SGD_count=0;
flag_pso=0;
flag_sgd=0;


for i = 1 : opts.numepochs
    % disp(X) 打印数组元素。如果X是个字符串，那就打印这个字符串
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    % tic 和 toc 是用来计时的，计算这两条语句之间所耗的时间
    tic;
    
    %这一次的result还没有计算，用上一次的result与之前隔10次迭代的result计算比较
    %if(i<=25||(result(end)<result(end-10)-0.005)&&(result(end)<result(end-20)-0.01))
    
    %if(i<15||abs(result_PSO-result(end-10))>0.005) %若之前PSO的误差值减去最后的误差,后面的迭代只要满足这个条件即可切换到pso，而不用管SGD进行了多少次
    if(mod(i,100)<=50)%取余运算，每50次切换一次，硬切换PSO与SGD
        
        %粒子更新
        for num=1:opts.sizepar
            %赋值函数，与cnnff分开
            net = cnnassign(net,num);
            % 在当前的网络权值和网络输入下计算网络的输出
            net = cnnff(net,x,y,num); % Feedforward
            
        end
        %适应度更新,个体极值与全体极值更新；速度和位置更新
%         net = cnnupdate(net,opts);
        net = cnnupdate_clpso(net,opts);
        %将每次迭代的适应度值赋给result
        if isempty(result)
            result=net.fitnessgbest;
        else
            result(end+1)=net.fitnessgbest;
        end
        
        %使PSO和SGD分开画图
        flag_sgd=0;
        if(flag_pso==0)
            flag_pso=1;
            num_pso=[num_pso,cell(1)];
        end
        if(isempty(num_pso{end}))
            num_pso{end}=numel(result);
        else
            num_pso{end}(end+1)=numel(result);
        end
        
        PSO_count=PSO_count+1;%累计PSO运行的次数
        result_PSO=net.fitnessgbest;%设置result_PSO变量保存最后一次调用PSO的误差值
    else
        
        % 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值
        % (也就是那些卷积核的元素）的导数
        % 将单独的这一次训练所需权值放到第三十一个粒子中,保持程序的兼容性
        num=opts.sizepar+1;
        
        net = cnnassign(net,num);
        
        %还要用cnnff前向计算一次，否则cnnbp需要的net.o则不是对应gebestpar的，而是上一次pso的最后一个粒子所产生的net.o
        %同时循环迭代时，BP下一次还是BP，则是对应的上一次更新完的粒子进行计算适应度。
        
        % P = randperm(N) 返回[1, N]之间所有整数的一个随机的序列，例如
        % randperm(6) 可能会返回 [2 4 5 6 1 3]
        % 这样就相当于把原来的样本排列打乱，再挑出一些样本来训练
        kk = randperm(m);
        
        for l = 1 : numbatches
            % 取出打乱顺序后的batchsize个样本和对应的标签
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            % 在当前的网络权值和网络输入下计算网络的输出
            net = cnnff(net, batch_x,batch_y,num); % Feedforward
            
            
            %         net = cnnbp(net, y); % Backpropagation
            net = cnnbp(net,batch_y);
            % 得到误差对权值的导数后，就通过权值更新方法去更新权值
            net = cnnapplygrads(net, opts,num);
            
            result(end+1) = 0.99*result(end) + 0.01*net.fitness(num);
            
            %使SGD和PSO分开画图
            flag_pso=0;
            if(flag_sgd==0)
                flag_sgd=1;
                num_sgd=[num_sgd,cell(1)];
            end
            if(isempty(num_sgd{end}))
                num_sgd{end}=numel(result);
            else
                num_sgd{end}(end+1)=numel(result);
            end
            
            %cnnff中的fitness并不一定是fitnessgbest，所以result的显示有可能会向上（梯度下降期间）
            %注意： 这里的result(end+1)是第i代随机梯度每组样本循环的前向误差，即适应度
            %          net = cnnapplygrads_original(net, opts);
            
            %虽然第一次fitness和fitnessgbest是相等的（因为net中的权值是gbestpar），不会更新，
            %但是下一次net.par{num}就是经cnnbp，cnnapplygrads更新过的粒子
            %这样省去了重复调用cnnff函数，节省了时间
            if result(end)<net.fitnessgbest
                net.fitnessgbest=result(end);
                net.gbestpar=net.par{num};
            end
        end
        %         if isempty(net.rL)
        %             net.rL(1) = net.L; % 代价函数值，也就是误差值
        %         end
        %         net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L; % 保存历史的误差值，以便画图分析
        SGD_count=SGD_count+1;
    end
    toc;
end
%画出最优适应度的变化过程，即收敛过程
disp(['本次实验共迭代',num2str(opts.numepochs),'次，其中PSO占' ,num2str(PSO_count), '次，SGD占' ,num2str(SGD_count), '次。'])
figure('Name','第四十四次实验');

for p=1:numel(num_pso)
    plot(num_pso{p},result(num_pso{p}),'b');
    hold on;
end
for s=1:numel(num_sgd)
    plot(num_sgd{s},result(num_sgd{s}),'r');
    hold on;
end

%legend('PSO','SGD');
title('适应度曲线');
xlabel('计算适应度次数');ylabel('适应度');

save fitnessgbest_each_iteration43 result num_pso num_sgd;
end
