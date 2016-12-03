function net = cnntrain_v_sgd2(net, x, y, opts)
m = size(x, 3); % m 保存的是 训练样本个数
numbatches = m / opts.batchsize;

net.rL = [];
% err = [];
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
    %可以在外面比较一下选出最优粒子
    for num=1:opts.sizepar
        %个体极值更新
        if net.fitness(num)<net.fitnesspbest(num)
            net.fitnesspbest(num)=net.fitness(num);
            net.pbestpar{num}=net.par{num};
        end
        %群体极值更新
        if net.fitness(num)<net.fitnessgbest
            net.fitnessgbest=net.fitness(num);
            net.gbestpar=net.par{num};
        end
    end
    % 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值
    % (也就是那些卷积核的元素）的导数
    % 将单独的这一次训练所需权值放到第三十一个粒子中,保持程序的兼容性
    num=opts.sizepar+1;
    net.sumgd=zeros(size(net.par{1,1})); %一次sgd的梯度之和，传到速度公式中
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
        net = cnnbp(net,opts,batch_y);
        % 得到误差对权值的导数后，就通过权值更新方法去更新权值
        %         net = cnnapplygrads(net, opts,num);
        net = cnnapplygrads_original(net, opts,num);
        
    end
    %适应度更新,个体极值与全体极值更新；速度和位置更新
    net = cnnupdate(net,opts);
    %         net = cnnupdate_clpso(net,opts);
    toc;
    
    %将每次迭代的适应度值赋给result
    if isempty(result)
        result=net.fitnessgbest;
    else
        result(end+1)=net.fitnessgbest;
    end
end
net.result=result;

figure('Name','第五十三次实验');
title('适应度曲线');
xlabel('计算适应度次数');ylabel('适应度');
saveas(figure(1),'第五十三次实验');

end
