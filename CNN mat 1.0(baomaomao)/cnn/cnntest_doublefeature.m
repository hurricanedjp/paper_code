function [mmy,er, bad] = cnnSVMtest(net, x, y)
    %  feedforward
    net = cnnff(net, x); % 前向传播得到输出
    subfv = [];
    i = 5;
    for j = 1 : numel(net.layers{i}.a) % 最后一层的特征map的个数
        sa = size(net.layers{i}.a{j}); % 第j个特征map的大小
		% 将所有的特征map拉成一条列向量。还有一维就是对应的样本索引。每个样本一列，每列为对应的特征向量
        subfv  = [subfv; reshape(net.layers{i}.a{j}, sa(1) * sa(2), sa(3))];   %跑platerecognition时把sa(3)改为1
    end
    mmy = [];
    mmy =[mmy;[net.fv;subfv]];
%     mmy = [];
%     mmy = [mmy;net.fv];
	% [Y,I] = max(X) returns the indices of the maximum values in vector I
    [~, h] = max(net.o); % 找到最大的输出对应的标签
    [~, a] = max(y); 	 % 找到最大的期望输出对应的索引
    bad = find(h ~= a);  % 找到他们不相同的个数，也就是错误的次数

    er = numel(bad) / size(y, 2); % 计算错误率
end
