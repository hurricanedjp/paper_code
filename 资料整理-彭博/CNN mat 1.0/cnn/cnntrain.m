function net = cnntrain(net, x, y, opts , test_x , test_y)
    m = size(x, 3); % m 保存的是 训练样本个数
    numbatches = m / opts.batchsize;
	% rem: Remainder after division. rem(x,y) is x - n.*y 相当于求余
	% rem(numbatches, 1) 就相当于取其小数部分，如果为0，就是整数
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
	
    net.rL = [];
    err = [];
    for i = 1 : opts.numepochs
		% disp(X) 打印数组元素。如果X是个字符串，那就打印这个字符串
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        % tic 和 toc 是用来计时的，计算这两条语句之间所耗的时间
		tic;
		% P = randperm(N) 返回[1, N]之间所有整数的一个随机的序列，例如
		% randperm(6) 可能会返回 [2 4 5 6 1 3]
		% 这样就相当于把原来的样本排列打乱，再挑出一些样本来训练
        kk = randperm(m);
        for l = 1 : numbatches
			% 取出打乱顺序后的batchsize个样本和对应的标签
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

			% 在当前的网络权值和网络输入下计算网络的输出
            net = cnnff(net, batch_x); % Feedforward
			% 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值
			%（也就是那些卷积核的元素）的导数
            net = cnnbp(net, batch_y); % Backpropagation
			% 得到误差对权值的导数后，就通过权值更新方法去更新权值
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L; % 代价函数值，也就是误差值
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L; % 保存历史的误差值，以便画图分析
        end
        if rem(i,100)==0
           [er, bad] = cnntest(net, test_x, test_y);
           err=[err er*100];
        end
        toc;
    end
    save err err;
end
