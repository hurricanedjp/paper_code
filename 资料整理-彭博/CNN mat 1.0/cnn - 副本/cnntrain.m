function net = cnntrain(net, x, y, opts , test_x , test_y)
    m = size(x, 3); % m ������� ѵ����������
    numbatches = m / opts.batchsize;
	% rem: Remainder after division. rem(x,y) is x - n.*y �൱������
	% rem(numbatches, 1) ���൱��ȡ��С�����֣����Ϊ0����������
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
	
    net.rL = [];
    err = [];
    for i = 1 : opts.numepochs
		% disp(X) ��ӡ����Ԫ�ء����X�Ǹ��ַ������Ǿʹ�ӡ����ַ���
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        % tic �� toc ��������ʱ�ģ��������������֮�����ĵ�ʱ��
		tic;
		% P = randperm(N) ����[1, N]֮������������һ����������У�����
		% randperm(6) ���ܻ᷵�� [2 4 5 6 1 3]
		% �������൱�ڰ�ԭ�����������д��ң�������һЩ������ѵ��
        kk = randperm(m);
        for l = 1 : numbatches
			% ȡ������˳����batchsize�������Ͷ�Ӧ�ı�ǩ
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

			% �ڵ�ǰ������Ȩֵ�����������¼�����������
            net = cnnff(net, batch_x); % Feedforward
			% �õ���������������ͨ����Ӧ��������ǩ��bp�㷨���õ���������Ȩֵ
			%��Ҳ������Щ����˵�Ԫ�أ��ĵ���
            net = cnnbp(net, batch_y); % Backpropagation
			% �õ�����Ȩֵ�ĵ����󣬾�ͨ��Ȩֵ���·���ȥ����Ȩֵ
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L; % ���ۺ���ֵ��Ҳ�������ֵ
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L; % ������ʷ�����ֵ���Ա㻭ͼ����
        end
        if rem(i,100)==0
           [er, bad] = cnntest(net, test_x, test_y);
           err=[err er*100];
        end
        toc;
    end
    save err err;
end
