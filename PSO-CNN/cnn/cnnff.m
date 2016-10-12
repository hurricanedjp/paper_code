function net = cnnff(net,x,y,num)
    n = numel(net.layers); % ����
    net.layers{1}.a{1} = x; % ����ĵ�һ��������룬���������������˶��ѵ��ͼ��
    inputmaps = 1; % �����ֻ��һ������map��Ҳ����ԭʼ������ͼ��

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c') % �����
            %  !!below can probably be handled by insane matrix operations
			% ��ÿһ������map������˵������Ҫ��outputmaps����ͬ�ľ����ȥ���ͼ��
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
				% ����һ���ÿһ������map������������map�Ĵ�С���� 
				% ������map�� - ����˵Ŀ� + 1��* ������map�� - ����˸� + 1��
				% ��������Ĳ㣬��Ϊÿ�㶼������������map����Ӧ������������ÿ��map�ĵ���ά
				% ���ԣ������z����ľ��Ǹò������е�����map��
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
					% ����һ���ÿһ������map��Ҳ������������map����ò�ľ���˽��о��
					% Ȼ�󽫶���һ������map�����н����������Ҳ����˵����ǰ���һ������map����
					% ��һ�־����ȥ�����һ�������е�����map��Ȼ����������map��Ӧλ�õľ��ֵ�ĺ�
					% ���⣬��Щ���Ļ���ʵ��Ӧ���У���������ȫ��������map���ӵģ��п���ֻ�����е�ĳ��������
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
				% ���϶�Ӧλ�õĻ�b��Ȼ������sigmoid�����������map��ÿ��λ�õļ���ֵ����Ϊ�ò��������map
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's') % �²�����
            %  downsample
            for j = 1 : inputmaps
                %  !! replace with variable
				% ��������Ҫ��scale=2��������ִ��mean pooling����ô���Ծ����СΪ2*2��ÿ��Ԫ�ض���1/4�ľ����
				z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid'); 
				% ��Ϊconvn������Ĭ�Ͼ������Ϊ1����pooling����������û���ص��ģ����Զ�������ľ�����
				% ����pooling�Ľ����Ҫ������õ��ľ���������scale=2Ϊ���������Ű�mean pooling��ֵ������
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
	% �����һ��õ�������map����һ����������Ϊ������ȡ������������
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a) % ���һ�������map�ĸ���
        sa = size(net.layers{n}.a{j}); % ��j������map�Ĵ�С
		% �����е�����map����һ��������������һά���Ƕ�Ӧ������������ÿ������һ�У�ÿ��Ϊ��Ӧ����������
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
        
    end
    
    %  feedforward into output perceptrons
	% ����������������ֵ��sigmoid(W*X + b)��ע����ͬʱ������batchsize�����������ֵ
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));  
    %  error
    net.e = net.o - y;
    %  loss function
    % ���ۺ����� �������,����Ӧ�Ⱥ��� 
    net.fitness(num) = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);%�������size��net.e,2����Ϊ������
end
