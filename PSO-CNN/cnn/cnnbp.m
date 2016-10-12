function net = cnnbp(net, y)
    n = numel(net.layers); % �������

    %  error
    net.e = net.o - y; 
    %  loss function
	% ���ۺ����� �������
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
	% ������Բο� UFLDL �� ���򴫵��㷨 ��˵��
	% ������ ������ ���� �в�
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
	% �в� ���򴫲��� ǰһ��
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1}); % ���һ������map�Ĵ�С����������һ�㶼��ָ������ǰһ��
    fvnum = sa(1) * sa(2); % ��Ϊ�ǽ����һ������map����һ�����������Զ���һ��������˵������ά��������
    for j = 1 : numel(net.layers{n}.a) % ���һ�������map�ĸ���
		% ��fvd���汣���������������������������cnnff.m������������map���ɵģ�������������Ҫ����
		% �任��������map����ʽ��d ������� delta��Ҳ���� ������ ���� �в�
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

	% ���� �����ǰ��Ĳ㣨����������в�ķ�ʽ��ͬ��
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a) % �ò�����map�ĸ���
                % net.layers{l}.d{j} ������� ��l�� �� ��j�� map �� ������map�� Ҳ����ÿ����Ԫ�ڵ��delta��ֵ
				% expand�Ĳ����൱�ڶ�l+1���������map�����ϲ�����Ȼ��ǰ��Ĳ����൱�ڶԸò������a����sigmoid��
				% ������ʽ��ο� Notes on Convolutional Neural Networks
				% for k = 1:size(net.layers{l + 1}.d{j}, 3)
					% net.layers{l}.d{j}(:,:,k) = net.layers{l}.a{j}(:,:,k) .* (1 - net.layers{l}.a{j}(:,:,k)) .*  kron(net.layers{l + 1}.d{j}(:,:,k), ones(net.layers{l + 1}.scale)) / net.layers{l + 1}.scale ^ 2;
				% end
				net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a) % ��l������map�ĸ���
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a) % ��l+1������map�ĸ���
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
	% ������ Notes on Convolutional Neural Networks �в�ͬ������� �Ӳ��� ��û�в�����Ҳû��
	% ��������������Ӳ�������û����Ҫ���Ĳ�����
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
					% dk ������� ���Ծ���� �ĵ���
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
				% db ������� ������bias�� �ĵ���
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
	% ���һ��perceptron��gradient�ļ���
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
