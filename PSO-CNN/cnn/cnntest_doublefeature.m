function [mmy,er, bad] = cnnSVMtest(net, x, y)
    %  feedforward
    net = cnnff(net, x); % ǰ�򴫲��õ����
    subfv = [];
    i = 5;
    for j = 1 : numel(net.layers{i}.a) % ���һ�������map�ĸ���
        sa = size(net.layers{i}.a{j}); % ��j������map�Ĵ�С
		% �����е�����map����һ��������������һά���Ƕ�Ӧ������������ÿ������һ�У�ÿ��Ϊ��Ӧ����������
        subfv  = [subfv; reshape(net.layers{i}.a{j}, sa(1) * sa(2), sa(3))];   %��platerecognitionʱ��sa(3)��Ϊ1
    end
    mmy = [];
    mmy =[mmy;[net.fv;subfv]];
%     mmy = [];
%     mmy = [mmy;net.fv];
	% [Y,I] = max(X) returns the indices of the maximum values in vector I
    [~, h] = max(net.o); % �ҵ����������Ӧ�ı�ǩ
    [~, a] = max(y); 	 % �ҵ��������������Ӧ������
    bad = find(h ~= a);  % �ҵ����ǲ���ͬ�ĸ�����Ҳ���Ǵ���Ĵ���

    er = numel(bad) / size(y, 2); % ���������
end
