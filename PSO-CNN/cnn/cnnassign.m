function net = cnnassign(net,num)
n = numel(net.layers); % ����
inputmaps = 1; % �����ֻ��һ������map��Ҳ����ԭʼ������ͼ��
pos_start=1;
for l = 2 : n   %  for each layer
    if strcmp(net.layers{l}.type, 'c') % �����
        % ��ÿһ������map������˵������Ҫ��outputmaps����ͬ�ľ����ȥ���ͼ��
        for j = 1 : net.layers{l}.outputmaps   %  for each output map
            for i = 1 : inputmaps   %  for each input map
                %pos_start��pos_start+net.layers{l}.kernelsize^2-1Ϊ����˶�Ӧ���ӵ�Ԫ��
                net.layers{l}.k_ij=net.par{num}(pos_start:pos_start+net.layers{l}.kernelsize^2-1);
                net.layers{l}.k{i}{j}=reshape(net.layers{l}.k_ij,net.layers{l}.kernelsize,net.layers{l}.kernelsize);
                %��ʱ����ƽ��������1��λ���������һλ����ʱΪƫ��b��λ��
                pos_start=pos_start+net.layers{l}.kernelsize^2;
            end
            
            %  add bias, pass through nonlinearity
            % ��Ӧλ�õĻ�b
            net.layers{l}.b{j}=net.par{num}(pos_start);
            %˳��һλ����һ������˵���ʼλ��
            pos_start=pos_start+1;
        end
        %  set number of input maps to this layers number of outputmaps
        inputmaps = net.layers{l}.outputmaps;
    end
end

%�����net.ffW��������֮���õ���Ӧλ��
net.ff_W=net.par{num}(pos_start:pos_start+numel(net.ffW)-1);
%����Ӧ����Ԫ��ת��Ȩֵ����
net.ffW=reshape(net.ff_W,size(net.ffW,1),size(net.ffW,2));

pos_start=pos_start+numel(net.ffW);
%�����net.ffb��������֮���õ���Ӧλ��
net.ff_b=net.par{num}(pos_start:pos_start+numel(net.ffb)-1);
%����Ӧ����Ԫ��ת��ƫ�þ���
net.ffb=reshape(net.ff_b,size(net.ffb,1),size(net.ffb,2));
end
