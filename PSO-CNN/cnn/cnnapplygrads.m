function net = cnnapplygrads(net,opts,num)
net.par{num}=[];
for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for ii = 1 : numel(net.layers{l - 1}.a)
                % ����ûʲô��˵�ģ�������ͨ��Ȩֵ���µĹ�ʽ��W_new = W_old - alpha * de/dW������Ȩֵ������
                net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                %�����������һ���������������潫������������
                net.layers{l}.k_ij=reshape(net.layers{l}.k{ii}{j},1,[]);
                %�������ӣ������Ӻ�����³�ʼ���Ĳ���
                net.par{num}=[net.par{num},net.layers{l}.k_ij];
            end
            net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            %��ƫ�ø�ֵ������
            net.par{num}=[net.par{num},net.layers{l}.b{j}];
        end
        
    end
end

net.ffW = net.ffW - opts.alpha * net.dffW;
%��ȫ���ӵ�Ȩֵ����������
net.ff_W=reshape(net.ffW,1,[]);
net.par{num}=[net.par{num},net.ff_W];
net.ffb = net.ffb - opts.alpha * net.dffb;
%��ȫ���ӵĻ�����������
net.ff_b=reshape(net.ffb,1,[]);
net.par{num}=[net.par{num},net.ff_b];
end
