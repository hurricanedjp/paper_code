function net = cnnapplygrads(net,opts,num)
net.par{num}=[];
for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for ii = 1 : numel(net.layers{l - 1}.a)
                % 这里没什么好说的，就是普通的权值更新的公式：W_new = W_old - alpha * de/dW（误差对权值导数）
                net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                %将卷积核拉成一条向量，好让下面将参数赋给粒子
                net.layers{l}.k_ij=reshape(net.layers{l}.k{ii}{j},1,[]);
                %赋给粒子，在粒子后加上新初始化的参数
                net.par{num}=[net.par{num},net.layers{l}.k_ij];
            end
            net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            %将偏置赋值给粒子
            net.par{num}=[net.par{num},net.layers{l}.b{j}];
        end
        
    end
end

net.ffW = net.ffW - opts.alpha * net.dffW;
%将全连接的权值连到粒子中
net.ff_W=reshape(net.ffW,1,[]);
net.par{num}=[net.par{num},net.ff_W];
net.ffb = net.ffb - opts.alpha * net.dffb;
%将全连接的基连到粒子中
net.ff_b=reshape(net.ffb,1,[]);
net.par{num}=[net.par{num},net.ff_b];
end
