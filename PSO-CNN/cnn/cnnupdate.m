function net=cnnupdate(net,opts)
%% 个体极值与全体极值的更新

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

%将全局最优粒子赋给第31个粒子
net.par{opts.sizepar+1}=net.gbestpar;

%% 更新粒子的速度位置

for num=1:opts.sizepar
    %速度更新
    net.vel{num}=opts.w*net.vel{num}+opts.c1*rand*(net.pbestpar{num}-net.par{num})+opts.c2*rand*(net.gbestpar-net.par{num})+net.sumgd;
    %需要考虑vel的最大值
    net.vel{num}(net.vel{num}>opts.velmax)=opts.velmax;
    net.vel{num}(net.vel{num}<opts.velmin)=opts.velmin;
    
    %粒子位置更新
    net.par{num}=net.par{num}+net.vel{num};
    %使粒子限制在解空间
    net.par{num}(net.par{num}>opts.parmax)=opts.parmax;
    net.par{num}(net.par{num}<opts.parmin)=opts.parmin;
    % 自适应变异，需要考虑位置的最大值
    total=numel(net.par{num});
    pos=unidrnd(total,1,floor(total/21));
    if rand>0.95
        net.par{num}(pos)=5*rands(1);
    end
    
end
