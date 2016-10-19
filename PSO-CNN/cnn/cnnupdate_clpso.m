function net=cnnupdate_clpso(net,i,opts)

%% 更新粒子的速度位置

for num=1:opts.sizepar
    net.Pc(num)=0.05+0.45*(exp(10*(num-1)/(opts.sizepar-1))-1)/(exp(10)-1);%参见clpso文章公式10
    
    if(net.flag(num)>=opts.m)
        %实现选择粒子的过程
        for d=1:size(net.par{num})
            if rand>net.Pc(num)
                net.pbestpar{num}(d)=net.pbestpar{num}(d);
            else
                f1=ceil(rand*opts.sizepar); f2=ceil(rand*opts.sizepar);
                if net.fitnesspbest(f1)>net.fitnesspbest(f2)
                    net.pbestpar{num}(d)=net.pbestpar{f1}(d);
                else
                    net.pbestpar{num}(d)=net.pbestpar{f2}(d);
                end
            end
        end
        net.flag(num)=0;
    end
    %速度更新
    opts.w(i)=opts.w0+(opts.w1-opts.w0)*i/opts.numepochs;
    rand_D=rand(size(net.par{num})); %设定rand_D使每一维有不同的rand
    %速度更新
    net.vel{num} = opts.w(i)*net.vel{num} + opts.c*rand_D.*(net.pbestpar{num}-net.par{num});%用点乘使内部元素互乘
%     net.vel{num}(net.vel{num}<0.01)= net.vel{num}(net.vel{num}<0.01)/0.6; %速度放大
    %需要考虑vel的最大值，限制无规律的跳动
    net.vel{num}(net.vel{num}>opts.velmax)=opts.velmax;
    net.vel{num}(net.vel{num}<opts.velmin)=opts.velmin;
    
    %粒子位置更新
    net.par{num}=net.par{num}+net.vel{num};
    %使粒子限制在解空间
    net.par{num}(net.par{num}>opts.parmax)=opts.parmax;
    net.par{num}(net.par{num}<opts.parmin)=opts.parmin;
    
    %个体极值更新
    if net.fitness(num)<net.fitnesspbest(num)
        net.fitnesspbest(num)=net.fitness(num);
        net.pbestpar{num}=net.par{num};
        net.flag(num)=0;
        
        %群体极值更新
        if net.fitness(num)<net.fitnessgbest
            net.fitnessgbest=net.fitness(num);
            net.gbestpar=net.par{num};
        end
        
    else
        net.flag(num)=net.flag(num)+1;
    end
    
    % 自适应变异，需要考虑位置的最大值
    total=numel(net.par{num});
    pos=unidrnd(total,1,floor(total/21));
    if rand>0.95
        net.par{num}(pos)=5*rands(1);
    end
    
end


%将全局最优粒子赋给第31个粒子
net.par{opts.sizepar+1}=net.gbestpar;