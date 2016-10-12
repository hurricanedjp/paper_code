function net=cnnupdate(net,opts)
%% ���弫ֵ��ȫ�弫ֵ�ĸ���

for num=1:opts.sizepar
    %���弫ֵ����
    if net.fitness(num)<net.fitnesspbest(num)
        net.fitnesspbest(num)=net.fitness(num);
        net.pbestpar{num}=net.par{num};
    end
    %Ⱥ�弫ֵ����
    if net.fitness(num)<net.fitnessgbest
        net.fitnessgbest=net.fitness(num);
        net.gbestpar=net.par{num};
    end
end

%��ȫ���������Ӹ�����31������
net.par{opts.sizepar+1}=net.gbestpar;

%% �������ӵ��ٶ�λ��
% %�趨�߽�ֵ
% velmax=10;
% velmin=-10;
% parmax=5;
% parmin=-5;
for num=1:opts.sizepar
    %�ٶȸ���
    net.vel{num}=opts.w*net.vel{num}+opts.c1*rand*(net.pbestpar{num}-net.par{num})+opts.c2*rand*(net.gbestpar-net.par{num});
    %��Ҫ����vel�����ֵ
%      net.vel{num}(net.vel{num}>velmax)=velmax;
%      net.vel{num}(net.vel{num}<velmin)=velmin;
    
    %����λ�ø���
    net.par{num}=net.par{num}+net.vel{num};
    % ����Ӧ���죬��Ҫ����λ�õ����ֵ
    total=numel(net.par{num});
    pos=unidrnd(total,1,floor(total/21));
        if rand>0.95
            net.par{num}(pos)=5*rands(1);
        end
     
end
