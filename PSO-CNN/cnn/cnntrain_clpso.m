function net = cnntrain_clpso(net, x, y, opts)
% %%���弫ֵ��ȫ�ּ�ֵ�����ڸ��弫ֵ��ȫ�弫ֵ�ǲ�̫�ó�ʼ���ģ���Ϊ��Ӱ�����ĸ��£������ʼ�ļ�ֵ��С��ѵ�����ﲻ����
% [bestfitness bestindex]=min(net.fitness);
% gbest=net.par{bestindex};   %ȫ�ֵ�һ�����λ��
% pbest=net.par;    %�����һ�����λ��
% net.fitnesspbest=net.fitness;   %���������Ӧ��ֵ
% net.fitnessgbest=bestfitness;   %ȫ�������Ӧ��ֵ

result=[];

for i = 1 : opts.numepochs
    % disp(X) ��ӡ����Ԫ�ء����X�Ǹ��ַ������Ǿʹ�ӡ����ַ���
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    % tic �� toc ��������ʱ�ģ��������������֮�����ĵ�ʱ��
    tic;
    %���Ӹ���
    for num=1:opts.sizepar
        %��ֵ��������cnnff�ֿ�
        net = cnnassign(net,num);
        % �ڵ�ǰ������Ȩֵ�����������¼�����������
        net = cnnff(net,x,y,num); % Feedforward
        
    end
    %��Ӧ�ȸ���,���弫ֵ��ȫ�弫ֵ���£��ٶȺ�λ�ø���
    %         net = cnnupdate(net,opts);
    net = cnnupdate_clpso(net,i,opts);
    %��ÿ�ε�������Ӧ��ֵ����result
    if isempty(result)
        result=net.fitnessgbest;
    else
        result(end+1)=net.fitnessgbest;
    end
    
    toc;
end
%����������Ӧ�ȵı仯���̣�����������
% disp(['����ʵ�鹲����',num2str(opts.numepochs),'�Σ�����PSOռ' ,num2str(PSO_count), '�Σ�SGDռ' ,num2str(SGD_count), '�Ρ�'])
figure('Name','����ʮ����ʵ��');
plot(result);
title('��Ӧ������');
xlabel('������Ӧ�ȴ���');ylabel('��Ӧ��');

save fitnessgbest_each_iteration46 result;
end
