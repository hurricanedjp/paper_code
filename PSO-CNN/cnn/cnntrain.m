function net = cnntrain(net, x, y, opts)
m = size(x, 3); % m ������� ѵ����������
numbatches = m / opts.batchsize;
% 	% rem: Remainder after division. rem(x,y) is x - n.*y �൱������
% 	% rem(numbatches, 1) ���൱��ȡ��С�����֣����Ϊ0����������
%     if rem(numbatches, 1) ~= 0
%         error('numbatches not integer');
%     end



% %%���弫ֵ��ȫ�ּ�ֵ�����ڸ��弫ֵ��ȫ�弫ֵ�ǲ�̫�ó�ʼ���ģ���Ϊ��Ӱ�����ĸ��£������ʼ�ļ�ֵ��С��ѵ�����ﲻ����
% [bestfitness bestindex]=min(net.fitness);
% gbest=net.par{bestindex};   %ȫ�ֵ�һ�����λ��
% pbest=net.par;    %�����һ�����λ��
% net.fitnesspbest=net.fitness;   %���������Ӧ��ֵ
% net.fitnessgbest=bestfitness;   %ȫ�������Ӧ��ֵ

net.rL = [];
% err = [];
result=[];
num_pso=[];%pso�����result�е����������㻭ͼ
num_sgd=[];%sgd�����result�е����������㻭ͼ
PSO_count=0;
SGD_count=0;
flag_pso=0;
flag_sgd=0;


for i = 1 : opts.numepochs
    % disp(X) ��ӡ����Ԫ�ء����X�Ǹ��ַ������Ǿʹ�ӡ����ַ���
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    % tic �� toc ��������ʱ�ģ��������������֮�����ĵ�ʱ��
    tic;
    
    %��һ�ε�result��û�м��㣬����һ�ε�result��֮ǰ��10�ε�����result����Ƚ�
    %if(i<=25||(result(end)<result(end-10)-0.005)&&(result(end)<result(end-20)-0.01))
    
    %if(i<15||abs(result_PSO-result(end-10))>0.005) %��֮ǰPSO�����ֵ��ȥ�������,����ĵ���ֻҪ����������������л���pso�������ù�SGD�����˶��ٴ�
    if(mod(i,100)<=50)%ȡ�����㣬ÿ50���л�һ�Σ�Ӳ�л�PSO��SGD
        
        %���Ӹ���
        for num=1:opts.sizepar
            %��ֵ��������cnnff�ֿ�
            net = cnnassign(net,num);
            % �ڵ�ǰ������Ȩֵ�����������¼�����������
            net = cnnff(net,x,y,num); % Feedforward
            
        end
        %��Ӧ�ȸ���,���弫ֵ��ȫ�弫ֵ���£��ٶȺ�λ�ø���
%         net = cnnupdate(net,opts);
        net = cnnupdate_clpso(net,opts);
        %��ÿ�ε�������Ӧ��ֵ����result
        if isempty(result)
            result=net.fitnessgbest;
        else
            result(end+1)=net.fitnessgbest;
        end
        
        %ʹPSO��SGD�ֿ���ͼ
        flag_sgd=0;
        if(flag_pso==0)
            flag_pso=1;
            num_pso=[num_pso,cell(1)];
        end
        if(isempty(num_pso{end}))
            num_pso{end}=numel(result);
        else
            num_pso{end}(end+1)=numel(result);
        end
        
        PSO_count=PSO_count+1;%�ۼ�PSO���еĴ���
        result_PSO=net.fitnessgbest;%����result_PSO�����������һ�ε���PSO�����ֵ
    else
        
        % �õ���������������ͨ����Ӧ��������ǩ��bp�㷨���õ���������Ȩֵ
        % (Ҳ������Щ����˵�Ԫ�أ��ĵ���
        % ����������һ��ѵ������Ȩֵ�ŵ�����ʮһ��������,���ֳ���ļ�����
        num=opts.sizepar+1;
        
        net = cnnassign(net,num);
        
        %��Ҫ��cnnffǰ�����һ�Σ�����cnnbp��Ҫ��net.o���Ƕ�Ӧgebestpar�ģ�������һ��pso�����һ��������������net.o
        %ͬʱѭ������ʱ��BP��һ�λ���BP�����Ƕ�Ӧ����һ�θ���������ӽ��м�����Ӧ�ȡ�
        
        % P = randperm(N) ����[1, N]֮������������һ����������У�����
        % randperm(6) ���ܻ᷵�� [2 4 5 6 1 3]
        % �������൱�ڰ�ԭ�����������д��ң�������һЩ������ѵ��
        kk = randperm(m);
        
        for l = 1 : numbatches
            % ȡ������˳����batchsize�������Ͷ�Ӧ�ı�ǩ
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            % �ڵ�ǰ������Ȩֵ�����������¼�����������
            net = cnnff(net, batch_x,batch_y,num); % Feedforward
            
            
            %         net = cnnbp(net, y); % Backpropagation
            net = cnnbp(net,batch_y);
            % �õ�����Ȩֵ�ĵ����󣬾�ͨ��Ȩֵ���·���ȥ����Ȩֵ
            net = cnnapplygrads(net, opts,num);
            
            result(end+1) = 0.99*result(end) + 0.01*net.fitness(num);
            
            %ʹSGD��PSO�ֿ���ͼ
            flag_pso=0;
            if(flag_sgd==0)
                flag_sgd=1;
                num_sgd=[num_sgd,cell(1)];
            end
            if(isempty(num_sgd{end}))
                num_sgd{end}=numel(result);
            else
                num_sgd{end}(end+1)=numel(result);
            end
            
            %cnnff�е�fitness����һ����fitnessgbest������result����ʾ�п��ܻ����ϣ��ݶ��½��ڼ䣩
            %ע�⣺ �����result(end+1)�ǵ�i������ݶ�ÿ������ѭ����ǰ��������Ӧ��
            %          net = cnnapplygrads_original(net, opts);
            
            %��Ȼ��һ��fitness��fitnessgbest����ȵģ���Ϊnet�е�Ȩֵ��gbestpar����������£�
            %������һ��net.par{num}���Ǿ�cnnbp��cnnapplygrads���¹�������
            %����ʡȥ���ظ�����cnnff��������ʡ��ʱ��
            if result(end)<net.fitnessgbest
                net.fitnessgbest=result(end);
                net.gbestpar=net.par{num};
            end
        end
        %         if isempty(net.rL)
        %             net.rL(1) = net.L; % ���ۺ���ֵ��Ҳ�������ֵ
        %         end
        %         net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L; % ������ʷ�����ֵ���Ա㻭ͼ����
        SGD_count=SGD_count+1;
    end
    toc;
end
%����������Ӧ�ȵı仯���̣�����������
disp(['����ʵ�鹲����',num2str(opts.numepochs),'�Σ�����PSOռ' ,num2str(PSO_count), '�Σ�SGDռ' ,num2str(SGD_count), '�Ρ�'])
figure('Name','����ʮ�Ĵ�ʵ��');

for p=1:numel(num_pso)
    plot(num_pso{p},result(num_pso{p}),'b');
    hold on;
end
for s=1:numel(num_sgd)
    plot(num_sgd{s},result(num_sgd{s}),'r');
    hold on;
end

%legend('PSO','SGD');
title('��Ӧ������');
xlabel('������Ӧ�ȴ���');ylabel('��Ӧ��');

save fitnessgbest_each_iteration43 result num_pso num_sgd;
end
