function net = cnntrain_seq(net, x, y, opts ,sizepar,w,c1,c2)
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
for i = 1 : opts.numepochs
    % disp(X) ��ӡ����Ԫ�ء����X�Ǹ��ַ������Ǿʹ�ӡ����ַ���
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    % tic �� toc ��������ʱ�ģ��������������֮�����ĵ�ʱ��
    tic;
    


        %���Ӹ���
        for num=1:sizepar
            %��ֵ��������cnnff�ֿ�
            net = cnnassign(net,num);
            % �ڵ�ǰ������Ȩֵ�����������¼�����������
            net = cnnff(net,x,y,num); % Feedforward
            
        end
        %��Ӧ�ȸ���,���弫ֵ��ȫ�弫ֵ���£��ٶȺ�λ�ø���
        net = cnnupdate(net,sizepar,w,c1,c2);
        
        %��ÿ�ε�������Ӧ��ֵ����result
        if isempty(result)
            result=net.fitnessgbest;
        else
            result(end+1)=net.fitnessgbest;
        end

 
        
        % �õ���������������ͨ����Ӧ��������ǩ��bp�㷨���õ���������Ȩֵ
        % (Ҳ������Щ����˵�Ԫ�أ��ĵ���
        % ����������һ��ѵ������Ȩֵ�ŵ�����ʮһ��������,���ֳ���ļ�����
        num=sizepar+1;
        
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
            
            %cnnff�е�fitness����һ����fitnessgbest������result����ʾ�п��ܻ����ϣ��ݶ��½��ڼ䣩
            %ע�⣺ �����result(end+1)�ǵ�i������ݶ�ÿ������ѭ����ǰ��������Ӧ��
            %          net = cnnapplygrads_original(net, opts);
            
            %��Ȼ��һ��fitness��fitnessgbest����ȵģ���Ϊnet�е�Ȩֵ��gbestpar����������£�
            %������һ��net.par{num}���Ǿ�cnnbp��cnnapplygrads���¹�������
            %����ʡȥ���ظ�����cnnff��������ʡ��ʱ��
            if net.fitness(num)<net.fitnessgbest
                net.fitnessgbest=net.fitness(num);
                net.gbestpar=net.par{num};
            end
        end
        %         if isempty(net.rL)
        %             net.rL(1) = net.L; % ���ۺ���ֵ��Ҳ�������ֵ
        %         end
        %         net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L; % ������ʷ�����ֵ���Ա㻭ͼ����


    toc;
end
%����������Ӧ�ȵı仯���̣�����������
disp('����ʵ��PSOռ' ,num2str(PSO_count), '�Ρ�')
disp('����ʵ��SGDռ' ,num2str(SGD_count), '�Ρ�')
plot(result);
title('��Ӧ������');
xlabel('������Ӧ�ȴ���');ylabel('��Ӧ��');

save fitnessgbest_each_iteration40 result;
end
