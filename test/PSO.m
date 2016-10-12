%% �ô���Ϊ����PSO��BP�����Ԥ��
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>
%% ��ջ���
clc
clear

%��ȡ����
load mnist_uint8.mat;

%�ڵ����
inputnum=784;
hiddennum=100;
outputnum=10;

%ͼ���ֵ��
train_x(train_x>=128)=1;
train_x(train_x<128)=0;
test_x(test_x>=128)=1;
test_x(test_x<128)=0;

%ѵ�����ݺ�Ԥ������
inputn=train_x;
input_test=test_x;
outputn=train_y;
output_test=test_y;

% %ѡ����������������ݹ�һ��
% [inputn,inputps]=mapminmax(input_train);
% [outputn,outputps]=mapminmax(output_train);

%��������
net=newff(inputn,outputn,hiddennum);

% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;
c2 = 1.49445;

maxgen=100;   % ��������������������
sizepop=30;   %��Ⱥ��ģ

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

for i=1:sizepop                 %30������
    pop(i,:)=5*rands(1,21);     %�����ʼ��Ȩֵ��ֵ��λ�ã�
    V(i,:)=rands(1,21);         %�����ʼ���ٶ�
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,inputn,outputn);
end


% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�ֵ�һ�����λ��
gbest=pop;    %�����һ�����λ��
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for i=1:maxgen   %��������
    i;
    
    for j=1:sizepop
        
        %�ٶȸ���
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %��Ⱥ����
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;  %λ�ó������ֵ���³����ֵ
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %����Ӧ����
        pos=unidrnd(21);
        if rand>0.95
            pop(j,pos)=5*rands(1,1);
        end
        
        %��Ӧ��ֵ
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,inputn,outputn);  %�������������ɰ����ݶ�����������ֻ��ǰ�򣿣���
    end
    
    %% ÿ�μ������������ӲŽ�����Ӧ�ȸ���
    for j=1:sizepop
        %�������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        %Ⱥ�����Ÿ���
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
        
    end
    
    yy(i)=fitnesszbest;
    
end %����100��

%% �������
plot(yy)
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');

x=zbest;
%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
net.trainParam.epochs=100; %100�ε�������
net.trainParam.lr=0.1;     %ѧϰ����
%net.trainParam.goal=0.00001;

%����ѵ��
[net,per2]=train(net,inputn,outputn);   %������train�У�ʹ�����ݶȣ�����

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps); %���һ�����an����ԭ��
error=test_simu-output_test;

web browser www.matlabsky.com
web browser http://www.matlabsky.com/thread-11198-1-1.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>