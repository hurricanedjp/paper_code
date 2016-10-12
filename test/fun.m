function error = fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)
%�ú�������������Ӧ��ֵ
%x          input     ����
%inputnum   input     �����ڵ���
%outputnum  input     ������ڵ���
%net        input     ����
%inputn     input     ѵ����������
%outputn    input     ѵ���������

%error      output    ������Ӧ��ֵ

%��ȡ
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);


%�����������
net.trainParam.epochs=20;        %ѵ����������������Ϊ20��
net.trainParam.lr=0.1;           %ѧϰ���ʣ���������Ϊ0.1
net.trainParam.goal=0.00001;     %ѵ��Ŀ����С����������=0.00001
net.trainParam.show=100;         %��ʾƵ�ʣ���������Ϊûѵ��100����ʾһ��
net.trainParam.showWindow=0;
 
%����Ȩֵ��ֵ
net.iw{1,1}=reshape(w1,hiddennum,inputnum);     %����㵽�����ȨֵԪ������
net.lw{2,1}=reshape(w2,outputnum,hiddennum);    %���㵽��һ���ȨֵԪ���������ǵ�������Ϊ���㵽������ȨֵԪ������
net.b{1}=reshape(B1,hiddennum,1);               %������Ԫ��ֵ��ƫ��ֵ��
net.b{2}=B2;                                    %�������Ԫ����ֵ

%����ѵ��
net=train(net,inputn,outputn);    %           ��ѵ���õ�Ȩֵ��ʹ���ݶȻ��ǣ�

an=sim(net,inputn);         %sim����������simulinkģ�ͣ���ѵ���õ�����Ľ������㣬an�ǵõ����������

error=sum(abs(an-outputn));