function error = fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)
%该函数用来计算适应度值
%x          input     个体
%inputnum   input     输入层节点数
%outputnum  input     隐含层节点数
%net        input     网络
%inputn     input     训练输入数据
%outputn    input     训练输出数据

%error      output    个体适应度值

%提取
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);


%网络进化参数
net.trainParam.epochs=20;        %训练次数，这里设置为20次
net.trainParam.lr=0.1;           %学习速率，这里设置为0.1
net.trainParam.goal=0.00001;     %训练目标最小误差，这里设置=0.00001
net.trainParam.show=100;         %显示频率，这里设置为没训练100次显示一次
net.trainParam.showWindow=0;
 
%网络权值赋值
net.iw{1,1}=reshape(w1,hiddennum,inputnum);     %输入层到隐层的权值元包矩阵
net.lw{2,1}=reshape(w2,outputnum,hiddennum);    %隐层到后一层的权值元包矩阵，若是单隐层则为隐层到输出层的权值元包矩阵
net.b{1}=reshape(B1,hiddennum,1);               %隐层神经元阈值（偏置值）
net.b{2}=B2;                                    %输出层神经元的阈值

%网络训练
net=train(net,inputn,outputn);    %           ？训练好的权值是使用梯度还是？

an=sim(net,inputn);         %sim函数（运行simulink模型）用训练好的网络的进行运算，an是得到的输出层结果

error=sum(abs(an-outputn));