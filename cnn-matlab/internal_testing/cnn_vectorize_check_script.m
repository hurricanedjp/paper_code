%%
clear all; clc
c = {cnn_cm('full', 1, 4), cnn_cm('1-to-1', 4), ...
     cnn_cm('1-to-2 2-to-1', 4), cnn_cm('1-to-1', 14), ...
     cnn_cm('1-to-1', 14), cnn_cm('full', 14, 1)};
net = cnn_new([36 32], c, [5 5; 2 2; 3 3; 2 2; 0 0; 0 0], ...
      repmat({'tansig'}, 1, length(c)), 'rprop');
K = 300; x = randn(36, 32, K); d = randn(1,K);
%%
w = cnn_vectorize_wb(net, net.w, net.b);
[w1, b1] = cnn_devectorize_wb(net, w);
net1 = net; net1.w = w1; net1.b = b1;
%%
y = cnn_sim(net, x);
y1 = cnn_sim(net1, x);
sumn(y ~= y1)