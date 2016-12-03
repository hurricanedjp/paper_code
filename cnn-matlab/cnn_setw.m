function new_net = cnn_setw(net, w)
% CNN_SETW: Set trainable parameters of CNN  
%
% SYNTAX
%      new_net = cnn_setw(net, w);
%
% PARAMETERS
%      net: CNN structure
%      w:  column vector containing new values for all trainable parameters
%
% EXAMPLE
%
%
% NOTES
% Son Lam Phung, started 12-Jan-2006.

new_net = net;
[w1, b1] = cnn_devectorize_wb(net, w);
new_net.w = w1;
new_net.b = b1;