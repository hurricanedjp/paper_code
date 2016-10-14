function [new_net, new_tr] = cnn_train(net, x, d, tr)
% CNN_TRAIN Train a CNN
%
% SYNTAX
%     [new_net, new_tr] = cnn_train(net, x, d, tr);
%
% PARAMETERS
%     net: CNN structure
%     x:   inputs         (3-D array H x W x K)
%     d:   desired output (2-D array NL x K)
%     tr:  existing training records
%
%     new_tr:  updated training records
%     new_net: trained net
%
% EXAMPLE
%     c = {cnn_cm('full', 1, 4), cnn_cm('1-to-1', 4), ...
%          cnn_cm('1-to-2 2-to-1', 4), cnn_cm('1-to-1', 14), ...
%          cnn_cm('1-to-1', 14), cnn_cm('full', 14, 2)};
%     net = cnn_new([36 32], c, [5 5; 2 2; 3 3; 2 2; 0 0; 0 0], ...
%           repmat({'tansig'}, 1, length(c)), 'rprop');
%     K = 5; x = randn(36, 32, K); d = randn(2,K);
%     [new_net, new_tr] = cnn_train(net, x, d);
%
% NOTES
% Son Lam Phung, started 12-Jan-2006.

% Call the appropriate cnn_train_... function according to
% the specified training method
if nargin < 4
    [new_net, new_tr] = ...
                   feval(['cnn_train_' net.train.method], net, x, d);
else
    [new_net, new_tr] = ...
                   feval(['cnn_train_' net.train.method], net, x, d, tr);
end