% DESCRIPTION Script to check function cnn_compute_gradient.m 
%
%
% NOTES
% Son Lam Phung, started 05-Nov-2006.

%% Creat network
clear; clc
H = 36;
W = 32;
c = {cnn_cm('full', 1, 4), cnn_cm('1-to-1', 4), ...
     cnn_cm('1-to-2 2-to-1', 4), cnn_cm('1-to-1', 14), ...
     cnn_cm('1-to-1', 14), cnn_cm('full', 14, 2)};
net = cnn_new([H W], c, [5 5; 2 2; 3 3; 2 2; 0 0; 0 0], ...
    repmat({'tansig'}, 1, length(c)), 'rprop');
fprintf('Number of network parameters %g.\n', net.P);

%% Compute gradient
K = 5; x = randn(36, 32, K); d = randn(2,K);
w = cnn_getw(net);
[y, s] = cnn_sim_verbose(net, x);
e = y{end} - d;
E = mse(e);
[dw, db] = cnn_compute_gradient(net, x, e, y, s); % Compute gradient
dE = cnn_vectorize_wb(net, dw, db);               % Vectorize gradient 

%% Check computed gradient
ratio = zeros(size(dE));
clc
for i = 1:length(w)
    st = 0.000001;
    % Modify only one weight by a small amount
    w_new = w; w_new(i) = w_new(i) + st; 
    
    % Compute output by new net
    net_new = cnn_setw(net, w_new);
    y_new = cnn_sim(net_new, x);
    e_new = y_new - d;
    E_new = mse(e_new);        % MSE of new net
    ratio(i) = (E_new - E)/st; % Limit deltaE/deltaw = gradient
    fprintf(['Weight %g: theoretical gradient = %2.6f, ' ...
             'actual gradient = %2.6f.\n'],i, dE(i), ratio(i));
end
%% Display weights that have error in gradient computation
dev = abs(dE - ratio); % Difference between theoretical and actual gradient
idxs = find(dev > 0.00001)'
length(idxs)