% DESCRIPTION Testing cnn_new.m, cnn_sim.m, and cnn_sim_verbose.m
%                     cnn_train 
%
% NOTES
% Son Lam Phung, started 04-Nov-2006.

%% Network creation
H = 32; % height of 2-D input
W = 32; % width of 2-D input
% Create connection matrix
% This matrix specifies how feature maps from one layer
% are connected to feature maps in the next layer
c = {cnn_cm('full', 1, 4), cnn_cm('1-to-1', 4), cnn_cm('1-to-2 2-to-1', 4), cnn_cm('1-to-1', 14), cnn_cm('1-to-1', 14), cnn_cm('full', 14, 2)}
% Creat CNN
net = cnn_new([H W], c, [5 5; 2 2; 3 3; 2 2; 0 0; 0 0], repmat({'tansig'}, 1, length(c)), 'rprop');

%% Network simulation
K = 2000;                                % Number of samples 
x = randn(H, W, K);                      % Network input
[y, s] = cnn_sim_verbose(net,x); y{end}  % Method 1: Network output & layer output
y = cnn_sim(net,x);                      % Method 2: Network ouput 

%% Network training
t = rand(2, K);                          % Network target output
[new_net, tr] = cnn_train(net, x, t);    % new_net is trained network
                                         % tr is training record

%% Plot training performance
plot(tr.epoch, tr.mse);
xlabel('Epoch')
ylabel('MSE')