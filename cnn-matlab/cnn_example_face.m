% DESCRIPTION 
%   Example of using the CNN MATLAB library
%   for face versus non-face classification
% NOTES
%   This example is quite memory intensive.
%   It has been tested on a PC with 4GB RAM
%   Son Lam Phung, started 15-Apr-2009.

train_x = double(reshape(train_x',48,48,596))/255;
test_x = double(reshape(test_x',48,48,501))/255;
train_y = double(train_y');
test_y = double(test_y');
x=train_x;
d=train_y;
x_test=test_x;
d_test=test_y;


%% Load training data
% load('data\train_data.mat')
whos

%% Create a CNN
H = 48;    % height of 2-D input
W = 48;    % width of 2-D input

% Create connection matrices
cm1 = cnn_cm('full', 1, 2);  % input to layer C1
                             % C1 has 2 planes
cm2 = cnn_cm('1-to-1', 2);   % C1 to S2
cm3 = [1 1 0 0 1; 0 0 1 1 1];% S2 to layer C3
cm4 = cnn_cm('1-to-1', 5);   % C3 to S4
cm5 = cnn_cm('1-to-1', 5);   % S4 to C5
cm6 = cnn_cm('full',5,1);    % C5 to F6
c = {cm1, cm2, cm3, cm4, cm5, cm6};

% Receptive sizes for each layer
rec_size = [5 5;   % C1
            2 2;   % S2
            3 3;   % C3
            2 2;   % S4
            0 0;   % C5 auto calculated
            0 0];  % F6 auto calculated
        
% Transfer function
tf_fcn = {'tansig',  % layer C1 
          'purelin', % layer S2 
          'tansig',  % layer C3 
          'purelin', % layer S4 
          'tansig',  % layer C5 
          'tansig'}  % layer F6 output

% Training method
train_method = 'rprop';        % 'gd'

% Create CNN
net = cnn_new([H W], c, rec_size, tf_fcn, train_method);

%% Network training
net.train.epochs = 1100;
[new_net, tr] = cnn_train(net, x, d); 
% new_net is trained network, tr is training record
save('data\trained_net.mat', 'new_net', 'net', 'tr');

%% Plotting training performance
plot(tr.epoch, tr.mse, 'b-', 'LineWidth', 2); grid
h = xlabel('Epochs'), set(h, 'FontSize', 14);
h = ylabel('Training MSE'), set(h, 'FontSize', 14);
set(gca, 'FontSize', 14);

y = cnn_sim(new_net, x); % network output
cr = sum((y >0) == (d >=0))/length(d)*100;
fprintf('Classification rate (train): cr = %2.2f%%\n',cr);

%% Network testing
% load('data\test_data.mat')
whos 
y_test = cnn_sim(new_net,x_test); % network output
cr_test = sum((y_test >0)==(d_test>=0))/length(d_test)*100;
fprintf('Classification rate (test): cr = %2.2f%%\n',cr_test);