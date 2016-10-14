function net = cnn_new(input_size, c, rec_size, tf_fcn, train_method)
% cnn_new: Create a new convolutional neural network 
%          C1 -> S2 -> C3 -> S4 -> C5 -> F  
%          C1 -> S2 -> C3 -> S4 -> ... -> C2a+1 -> F2a+2       
%
% SYNTAX
%   net = cnn_new(input_size, c, rec_size, tf_fcn, train_method)
%
% PARAMETERS
%   input_size: size of input image, e.g. [32 32]
%
%   c:          list of connection matrices, cell array
%               c{1} is connection matrix from input to C1
%               c{2} is connection matrix from C1 to S2, and so on
%               c{l}(i,j) = true if feature map i of layer l-1 
%                           is connected to feature map j of layer l  
%
%   rec_size:   size of receptive fields, 2D array
%               rec_size(l,:) is the size of receptive field for layer l
%      
%   tf_fcn:     transfer function of network layers, cell array of strings
%               tf_fcn{i} is the transfer function of layer Li
%
%   train_method: training method for network, string
%               'gd', 'rprop' 
%
% EXAMPLES
%   c = {cnn_cm('full', 1, 2), cnn_cm('1-to-1', 2), ...
%        cnn_cm('1-to-2', 2), cnn_cm('1-to-1', 4), ...
%        cnn_cm('1-to-2', 4), cnn_cm('full', 8, 2)}
%   net = cnn_new([32 32], c, [5 2 5 2 0 0; 5 2 5 2 0 0]', ...
%         repmat({'tansig'}, 1, length(c)), 'rprop');
%
%   c = {cnn_cm('full', 1, 6), cnn_cm('1-to-1', 6), ...
%        cnn_cm('toeplitz', 6, 16, 3), cnn_cm('1-to-1', 16), ...
%        cnn_cm('full', 16, 120), cnn_cm('full', 120, 2)}
%   net = cnn_new([32 32], c, [5 2 5 2 0 0; 5 2 5 2 0 0]', ...
%         repmat({'tansig'}, 1, length(c)), 'rprop');
%
%   c = {cnn_cm('full', 1, 6), cnn_cm('1-to-1', 6), ...
%        cnn_cm('toeplitz', 6, 16, 3), cnn_cm('1-to-1', 16), ...
%        cnn_cm('full', 16, 10), cnn_cm('full', 10, 1)}
%   net = cnn_new([32 32], c, [5 2 5 2 0 0; 5 2 5 2 0 0]', ...
%         repmat({'tansig'}, 1, length(c)), 'rprop');
%
% NOTES
%   See also cnn_cm.m
%
% Son Lam Phung, started 11-Jan-2006, revised 01-Nov-2006.

%% Default parameters......................................................
if nargin < 1
    input_size = [32 32];
end

if nargin < 2
    c = {cnn_cm('full', 1, 6), cnn_cm('1-to-1', 6), ...
        cnn_cm('toeplitz', 6, 16, 3), cnn_cm('1-to-1', 16), ...
        cnn_cm('full', 16, 120), cnn_cm('full', 120, 1)};
end

if nargin < 3
    rec_size = [5 2 5 2 0 0; 5 2 5 2 0 0]';
end

if nargin < 4
    tf_fcn = repmat({'ltanh'}, 1, length(c));
end

if nargin < 5
    train_method = 'rprop';
end

%% ........................... Create a CNN ............................ %%
net.L = length(c);                     % number of layers
net.w = cell(1,net.L);                 % weights  
net.b = cell(1,net.L);                 % biases
net.c = c;                             % connection matrices
net.rec_size = rec_size;               % size of receptive fields 
net.hrec_size = floor(net.rec_size/2); % half the size of receptive fields
net.f = tf_fcn;                        % transfer function
net.input_size = input_size;           % size of input images
net.no_fms = zeros(1,net.L);        % number of feature maps in each layer 
net.fm_size = zeros(net.L, 2);      % size of feature map  
net.layers = cell(1, net.L);        % network layers

%% Layer 1: convolution layer..............................................
no_params = 0;                % total number of free parameters in networks
layer = 1;
net.layers{layer}.type = 'C';
net.layers{layer}.name = [net.layers{layer}.type int2str(layer)];
net.layers{layer}.connection = 'full';
net.no_fms(layer) = size(c{layer},2);
net.fm_size(layer, :) = input_size - 2 * net.hrec_size(layer, :);

net.b{layer} = randn(net.no_fms(layer),1);
net.w{layer} = randn([net.rec_size(layer,:) net.no_fms(layer)]);
no_params = no_params + numel(net.w{layer}) + numel(net.b{layer});

%% Layer 2 to L - 1: pairs of {sumsampling layer -> convolution layer}.....
for layer = 2:net.L-1
    net.no_fms(layer) = size(c{layer},2);
    if (mod(layer,2) == 0) 
        % -- subsampling layer S
        net.layers{layer}.type = 'S';
        net.layers{layer}.name = [net.layers{layer}.type int2str(layer)];
        net.layers{layer}.connection = '1-to-1';
        net.fm_size(layer, :) = floor(net.fm_size(layer-1,:)/2);
        
        net.w{layer} = randn(net.no_fms(layer),1); % weights
        net.b{layer} = randn(net.no_fms(layer),1); % biases

        no_params = no_params + numel(net.w{layer}) + numel(net.b{layer});
    else
        % -- convolution layer C
        net.layers{layer}.type = 'C';
        net.layers{layer}.name = [net.layers{layer}.type int2str(layer)];
        net.layers{layer}.connection = 'custom';
               
        if (layer == net.L - 1)
            % receptive field of last convolution layer =
            % feature map of previous layer
            net.rec_size(layer, :) = net.fm_size(layer-1,:);            
        end

        net.fm_size(layer, :) = net.fm_size(layer-1,:) ...
                                - net.rec_size(layer, :) + 1;

        net.b{layer} = randn(net.no_fms(layer),1);       % biases
        % update number of parameters
        no_params = no_params + numel(net.b{layer});     

        net.w{layer} = zeros(net.rec_size(layer,1), ...  % weights
                             net.rec_size(layer,2), ...
                             net.no_fms(layer-1), ...
                             net.no_fms(layer));
        % initialize weights                         
        for p = 1:net.no_fms(layer-1)
            for q = 1:net.no_fms(layer)
                if (net.c{layer}(p, q) == true)
                    net.w{layer}(:,:, p, q) = randn(net.rec_size(layer,:));
                     % update number of parameters
                    no_params   = no_params + prod(net.rec_size(layer,:));
                end
            end
        end
    end
end

%% Layer L: output perceptron layer........................................
layer = net.L;
net.layers{layer}.type = 'F';
net.layers{layer}.name = [net.layers{layer}.type int2str(layer)];
net.layers{layer}.connection = 'full';
net.no_fms(layer) = size(c{layer},2);
net.w{layer} = randn(net.no_fms(layer-1), net.no_fms(layer));
net.b{layer} = randn(net.no_fms(layer),1);
no_params   = no_params + numel(net.w{layer}) + numel(net.b{layer});
net.P = no_params;

%% .................... Network Training Parameters .................... %%
net.train.epochs = 20;               % epoch limit
net.train.goal   = 0.0;              % target mse
net.train.show   = 1;                % epochs before next show of progress 
net.train.method = train_method;     % 'gd', 'rprop'

% gradient descent
net.train.gd.lr = 0.15;              % learning rate

% rprop
net.train.rprop.etap = 1.01;         % increasing multiplier
net.train.rprop.etam = 0.99;         % decreasing multiplier
net.train.rprop.delta_init = 0.01;   % intial learning rate
net.train.rprop.delta_max = 10.0;    % maximum learning rate