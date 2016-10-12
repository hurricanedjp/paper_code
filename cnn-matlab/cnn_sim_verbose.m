function [y, s] = cnn_sim_verbose(net, x)
% CNN_SIM_VERBOSE: Compute CNN output   
%
% SYNTAX
%          [y, s] = cnn_sim_verbose(net, x)  
%
% PARAMETERS
%          net: a CNN network structure
%          x:   input, 3-D array HxWxK
%
%          y:   outputs for network layers
%               y{l} is the output of layer l
%               y{end} is the final network outputs
%          s:   weighted sum (plus bias) for network layers
%               s{l} is the output of layer l
%               y{l} = f(s{l}) where f is activation function
%
% EXAMPLE
%   c = {cnn_cm('full', 1, 4), cnn_cm('1-to-1', 4), ...
%        cnn_cm('1-to-2 2-to-1', 4), cnn_cm('1-to-1', 14), ...
%        cnn_cm('1-to-1', 14), cnn_cm('full', 14, 2)};
%   net = cnn_new([36 32], c, [5 5; 2 2; 3 3; 2 2; 0 0; 0 0], ...
%         repmat({'tansig'}, 1, length(c)), 'rprop');
%   K = 3; x = randn(36, 32, K);
%   [y, s] = cnn_sim_verbose(net,x);
%
% NOTES
% Son Lam Phung, started 11-Jan-2006, revised 4 Novemeber 2006

%% Initialization..........................................................
y = cell(1, net.L); % output
s = cell(1, net.L); % weighted sum
K = size(x, 3);     % number of samples

%% Layer 1: Convolution C1.................................................
layer = 1;
s{layer} = repmat(0, [net.fm_size(layer, :), K, net.no_fms(layer)]);
hrec_size = net.hrec_size(layer,:); % half receptive size

% Compute each feature map
for i = 1:net.no_fms(layer)
    % Convolution
    t = imfilter(x, net.w{layer}(:,:,i), 'same', 'corr');

    % Extract only the meaningful part of matrix
    s{layer}(:,:,:,i) = t(hrec_size(1)+1:end-hrec_size(1), ...
                          hrec_size(2)+1:end-hrec_size(2), :) + ...
                          net.b{layer}(i);
end

% Apply activation function
y{layer} = feval(net.f{layer}, s{layer}); % Apply activation function

%% Layer 2 to L - 2: Pairs of {Subsampling Layer -> Convolution Layer}.....
for layer = 2:net.L-2
    if (mod(layer,2) == 0) 
        % -- Subsampling layer
        s{layer} = repmat(0, [net.fm_size(layer,:), K, net.no_fms(layer)]);

        % Compute each feature map
        for i = 1:net.no_fms(layer)
            % 2x2 block sum -> multiply weight and add bias
            s{layer}(:,:,:,i) = block_sum_2(y{layer-1}(:,:,:,i)) * ...
                                net.w{layer}(i) + net.b{layer}(i);                            
        end
    else
        % -- Convolution layer
        s{layer} = repmat(0, [net.fm_size(layer, :), K, net.no_fms(layer)]);
        hrec_size = net.hrec_size(layer,:); % half receptive size

        % Compute each feature map
        for q = 1:net.no_fms(layer)
            % Find all feature maps in {layer-1} that 
            % go to this feature map in {layer}
            fm_idxs = find(net.c{layer}(:, q))';
            
            % Compute contribution from each feature map in {layer-1}
            for p = fm_idxs
                % Convolution
                t = imfilter(y{layer-1}(:,:,:,p), ...
                             net.w{layer}(:,:,p,q), 'same', 'corr');
            
                % Extract only the meaningful part of matrix          
                s{layer}(:,:,:,q) = s{layer}(:,:,:,q) + ...
                                    t(hrec_size(1)+1:end-hrec_size(1), ...
                                      hrec_size(2)+1:end-hrec_size(2),:);        
            end
            
            % Add with bias term
            s{layer}(:,:,:,q) = s{layer}(:,:,:,q) + net.b{layer}(q);
        end       
    end

    % Apply activation function
    y{layer} = feval(net.f{layer}, s{layer});
end

%% Layer L-1: Last convolution layer.......................................
layer = net.L-1;
s{layer} = repmat(0, [net.fm_size(layer, :), K, net.no_fms(layer)]);

% Compute each feature map
for q = 1:net.no_fms(layer)
    % Find all feature maps in {layer-1} that 
    % go to this feature map in {layer}
    fm_idxs = find(net.c{layer}(:, q))';

    % Compute contribution from each feature map in {layer-1}
    for p = fm_idxs
        % Replicate weight matrix
        w_rep = repmat(net.w{layer}(:,:,p,q),[1 1 K]);
        
        % Compute weighted sum
        t = sum(sum(y{layer-1}(:,:,:,p) .* w_rep,1),2);
        
        % Add to s
        s{layer}(:,:,:,q) = s{layer}(:,:,:,q) + t;        
    end
    
   % Add with bias term
    s{layer}(:,:,:,q) = s{layer}(:,:,:,q) + net.b{layer}(q);
end       

% Apply activation function
y{layer} = feval(net.f{layer}, s{layer});

%% Layer L: output perceptron layer........................................
layer = net.L;

% Re-arrange previous output
yt = squeeze(y{layer-1}); % yt has size K x net.no_fms{layer-1}             

% Compute weighted sum and bias
s{layer} = (yt * net.w{layer})' + repmat(net.b{layer}, [1 K]);

% Apply activation function
y{layer} = feval(net.f{layer}, s{layer});