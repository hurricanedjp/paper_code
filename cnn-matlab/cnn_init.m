function new_net = cnn_init(net)
% CNN_INIT Initialise trainable parameters in CNN
%
% SYNTAX
%   new_net = cnn_init(net);
%
% PARAMETERS
%   net:   CNN structure
%   new_net:  new CNN structure
%
% EXAMPLE
%   new_net = cnn_init(net);
%
% NOTES
% Son Lam Phung, started 22-Jan-2006.

new_net = net;

%% Layer 1: convolution layer..............................................
layer = 1;
new_net.b{layer} = randn(size(new_net.b{layer}));
new_net.w{layer} = randn(size(new_net.w{layer}));

%% Layer 2 to L - 1: pairs of {sumsampling layer -> convolution layer}.....
for layer = 2:net.L-1
    new_net.b{layer} = randn(size(new_net.b{layer}));
    
    if (mod(layer,2) == 0) 
        % -- Subsampling layer
        new_net.w{layer} = randn(size(new_net.w{layer}));
        
    else
        % -- Convolution layer
        for p = 1:new_net.no_fms(layer-1)
            for q = 1:new_net.no_fms(layer)
                if (new_net.c{layer}(p, q) == true)
                    new_net.w{layer}(:,:, p, q) = ...
                        randn(new_net.rec_size(layer,:));
                end
            end
        end
    end
end

%% Layer L: output perceptron layer........................................
layer = net.L;
new_net.b{layer} = randn(size(new_net.b{layer}));
new_net.w{layer} = randn(size(new_net.w{layer}));