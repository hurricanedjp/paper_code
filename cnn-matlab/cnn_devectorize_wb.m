function [net_w, net_b] = cnn_devectorize_wb(net, w)
% CNN_DEVECTORIZE_WB Extract w and b of CNN from a vector of parameters
%
% SYNTAX
%        [net_w, net_b] = cnn_devectorize_wb(net, w)
%
% PARAMETERS
%        net: CNN structure
%        w:   vector containing all trainable parameters
%   
% EXAMPLE
%
%
% NOTES
% Son Lam Phung, started 12-Jan-2006.

net_w = net.w;
net_b = net.b;

idx_end = 0;

%% Layer 1: convolution layer C1...........................................
layer = 1;
idx_start = idx_end + 1;
idx_end = idx_start + numel(net_w{layer}) - 1;
net_w{layer} = reshape(w(idx_start:idx_end), size(net_w{layer}));

idx_start = idx_end + 1;
idx_end = idx_start + numel(net_b{layer}) - 1;
net_b{layer} = reshape(w(idx_start:idx_end), size(net_b{layer}));

%% Layer 2 to L - 1: pairs of {subsampling layer -> convolution layer}.....       
for layer = 2:net.L-1
    if (mod(layer,2) == 0) 
        % -- Subsampling layer
        idx_start = idx_end + 1;
        idx_end = idx_start + numel(net_w{layer}) - 1;
        net_w{layer} = reshape(w(idx_start:idx_end), size(net_w{layer}));

        idx_start = idx_end + 1;
        idx_end = idx_start + numel(net_b{layer}) - 1;
        net_b{layer} = reshape(w(idx_start:idx_end), size(net_b{layer}));
    else
        % -- Convolution layer
        for p = 1:net.no_fms(layer-1)
            for q = 1:net.no_fms(layer)
                if (net.c{layer}(p,q) == true)
                    idx_start = idx_end + 1;
                    idx_end = idx_start + numel(net_w{layer}(:,:,p,q)) - 1;
                    net_w{layer}(:,:,p,q) = ...
                        reshape(w(idx_start:idx_end), ...
                                size(net_w{layer}(:,:,p,q)));
                end
            end
        end
        
        idx_start = idx_end + 1;
        idx_end = idx_start + numel(net_b{layer}) - 1;
        net_b{layer} = reshape(w(idx_start:idx_end), size(net_b{layer}));
    end
end

%% Layer L: output perceptron layer........................................
layer = net.L;
idx_start = idx_end + 1;
idx_end = idx_start + numel(net_w{layer}) - 1;
net_w{layer} = reshape(w(idx_start:idx_end), size(net_w{layer}));

idx_start = idx_end + 1;
idx_end = idx_start + numel(net_b{layer}) - 1;
net_b{layer} = reshape(w(idx_start:idx_end), size(net_b{layer}));