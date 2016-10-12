function w = cnn_vectorize_wb(net, net_w, net_b)
% CNN_VECTORIZE_WB: Vectorize w and b of CNN
%
% SYNTAX
%        w = cnn_vectorize_wb(net, net_w, net_b);
%
% PARAMETERS
%        net:   CNN structure
%        net_w: cell array of CNN weights
%        net_b: cell array of CNN biases
%
% EXAMPLE
%
%
% NOTES
% Son Lam Phung, started 12-Jan-2006.

w = zeros(net.P, 1);
idx_end = 0;

%% Layer 1: convolution layer C1...........................................
layer = 1;
idx_start = idx_end + 1;
idx_end = idx_start + numel(net_w{layer}) - 1;
w(idx_start:idx_end) = net_w{layer}(:);

idx_start = idx_end + 1;
idx_end = idx_start + numel(net_b{layer}) - 1;
w(idx_start:idx_end) = net_b{layer}(:);

%% Layer 2 to L - 1: pairs of {sumsampling layer -> convolution layer}.....    
for layer = 2:net.L-1
    if (mod(layer,2) == 0) 
        % -- Subsampling layer
        idx_start = idx_end + 1;
        idx_end = idx_start + numel(net_w{layer}) - 1;
        w(idx_start:idx_end) = net_w{layer}(:);

        idx_start = idx_end + 1;
        idx_end = idx_start + numel(net_b{layer}) - 1;
        w(idx_start:idx_end) = net_b{layer}(:);
    else
        % -- Convolution layer
        for p = 1:net.no_fms(layer-1)
            for q = 1:net.no_fms(layer)
                if (net.c{layer}(p,q) == true)
                    w_tmp = squeeze(net_w{layer}(:,:,p,q));
                    idx_start = idx_end + 1;
                    idx_end = idx_start + numel(w_tmp) - 1;
                    w(idx_start:idx_end) = w_tmp(:);
                end
            end
        end
        idx_start = idx_end + 1;
        idx_end = idx_start + numel(net_b{layer}) - 1;
        w(idx_start:idx_end) = net_b{layer}(:);
    end
end

%% Layer L: output perceptron layer........................................
layer = net.L;
idx_start = idx_end + 1;
idx_end = idx_start + numel(net_w{layer}) - 1;
w(idx_start:idx_end) = net_w{layer}(:);

idx_start = idx_end + 1;
idx_end = idx_start + numel(net_b{layer}) - 1;
w(idx_start:idx_end) = net_b{layer}(:);