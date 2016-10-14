function [dw, db] = cnn_compute_gradient(net, x, e, y, s)
% CNN_COMPUTE_GRADIENT Compute gradient of MSE function for CNN
%
% SYNTAX
%        [dw, db] = cnn_compute_gradient(net, x, e, y, s); 
%
% PARAMETERS
%        net: CNN structure
%        x:   network input, 3-D array HxWxK  
%        e:   network error, 2-D array N{L}xK
%        y:   outputs of network layers
%        s:   weight sums of network layers
%
%       dw:   partial derivatives w.r.t. weights
%       db:   partial derivatives w.r.t. biases
%
% EXAMPLE
%
%
% NOTES
% Son Lam Phung, started 12-Jan-2006, last revised 04-Nov-2006
% Version 1, 12-Jan-2006
%            fixed structure C1->S2->C3->S4->C5->F6
% Version 2, 04-Nov-2006
%            generalized structure: any number of layers

%% ................ Stage 1: Compute Error Sensitivity ................. %%
K = size(x, 3);       % Number of input samples
es = cell(1, net.L);  % Allocate storage for error sensitivity

%% Layer L: output perceptron layer........................................
layer = net.L;
N = net.no_fms(layer);
es{layer} = 2/(K*N) * e .* feval(['d' net.f{layer}], s{layer}, y{layer});

%% Layer L-1: last convolutional layer.....................................
layer = net.L - 1;

dy = feval(['d' net.f{layer}], s{layer}, y{layer}); % f'(s)

es{layer} = reshape((net.w{layer+1} * es{layer+1})', ...
                    [1 1 K net.no_fms(layer)]);     % Back-propagate
                
es{layer} = es{layer} .* dy;                        % Multiply f'(s)

%% Layer L - 2: last subsampling layer.....................................
% This layer is special because convolution mask of layer L-1
% has the same size as feature map of layer L-2
layer = net.L - 2;
S1 = size(s{layer},1);
S2 = size(s{layer},2);
es{layer} = repmat(0, size(s{layer}));
dy = feval(['d' net.f{layer}], s{layer}, y{layer}); % f'(s)

% Replicate matrix es{layer+1}(n,n) into a S1 x S2 x * x *
es_rep = repmat(es{layer+1}, [S1 S2 1 1]);

% Compute for each feature map
for n = 1:net.no_fms(layer)
    % Back-propagate
    es{layer}(:,:,:,n) = es_rep(:,:,:,n) .* ...
                         repmat(net.w{layer+1}(:,:,n,n), [1 1 K 1]);
end

es{layer} = es{layer} .* dy; % Multiply f'(s)

%% Layers L-3 to 1: pairs of {convolution layer -> subsampling layer}......
for layer = (net.L-3):-1:1
    dy = feval(['d' net.f{layer}], s{layer}, y{layer}); 
    if (mod(layer, 2) == 1)
        %-- Convolution layer
        size_s = size(s{layer});
        es{layer} = repmat(0, size_s);

        % Enlarge es{layer+1} by a factor of 2 
        % in first and second dimension
        size_es = size(es{layer+1});
        es_rep = zeros([size_s(1:2) size_es(3:4)]);
        for i = 1:2
            for j = 1:2
                es_rep(i:2:end,j:2:end,:,:) = es{layer+1};
            end
        end

        % Back-propagate
        for n = 1:net.no_fms(layer)
            es{layer}(:,:,:,n) = es_rep(:,:,:,n) * net.w{layer+1}(n);
        end
    else
        %-- Subsampling layer        
        size_s = size(s{layer});
        es{layer} = repmat(0, size_s);

        for n = 1:net.no_fms(layer)
            % Find all feature maps in {layer+1}
            % that go from feature map n
            fm_idxs = find(net.c{layer+1}(n, :));

            % Adding up contribution from feature maps in {layer+1}
            for m = fm_idxs
                % Back-propagate
                es{layer}(:,:,:,n) = ...
                    es{layer}(:,:,:,n) + ...
                    imfilter(es{layer+1}(:,:,:,m), ...
                             rot90(net.w{layer+1}(:,:,n,m),2), ...  
                             'full', 'corr');
            end
        end
    end
    es{layer} = es{layer} .* dy; % Multiply f'(s)    
end

%% ..................... Stage 2: Compute Gradient ..................... %%
% Allocate memory
dw = cell(1, net.L); % Weights
db = cell(1, net.L); % Biases

%% Layer L: output perceptron layer........................................
layer = net.L;
dw{layer} = (es{layer} * squeeze(y{layer-1}))';
db{layer} = sum(es{layer},2);

%% Layer L-1: last convolutional layer.....................................
layer = net.L - 1;
size_y = size(y{layer-1});

% Replicate es{layer} to size_y(1) x size_y(2) x * x *
es_rep = repmat(es{layer}, [size_y(1) size_y(2) 1 1]); 
tmp = sum(es_rep .* y{layer-1},3);

% Weights
for n = net.no_fms(layer-1):-1:1
    dw{layer}(:,:,n,n) = tmp(:,:,:,n);
end

% Biases
db{layer} = sum(squeeze(es{layer}), 1)';

%% Layer L - 2: last subsampling layer.....................................
layer = net.L - 2;
es_y = block_sum_2(y{layer-1}) .* es{layer};
dw{layer} = squeeze(sum(sum(sum(es_y,3),2),1));
db{layer} = squeeze(sum(sum(sum(es{layer},3),2),1));

%% Layers L-3 to 2: pairs of {convolution layer -> subsampling layer}......
for layer = (net.L-3):-1:2
    if (mod(layer, 2) == 1)
    %-- Convolution layer
        size_es = size(es{layer});
        size_w = size(net.w{layer});
        hrec_size = net.hrec_size(layer);
        dw{layer} = repmat(0,size_w);

        for p = 1:net.no_fms(layer-1)
            % Find all feature maps in {layer}
            % that go from feature map p {layer-1}
            fm_idxs = find(net.c{layer}(p,:));
            
            for m = 1:size_es(1)
                for n = 1:size_es(2)          
                    % Repeat es{layer}(m,n,:,fm_idxs) 
                    % into size_w(1) x size_w(2) x * x *     
                    es_rep = zeros([size_w(1) size_w(2) ...
                                    K length(fm_idxs)]);
                    for i = 1:size_w(1)
                        for j = 1:size_w(2)
                            es_rep(i,j,:,:) = es{layer}(m,n,:,fm_idxs);
                        end
                    end

                    %Repeat y{layer-1}(m:m+2*hrec_size,n:n+2*hrec_size,:,p) 
                    %into * x * x * x length(fm_idxs)
                    y_rep = repmat(...
                        y{layer-1}(m:m+2*hrec_size,n:n+2*hrec_size,:,p),...
                        [1 1 1 length(fm_idxs)]);

                    dw{layer}(:,:,p,fm_idxs)=dw{layer}(:,:,p,fm_idxs)+ ...
                                             sum(es_rep .* y_rep,3);
                end
            end
        end
        db{layer} = squeeze(sum(sum(sum(es{layer},3),2),1));
    else
        % -- Subsampling layer
        es_y = block_sum_2(y{layer-1}) .* es{layer};
        dw{layer} = squeeze(sum(sum(sum(es_y,3),2),1));
        db{layer} = squeeze(sum(sum(sum(es{layer},3),2),1));
    end
end
%% Layer 1: convolution layer..............................................
layer = 1;
size_es = size(es{layer});
size_w = size(net.w{layer});

dw{layer} = zeros(size_w);
hrec_size = net.hrec_size(layer);

for q = 1:net.no_fms(layer)
    for m = 1:size_es(1)
        for n = 1:size_es(2)           
            % Repeat part of es{layer} into size_w(1) x size_w(2) x * x *
            es_rep = repmat(es{layer}(m,n,:,q), ...
                            [size_w(1) size_w(2) 1 1]);
            
            dw{layer}(:,:,q) = dw{layer}(:,:,q) + ...
                               sum(es_rep .* ...
                                   x(m:m+2*hrec_size,n:n+2*hrec_size,:),3);  
        end
    end
end
db{layer} = squeeze(sum(sum(sum(es{layer},3),2),1));