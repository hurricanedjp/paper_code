% DESCRIPTION
%
%
% NOTES
% Son Lam Phung, started 14-Apr-2009.

load('pyranet_train_data_face_w32');
load('pyranet_train_data_nonface_w32');

%%
K = 1000;
N = size(x_face,3);
skip = floor(N/K);
train_idxs = 1:skip:(K-1)*skip + 1;
test_idxs = setdiff(1:N, train_idxs);

x1 = x_face(:,:,train_idxs);
x2 = x_nonface(:,:, train_idxs);
d1 = ones(1,K);
d2 = -ones(1,K);
x = cat(3, x1, x2);
d = [d1 d2];
save('train_data.mat', 'x', 'd');

%%
x1 = x_face(:,:,test_idxs);
x2 = x_nonface(:,:, test_idxs);
K_test = length(test_idxs);
d1 = ones(1,K_test);
d2 = -ones(1,K_test);
x_test = cat(3, x1, x2);
d_test = [d1 d2];
save('test_data.mat', 'x_test', 'd_test');