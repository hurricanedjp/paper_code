%% CNN used by Garcia and Delakis for face detection
c = {cnn_cm('full', 1, 4), cnn_cm('1-to-1', 4), ...
     cnn_cm('1-to-2 2-to-1', 4), cnn_cm('1-to-1', 14), ...
     cnn_cm('1-to-1', 14), cnn_cm('full', 14, 1)};
net = cnn_new([36 32], c, [5 5; 2 2; 3 3; 2 2; 0 0; 0 0], ...
      repmat({'tansig'}, 1, length(c)), 'rprop');

%%
c = {cnn_cm('full', 1, 6), cnn_cm('one', 6), cnn_cm('toeplitz', 6, 16, 3), cnn_cm('one', 16), cnn_cm('full', 16, 120), cnn_cm('full', 120, 1)}
net = cnn_new([32 32], c, [5 2 5 2 5 0], repmat({'tansig'}, 1, length(c)), 'rprop')

%%
c = {cnn_cm('full', 1, 6), cnn_cm('one', 6), cnn_cm('toeplitz', 6, 16, 3), cnn_cm('one', 16), cnn_cm('full', 16, 10), cnn_cm('full', 10, 1)}
net = cnn_new([32 32], c, [5 2 5 2 5 0], repmat({'tansig'}, 1, length(c)), 'rprop')

%%
c = {cnn_cm('full', 1, 4), cnn_cm('one', 4), cnn_cm('toeplitz', 4, 6, 3), cnn_cm('one', 6), cnn_cm('full', 6, 4), cnn_cm('full', 4, 1)}
net = cnn_new([32 32], c, [5 2 5 2 5 0], repmat({'tansig'}, 1, length(c)), 'rprop')

%%
c = {cnn_cm('full', 1, 3), cnn_cm('one', 3), cnn_cm('toeplitz', 3, 6, 2), cnn_cm('one', 6), cnn_cm('full', 6, 6), cnn_cm('full', 6, 1)}
net = cnn_new([32 32], c, [5 2 5 2 5 0], repmat({'tansig'}, 1, length(c)), 'rprop')

%%
c = {cnn_cm('full', 1, 4), cnn_cm('one', 4), cnn_cm('binary', 4, 8), cnn_cm('one', 8), cnn_cm('full',8,5), cnn_cm('full', 5, 1)}
net = cnn_new([32 32], c, [5 2 5 2 5 0], repmat({'tansig'}, 1, length(c)), 'rprop')
