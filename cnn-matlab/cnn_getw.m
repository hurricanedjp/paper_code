function w = cnn_getw(net)
% CNN_GETW: Get all trainable parameters of CNN as a column vector 
%
% SYNTAX
%        w = cnn_getw(net);
%
% PARAMETERS
%        net: CNN structure
%        w:   column vector containing all trainable parameters
%
% EXAMPLE
%
%
% NOTES
% Son Lam Phung, started 12-Jan-2006.

w = cnn_vectorize_wb(net, net.w, net.b);