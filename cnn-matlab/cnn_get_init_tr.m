function tr = cnn_get_init_tr()
% CNN_GET_INIT_TR Get initial training record for CNN
%
% SYNTAX
%       tr = cnn_get_init_tr;
%
% PARAMETERS
%       tr: initial training record
%             tr.mse = [];           % Mean square error
%             tr.time = [];          % Training time in seconds
%             tr.epoch = [];         % Number of training epochs
%             tr.output_eval = [];   % Total network output evaluations
%             tr.gradient_eval = []; % Total gradient evaluations
%             tr.hessian_eval = [];  % Total Hessian matrix evaluations 
%             tr.jacobian_eval = []; % Total Jacobian matrix evaluations 
% EXAMPLE
%       tr = cnn_get_init_tr;      
%
% NOTES
% Son Lam Phung, started 12-Jan-2006.

tr.mse = [];           % Mean square error
tr.time = [];          % Training time in seconds
tr.epoch = [];         % Number of training epochs
tr.output_eval = [];   % Total network output evaluations
tr.gradient_eval = []; % Total gradient evaluations
tr.hessian_eval = [];  % Total Hessian matrix evaluations 
tr.jacobian_eval = []; % Total Jacobian matrix evaluations