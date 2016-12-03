function d = dltanh(x, y)
% DLTANH Derivative of ltanh
%
% SYNTAX
%        d = dltanh(x, y); 
%
% PARAMETERS
%        x: input 
%        y: value of function at x
%
% EXAMPLE
%       x = 0.2; y = ltanh(x);
%       d = dltanh(x,y)
%
% NOTES
%       d is more efficiently computed if y is known.
%
% Son Lam Phung, started 11-Jan-2006.

A = 1.7159;
S = 2/3;
d = A*S*(1-y/A) .* (1+y/A);