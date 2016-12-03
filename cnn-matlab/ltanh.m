function y = ltanh(x)
% LTANH Hyperbolic tangent function
%       Proposed by LeCun
%
% SYNTAX
%       y = ltanh(x)
%
% PARAMETERS
%       x: input
%       y: output
%
% EXAMPLE
%       y = ltanh(0.2);
%
% NOTES
%       y = 1.7159 * tanh(2/3*x);
%
% Son Lam Phung, started 11-Jan-2006.

A = 1.7159;
S = 2/3;
y = A*tanh(x*S);