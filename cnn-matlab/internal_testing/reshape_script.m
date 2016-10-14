% DESCRIPTION
%
%
% NOTES
% Son Lam Phung, started 05-Nov-2006.

clc
N = 4;
K = 3;
t = rand(N,K)
%%
t1 = reshape(t', [1 1 K N]); % this is the way to reshape
%%
k = 2; n = 3;
[t(n,k) t1(1,1,k,n)]
%%
k = 1; n = 2;
[t(n,k) t1(1,1,k,n)]
%%
k = 3; n = 2;
[t(n,k) t1(1,1,k,n)]