function y = block_sum_2(x)
% BLOCK_SUM_2 Compute 2x2 block sum of 3-D matrix
%
% SYNTAX
%      y = block_sum_2(x)   
%
% PARAMETERS
%      x: input array,   D1xD2x*
%      y: output array, (D1/2)x(D2/2)x*
%
% EXAMPLE
%      x = rand(4, 4, 3);
%      y = block_sum_2(x);
%
% NOTES
% Son Lam Phung, started 12-Jan-2006, revised 04-Nov-2006.

if (ndims(x) == 3)
    y = x(1:2:end, 1:2:end, :) + x(1:2:end, 2:2:end, :) + ...
        x(2:2:end, 1:2:end, :) + x(2:2:end, 2:2:end, :);
elseif (ndims(x) == 4)
    y = x(1:2:end, 1:2:end, :, :) + x(1:2:end, 2:2:end, :, :) + ...
        x(2:2:end, 1:2:end, :, :) + x(2:2:end, 2:2:end, :, :);
end