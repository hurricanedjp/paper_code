function c = cnn_cm(connect_type, src_fms, dst_fms, group_fms)
% CNN_CM: Create a connection matrix from one layer to the next layer
%         in a CNN  
% SYNTAX
%   c = cnn_cm(connect_type, src_fms, dst_fms, group_fms);
%
% PARAMETERS
%   connect_type: a string
%         'full'           full connection
%         '1-to-1'         1 feature map to exactly 1 feature map
%         '1-to-2'         1 feature map branchs to exactly 2 feature maps
%         '2-to-1'         2 feature maps merge to 1 feature map
%         '1-to-2 2-to-1'  combination of '1-to-2' and '2-to-1'
%         'toeplitz'       toeplitz-like connection
%         'custom'         custom connection 
%
%   src_fms        : number of source feature maps
%   dst_fms        : number of destination feature maps
%   group_fms      : group feature maps
%
% EXAMPLES
%   c = cnn_cm('full', 3, 4)
%   c = cnn_cm('1-to-1', 3)
%   c = cnn_cm('1-to-2', 2)
%   c = cnn_cm('2-to-1', 4)
%   c = cnn_cm('1-to-2 2-to-1', 4);
%   c = cnn_cm('toeplitz', 3, 4, 2)
%   c = cnn_cm('custom', 3, 4)
%
% NOTES
%   c(i,j) = true means there is a connection 
%            from feature map i in source layer
%            to feature map j in the destination layer
%
% Son Lam Phung, started 13-Jan-2006, revised 1-Nov-2006.

%% Default parameters......................................................
if nargin < 3
    dst_fms = 0;
end

if nargin < 4
    group_fms = 0;
end

if group_fms > src_fms
    group_fms = mod(group_fms, src_fms) + 1;
end

%% Create connection matrix according to connect_type......................
switch connect_type
    case {'full'}
        % full connection
        c = true(src_fms, dst_fms);
        
    case {'1-to-1'}
        % 1 feature map connects to exactly 1 feature map
        c = (eye(src_fms) ~=0);  

    case {'1-to-2'}    
        % 1 feature map branchs to exactly 2 feature maps        
        c = false(src_fms, 2*src_fms);
        for p=1:src_fms
            c(p, 2*p -1:2*p)= true;
        end
        
    case {'2-to-1'} 
        % 2 feature maps merge to 1 feature map
        dst_fms = floor(src_fms/2);
        c = false(2*dst_fms, dst_fms);
        for p=1:dst_fms
            c(2*p-1:2*p,p)= true;
        end
        
    case {'1-to-2 2-to-1'}
        dst_fms = 2*src_fms + src_fms*(src_fms-1)/2;
        c = false(src_fms, dst_fms);
        for p=1:src_fms
            c(p, 2*p-1:2*p)= true;
        end
        
        i = 2*src_fms;
        for p1 = 1:src_fms-1
            for p2 = p1+1:src_fms
                i = i + 1;
                c(p1, i) = true;
                c(p2, i) = true;
            end
        end
        
    case {'toeplitz'}
        % toeplitz-like connection 
        % group_frms determines how many consecutive feature maps are used
        c = false(src_fms, dst_fms);

        for q = 1:dst_fms
            for s = q:q+group_fms-1
                if s <= src_fms
                    p = s;
                else
                    p = mod(s, src_fms);
                    if (p == 0)
                        p = src_fms;
                    end
                end
                c(p, q) = true;
            end
        end
        
    otherwise
        % custom connection 
        % user can modify individual entries in c
        c = true(src_fms, dst_fms);
end