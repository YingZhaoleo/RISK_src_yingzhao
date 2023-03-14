function [Hu,Hw,Hy,Hbar] = build_handkel(u,w,y,depth)
 % build hankel matrix 
 % Args:
%     depth: depth w.r.t to u in in each column, depth for y will be depth+1
%     u,y: sequence of input/output data, y has one more data than u, each
%           column is a measurement in one time stamp 
% Returns
%     Hu: Hankel matrix of u with depth rows
%     Hy: Hankel matrix of y with depth+1 row
%     Hbar : the permuted Hankel matrix in the paper
    Hu = []; Hy = [];Hw = []; Hbar = [];
    for i = 1:size(u,2)-depth+1
        Hu(:,end+1) = vec(u(:,i:i+depth-1));
        Hw(:,end+1) = vec(w(:,i:i+depth-1));
        Hy(:,end+1) = vec(y(:,i:i+depth));
        Hbar(:,end+1) = [vec([y(:,i:i+depth-1);u(:,i:i+depth-1);]);y(:,i+depth)];
    end

end