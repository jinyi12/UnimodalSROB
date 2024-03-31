function x = tikhonov_poly(b, A, sizeparams, k1, k2, k3)
    % solves linear regression Ax = b with L2/tikhonov regularization penalty
    %
    % INPUTS
    % A     data matrix
    % b     right-hand side
    % k     Tikhonov weighting
    % sizeparams  size of the operators contains:
    %   l   number of linear terms
    %   s   number of quadratic terms
    %   mr  number of bilinear terms
    %   m   number of input terms
    %   c   number of constant terms
    %   drp number of poly terms
    % k1    Tikhonov weighting for the first and second operator corresponding to the constant and linear terms
    % k2    Tikhonov weighting for the third operator corresponding to the quadratic term
    % k3    Tikhonov weighting for the fourth operator corresponding to the polynomial term
    %
    % OUTPUT
    % x     solution
    %
    % AUTHOR
    % Elizabeth Qian (elizqian@mit.edu) 12 June 2019
    
    [~,q] = size(b);
    [~,p] = size(A);

    % create the Tikhonov matrix
    l = sizeparams.l;
    s = sizeparams.s;
    mr = sizeparams.mr; % not used
    m = sizeparams.m; 
    c = sizeparams.c;
    drp = sizeparams.drp;

%     operators.A = temp(:,1:l);
% operators.F = temp(:,l+1:l+s);
% operators.H = F2H(operators.F);
% operators.N = temp(:,l+s+1:l+s+mr);
% operators.B = temp(:,l+s+mr+1:l+s+mr+m);
% operators.C = temp(:,l+s+mr+m+1:l+s+mr+m+c);
% operators.P = temp(:,l+s+mr+m+c+1:l+s+mr+m+c+drp);


    % create the pseudo matrices for augmenting the data matrix A, we only use the constant, linear, quadratic and polynomial terms
    pseudo = eye(p);
    pseudo(1:l,:) = sqrt(k1)*pseudo(1:l,:);
    pseudo(l+1:l+s,:) = sqrt(k2)*pseudo(l+1:l+s,:);

    if m > 0
        pseudo(l+s+1:l+s+m,:) = sqrt(k1)*pseudo(l+s+1:l+s+m,:);
        pseudo(l+s+m+1:l+s+c,:) = sqrt(k1)*pseudo(l+s+m+1:l+s+c,:);
        pseudo(l+s+c+1:l+s+c+drp,:) = sqrt(k3)*pseudo(l+s+c+1:l+s+c+drp, :);
    else
        pseudo(l+s+1:l+s+c,:) = sqrt(k1)*pseudo(l+s+1:l+s+c,:);
        pseudo(l+s+c+1:l+s+c+drp,:) = sqrt(k3)*pseudo(l+s+c+1:l+s+c+drp,:);    
    end


    % pseudo = [pseudo1;pseudo2;pseudo3;pseudo4];
    Aplus  = [A;pseudo];
    bplus  = [b;zeros(p, q)];
    
    x = Aplus\bplus;
end