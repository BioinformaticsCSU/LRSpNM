% Objective: min_X  alpha/2*||X.*Omega-D.*Omega||_F^2 + ||X||_Sp^p + c1*Tr(X'LdX)+c2*Tr(XLtX')
% Written by: Gaoyan Wu
% Email: gaoyan_wu@csu.edu.cn

% Optimization with ADMM
function [X ,iter] = LRSpNM(D, Ld, Lt, Omega, p, alpha, c1, c2)
% D: m*n data matrix
% Ld: drug Laplacian matrix 
% Lt: target Laplacian matrix
% Omega: m*n matrix, Omega_ij=1 if Dij is observed, otherwise Omega_ij=0
% r: parameter alpha
% p: the p value of the Sp norm
% c1: parameter
% c2: parameter
% X: m*n recovered data matrix

ITER = 1000;
tol=1e-5;
[m, n] = size(D);
transpose = 0;
if m<n
    transpose = 1;
    D = D';
    Omega = Omega';
    [m, n] = size(D);
    change = Ld;
    Ld = Lt;
    Lt = change;
end;
X = D.*Omega;

% initialize
Y1 = zeros(m,n);
Y2 = zeros(m,n);

mu1 = 0.01;
mu2 = 0.01;
rho = 1.1; 

for iter = 1:ITER
    % update W 
    G = X + (1/mu1)*Y1;
    [U, S, V] = svd(G,0);
    s = diag(S);
    lambda = 1/mu1;
    for i = 1:length(s)
        s1(i) = findrootp0(s(i), lambda, p); 
    end;
    W = U*diag(s1)*V';
    
    % update Z
    tran=(1/mu2)*(Y2+alpha*(D.*Omega))+X;
    Z=tran-(alpha/(alpha+mu2))*(tran.*Omega);
    Z(Z<0)=0;
    Z(Z>1)=1;
    
    % update X
    % Sylvester equation£º
    % AX + XB = C £¨both of A and B are square matrix£©
    % Solution£ºX = Sylvester(A,B,C)
    I1 = eye(m);
    I2 = eye(n);
    A = 2*c1*Ld + mu1*I1;
    B = 2*c2*Lt + mu2*I2;
    C = mu1*W+ mu2*Z - Y1 - Y2;
    X_1 = sylvester(A,B,C);

    % updata Y1, Y2, mu1, mu2
    Y1 = Y1 + mu1*(X-W);
    Y2 = Y2 + mu2*(X-Z);
    mu1 = min(10^10,mu1*rho);
    mu2 = min(10^10,mu2*rho);
    
    % stopping criterion
     stop = norm(X_1-X,'fro')/max(norm(X_1,'fro'),1);
    if stop < tol
        break
    end
    
    X = X_1;
end;

if transpose == 1
    X = X';
end;

function x = findrootp0(a, r, p)

x = 0;
if p == 1  
    if a > r
        x = a-r;
    end;
elseif p == 0
    if a > sqrt(2*r)
        x = a;
    end;
else
v = (r*p*(1-p))^(1/(2-p))+eps;
v1 = v+r*p*v^(p-1);
ob0 = 0.5*a^2;
if a > v1
    x = a;
    for i = 1:10
        f = (x-a) + r*p*x^(p-1);
        g = 1-r*p*(1-p)*x^(p-2);
        x = x-f/g;
    end;
    ob1 = 0.5*(x-a)^2 + r*x^p;
    x_can = [0,x];
    [temp,idx] = min([ob0,ob1]);
    x = x_can(idx);
end;
end;
