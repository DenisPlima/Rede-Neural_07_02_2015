function [x,e]=ekfopt(h,x,tol,P,Q,R)
%EKFOPT     Unconstrained optimization using the extended Kalman filter
%
%       [x,e]=ekfopt(f,x,tol,P,Q,R) minimizes e=norm(f(x)) until e<tol. P, Q
%       and R are tuning parmeters relating to the Kalman filter
%       performance. P and Q should be n x n, where n is the number of 
%       decision variables, while R should be m x m, where m is the 
%       dimension of f. Normally, Q and R should be set to d*I e*I where 
%       both d and e are vary small positive scalars. P could be initially 
%       set to a*I where a is the estimated distence between the initial 
%       guess and the optimal valuve. This function can also be used to solve 
%       a set of nonlinear equation, f(x)=0.
%
% This is an example of what a nonlinear Kalman filter can do. It is
% strightforward to replace the extended Kalman filter with the unscented 
% Kalman filter to achieve this same functionality. The advantage of using 
% the unscented Kalman filter is that it is derivative-free. Therefore, it 
% is also suitable for non-analytical functions where no derivatives can 
% be obtained.
%


n=numel(x);     %numer of decision variables
f=@(x)x;        %virtual state euqation to update the decision parameter
e=h(x);         %initial residual
[m1,m2]=size(e);%number of equations to be solved
% m=numel(e);
if nargin<3,    %default values
    tol=1e-9;
end
if nargin<4
    P=eye(n);
end
if nargin<5
    Q=1e-6*eye(n);
end
if nargin<6
    R=1e-6*eye(m);
end
k=1;            %number of iterations
z=zeros(m1,m2); %target vector
%z=zeros(m,1);   %target vector
ne=norm(e);
    tic
while k< 100%ne>tol
    [x,P]=ekf(f,x,P,h,z,Q,R);               %the unscented Kalman filter
    %-----------------------------------------------------------------------------------
    Weight=x;   %Weights Vector
    hidden=8;    %Number of Hidden Neurons
    Ne=100;      %Number of Testing Samples
    NS=2*hidden+hidden+1;
    W1_e=Weight(1:hidden);
    b1_e=Weight(hidden+1:2*hidden);
    W2_e=Weight(2*hidden+(1:hidden))';
    b2_e=Weight(NS);
    %------------------------------------------------------------------------------------
    
    %Evaluate NN
    x2=randn(1,Ne);
    y2=(x2+2.*cos(x2).*-5.*sin(x2)+ x2.^3);
    z2=W2_e*logsig(W1_e*x2+b1_e(:,ones(1,Ne)))+b2_e(:,ones(1,Ne));
    
    %-------------------------------------------------------------------------------------
    
    %Mse Training    
    MSE_Training=sum((y2-z2).^2)/length(y2);
    Error_Training(k)=MSE_Training;     
      
    %-----------------------------------------------------------------------------------
    e=h(x);
    ne=norm(e);                                 %residual
    if mod(k,100)==1
        %fprintf('k=%d e=%g\n',k,ne)    %display iterative information
        fprintf('e=%g\n',ne)    %display iterative information
    end
    vet3(k) = ne;
    vet4(k) =k;
    k=k+1;                                  %iteration count
end
 toc

Error_Training(99)
fprintf('k=%d e=%g\n',k,ne)            %final result
save Error_Ekf.mat Error_Training
figure(1)
semilogy(Error_Training)
title('Training Error Analysis')
xlabel('Iterations')
ylabel('Training Error')