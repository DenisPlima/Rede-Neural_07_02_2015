%----------------------------------------------------------------------------------------------------------------
%                                    TRAINING A NEURAL NETWORK
%----------------------------------------------------------------------------------------------------------------
clear all
clc

rand('state',0)
randn('state',0)


% DATA SAMPLES TO TRAINING (Will be Offline Data)
N=100; %number of samples
x=randn(1,N)+0.01*randn(1); %training data x
y=(x+2*cos(x).*-5.*sin(x)+ x.^3)+0.01*randn(1);  %training data y


%--------------------------------------------------------------------------------------------------------------
nh=4;         %MLP NN hidden nodes
ns=2*nh+nh+1;
f=@(z)y-(z(2*nh+(1:nh))'*logsig(z(1:nh)*x+z(nh+1:2*nh,ones(1,N)))+z(end,ones(1,N)));
theta0=rand(ns,1);
theta=ukfopt(f,theta0,1e-3,1e-7*eye(ns),1e-7*eye(ns),1e-7*eye(N));
W1=theta(1:nh);
b1=theta(nh+1:2*nh);
W2=theta(2*nh+(1:nh))';
b2=theta(ns);

M=500;         %Test data length
x1=randn(1,M);
y1=(x1+2.*cos(x1).*-5.*sin(x1)+ x1.^3);


z1=W2*logsig(W1*x1+b1(:,ones(1,M)))+b2(:,ones(1,M));

Error=abs(y1-z1).^2;
MSE_test = sum(Error(:))/numel(y1);
MSE=sum((y1-z1).^2)/length(y1);
%MSE_test
MSE
plotregression(y1,z1)
figure (2)
plot(x1,y1,'.b',x1,z1,'.r',x,y,'.g')
legend('Actual','NN model','Training')



