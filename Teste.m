% Example 3: Training a MLP NN. You may have to try several time to get
% results. The best way is to use nnukf.
clear all
clc

tic

rand('state',0)
randn('state',0)

N=40;        %training data length
alfa = 1^-3;
beta = 1^-2;

for teta = 1:N
    xt(teta) = (-11+0.5*teta)+alfa*rand(1);
end 
x = xt;
%x=randn(1,N); %training data x
Y_1=2; Y_2=1+exp( (1/1+exp(0.1-0.5*x)) + (1/1+exp(0.5+0.4*x)) -1); Y_3=0.5*sin(0.5*x); Y_4=0.5*(x./1+x.^2); Y_5=Y_1./(Y_2+Y_3+Y_4); Y = Y_5;
funcao = Y_5+beta*rand(1);%sin(x)+ cos(x);
y=funcao;     %training data y=f(x)

nh=3;         %MLP NN hiden nodes

ns=2*nh+nh+1;
f=@(z)y-(z(2*nh+(1:nh))'*tanh(z(1:nh)*x+z(nh+1:2*nh,ones(1,N)))+z(end,ones(1,N)));
theta0=rand(ns,1);
theta=ukfopt(f,theta0,1e-3,0.5*eye(ns),1e-7*eye(ns),1e-6*eye(N));
%theta=ekfopt(f,theta0,1e-3,0.5*eye(ns),1e-7*eye(ns),1e-6*eye(N));
W1=theta(1:nh);
b1=theta(nh+1:2*nh);
W2=theta(2*nh+(1:nh))';
b2=theta(ns);

M=500;         %Test data length
x1=randn(1,M);
y1=funcao;
z1=W2*tanh(W1*x1+b1(:,ones(1,M)))+b2(:,ones(1,M));
rafa=funcao;
figure(2)
plot(x,y1,'.b',x1,z1,'*r',x,rafa,'^y')
legend('Actual','NN model','Real model')

tempo = toc;



fprintf('\n\nTempo Gasto: %d\n\n',tempo);