clear all
clc

rand('state',0);
n=10;
nh=4;
W1=rand(nh,n);
b1=rand(nh,1);
W2=rand(n,nh);
b2=rand(n,1);
x0=zeros(n,1);
f=@(x)W2*tanh(W1*x+b1)+b2;
tol=1e-6;
P=1000*eye(n);
Q=1e-7*eye(n);
R=1e-7*eye(n);
x=ekfopt(f,x0,tol,P,Q,R);