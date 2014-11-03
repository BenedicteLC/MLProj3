function [predictions]=logregworking(X,ytrain,xtest,ytest)

% c=X;
% [n m] = size(c);

% c1n=sum(ytrain);
% c0n=n-c1n;
% 
% c1=zeros(c1n,m);
% c0=zeros(c0n,m);
% 
% index1=1;
% index0=1;
% for i=1:n
%     if (I(ytrain(i)==1))
%         c1(index1,:)=c(i,:);
%         index1=index1+1;
%     else
%         c0(index0,:)=c(i,:);
%         index0=index0+1;
%     end
% end
        
 %scatter(c1(:,1),c1(:,2),6,'b'),hold on;
 %scatter(c0(:,1),c0(:,2),6,'r'); hold on;

xtest= [ones(size(xtest,1),1), xtest];
 
alpha = 0.01;
epsilon=0.001;

[n,m] = size(X);

Ix = 500;

X = [ones(size(X,1),1), X];

W=zeros(m+1,1);

for i=1:Ix
sigma=sigmoid(X*W);
W=W+alpha*X'*(ytrain-sigma);
end;   
      
% error=0;
% for i = 1:n
%   error=error+(ytrain(i)*log(sigmoid(W'*X(i,:)')) + (1-ytrain(i))*log(1-sigmoid(W'*X(i,:)')));   
% end

% error=-1*error;

[rows columns]=size(xtest);
[Rtest Rcol]= size(ytest);
yresult=zeros(Rtest, Rcol);

for i=1:rows
    probSuccess=sigmoid(W'*xtest(i,:)');
    if (probSuccess>=0.5)
        ypredicted=1;
        %scatter(xtest(:,2),xtest(:,3),'g');
    
    else
        ypredicted=0;
        %scatter(xtest(:,2),xtest(:,3),'k');
    end
    
    predictions(i)=ypredicted;
    
    if ypredicted==ytest(i,1)
        yresult(i,1)=1;
    end
  
end

% accuracy2=(sum(yresult))/Rtest;
% [Rtrain Ctrain]= size(ytrain);
% yresult2=zeros(Rtrain, Ctrain);
% 
% for i=1:n
%     probSuccess=sigmoid(W'*X(i,:)');
%     if (probSuccess>=0.5)
%         ypredicted=1;
%         scatter(xtest(:,2),xtest(:,3),'g');
%     
%     else
%         ypredicted=0;
%         scatter(xtest(:,2),xtest(:,3),'k');
%     end
%     
%     if ypredicted==ytrain(i,1)
%         yresult2(i,1)=1;
%     end
%   
% end
% accuracy1=(sum(yresult2))/Rtrain;

end