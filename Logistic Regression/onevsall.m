function [predictions,testAccuracy]=onevsall(X,ytrain,xtest,ytest,numclasses)

    n = size(X);
    
    [c d]= size(xtest);
     predictions=zeros(c);
    predictions=predictions(1,:);
    
 
    for i=1:numclasses
    
    ytrainmodified=ytrain;
    ytestmodified=ytest;
    
   
    %set all other class outputs as 0 and the indexed one as 1
        for a=1:n
            
            if (ytrain(a)==i)
                ytrainmodified(a)=1;
            else
                ytrainmodified(a)=0;
            end
        end
     
     for b=1:c
         
        if (ytest(b,1)==i)
            ytestmodified(b)=1;
        else
            ytestmodified(b)=0;
        end
        
    
     end
         
     
    output=logregworking(X,ytrainmodified,xtest,ytestmodified);
    %recode output
        
    for b=1:c
        
        if (output(b)==1)
           predictions(b)=i;
        end  
    end
        
    end
    
    testAccuracy=0;
    for i=1:c
if(predictions(1,i)==ytest(i,1))
    testAccuracy=testAccuracy+1;
end
    end
    testAccuracy=testAccuracy/c;
    
end
