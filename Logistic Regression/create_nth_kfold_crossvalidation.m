function [Xtrain, Ytrain, Xval, Yval] = create_nth_kfold_crossvalidation(Xall, Yall, k, kk)

%seperate dataset into training set and cross-validation set
% Xall: dataset
% Yall: labels for dataset
% k: number of the folds
% kk: choose the kk-th part as cross-validation set and all other parts
% as training set

    num = ceil(length(Yall)/k);
    Xall_tmp = Xall;
    Yall_tmp = Yall;
    stt = (kk-1)*num;
    if kk == k
        Xval = Xall(stt+1:end,:);
        Yval = Yall(stt+1:end,:);
        Xtrain = Xall(1:stt,:);
        Ytrain = Yall(1:stt,:);
    else
        Xval = Xall(stt+1:stt+num,:);
        Yval = Yall(stt+1:stt+num,:);
        Xall_tmp(stt+1:stt+num,:) = [];
        Yall_tmp(stt+1:stt+num,:) = [];
        Xtrain = Xall_tmp;
        Ytrain = Yall_tmp;
    end
end