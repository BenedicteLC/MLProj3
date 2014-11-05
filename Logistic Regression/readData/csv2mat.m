%%read data from csv file and save it into a mat file
%in our codes, we save:
%test_inputs.csv --> testXData.mat
%train_inputs.csv --> trainData.mat
%train_outputs.csv --> trainYData.mat

function csv2mat(inCsvName, outMatName)

fid = fopen(inCsvName, 'r'); %open csv file

headerInfo = fgets(fid);  %get header info 

rowData = fgetl(fid); %get 1st line of sample
% for test_inputs.csv, it is zeros(20000, 2304);
% for train_inputs.csv, it is zeros(50000, 2304);
% for train_outputs.csv, it is zeros(50000, 1);
testXData = zeros(20000, 2304); 
ind = 1;
while rowData > 0
    tmp = str2num(rowData);
    testXData(ind,:) = tmp(2:end);
    ind = ind+1;
    rowData = fgetl(fid);
    disp(ind);
end

save outMatName testXData

fclose(fid);
