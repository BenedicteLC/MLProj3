
function confusion_matrix(pred ,y)

% confusion matrix to show the classification results
% pred - the predicted labels
% y - the actual labels 

%initialization
num_class = 10;
mat = zeros(num_class, num_class);

for i = 0:num_class-1
    y_actual = find(y == i);
    for j = 0:num_class-1
        y_pred = find(pred == j);
        res = intersect(y_actual, y_pred);
        mat(i+1,j+1) = length(res);
    end
end

mat_num = sum(mat,2);
for i = 1:num_class
    mat(i,:) = mat(i,:)/mat_num(i);
end

imagesc(1:num_class,1:num_class,mat);
ylabel('Actual Class','FontSize',12,'fontWeight','bold')
xlabel('Predicted Class','FontSize',12,'fontWeight','bold')
tick = {'0'; '1'; '2'; '3'; '4'; '5'; '6'; '7'; '8'; '9'};
set(gca,'YTickLabel',tick)
set(gca,'XTickLabel',tick)
set(gca, 'XTick', 1:num_class, 'YTick', 1:num_class);
set(gca,'FontSize',12)
colorbar

print(gcf,'-dpng','Confusion Matrix');