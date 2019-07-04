%% Objective Function for crossvalind
function f=objfun_cross(cv,wine,wine_labels)
% cv为食物源

cmd = [' -c ',num2str(cv(1)),' -g ',num2str(cv(2))]; %分别是被优化的两个值

%对SVM模型进行训练和预测，此处使用K折交叉验证的方法
% 此时将训练集划分成对应的。。折。。
accuracy_array = zeros(5,1);%保存每一次的准确率

indices=crossvalind('Kfold',wine_labels,5);
for k=1:5
    test_wine_index = (indices == k);%获得测试集元素在数据集中对应的单元编号
    train_wine_index =~test_wine_index;
    % 划分训练集和测试集
    train_wine = wine(train_wine_index, :);
    test_wine = wine(test_wine_index,:);
    % 得到对应的标签类别
    train_wine_label = wine_labels(train_wine_index,1);
    test_wine_label = wine_labels(test_wine_index,1);
    
    model=libsvmtrain(train_wine_label,train_wine,cmd); % SVM模型训练
    [~,fitness,~]=libsvmpredict(test_wine_label,test_wine,model); % SVM模型预测及其精度
    accuracy_array(k,1) = fitness(1); 
end
    accuracy = mean(accuracy_array);
    %适应度函数为一个精度与子集大小的一个函数
    f=1-accuracy/100; % 以准确率作为优化的目标函数值，此时相当于是错误率，与此对应的是外面的最优寻找的是最小的值