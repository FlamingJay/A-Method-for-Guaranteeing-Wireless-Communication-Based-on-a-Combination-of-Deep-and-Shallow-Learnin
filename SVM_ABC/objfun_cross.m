%% Objective Function for crossvalind
function f=objfun_cross(cv,wine,wine_labels)
% cvΪʳ��Դ

cmd = [' -c ',num2str(cv(1)),' -g ',num2str(cv(2))]; %�ֱ��Ǳ��Ż�������ֵ

%��SVMģ�ͽ���ѵ����Ԥ�⣬�˴�ʹ��K�۽�����֤�ķ���
% ��ʱ��ѵ�������ֳɶ�Ӧ�ġ����ۡ���
accuracy_array = zeros(5,1);%����ÿһ�ε�׼ȷ��

indices=crossvalind('Kfold',wine_labels,5);
for k=1:5
    test_wine_index = (indices == k);%��ò��Լ�Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
    train_wine_index =~test_wine_index;
    % ����ѵ�����Ͳ��Լ�
    train_wine = wine(train_wine_index, :);
    test_wine = wine(test_wine_index,:);
    % �õ���Ӧ�ı�ǩ���
    train_wine_label = wine_labels(train_wine_index,1);
    test_wine_label = wine_labels(test_wine_index,1);
    
    model=libsvmtrain(train_wine_label,train_wine,cmd); % SVMģ��ѵ��
    [~,fitness,~]=libsvmpredict(test_wine_label,test_wine,model); % SVMģ��Ԥ�⼰�侫��
    accuracy_array(k,1) = fitness(1); 
end
    accuracy = mean(accuracy_array);
    %��Ӧ�Ⱥ���Ϊһ���������Ӽ���С��һ������
    f=1-accuracy/100; % ��׼ȷ����Ϊ�Ż���Ŀ�꺯��ֵ����ʱ�൱���Ǵ����ʣ���˶�Ӧ�������������Ѱ�ҵ�����С��ֵ