tic % 计时器
%% 清空环境，准备数据
close all
clear
clc
format compact

load ae_train.mat
load train_label.mat


%% %%%%%%%%%%%%%用ABC算法优化SVM中的参数c和g开始%%%%%%%%%%%%%%%%%%%%
%% 参数初始化
NP=10; % 蜂群规模
FoodNumber=NP/2; % 蜜源（解）数量，同时也是采蜜蜂和观察蜂的数量，即SN
limit=50; % 当有蜜源连续没被更新的次数超过limit时，该蜜源将被重新初始化
maxCycle=10; % 最大迭代次数
% 待优化参数信息
D=2; % 待优化参数个数，次数为c和g以及特征的选择与否，0为不选，1为选
ub=[100   100]; % 参数取值上界，此处将c和g的上界设为100
lb=[0.01 0.01]; % 参数取值下界，此处将c和g的下界设为0.01

runtime=1; % 可用于设置多次运行（让ABC算法运行runtime次）以考察程序的稳健性

BestGlobalMins=ones(1,runtime); % 全局最小值初始化，这里的优化目标为SVM预测结果中准确率与子集个数的一个函数
BestGlobalParams=zeros(runtime,D); % 用于存放ABC算法优化得到的最优参数

for r=1:runtime % 运行ABC算法runtime次
    % 初始化蜜源
fprintf('******************************************************************************\n');
Range = repmat((ub-lb),[FoodNumber 1]); %以（ub-lb）的内容堆叠在（FoodNumber*1）的矩阵Range中。Range的矩阵大小，是[列*FoodNumber 行*1]
Lower = repmat(lb, [FoodNumber 1]);
Foods = rand(FoodNumber,D) .* Range + Lower; %蜜源，rand函数生成了一个FoodNumber*D的矩阵，其中值的范围在[0,1[]之间。
    
 %% 计算每个蜜源（解）的目标函数值
ObjVal=ones(1,FoodNumber); 
    for k = 1:FoodNumber
%         ObjVal(k) = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
        ObjVal(k) = objfun_cross(Foods(k,:),train_data,train_attack_label);
    end
Fitness=calculateFitness(ObjVal); % 计算适应度值
    
trial=zeros(1,FoodNumber); % 用于记录第i个蜜源有连续trail(i)次没被更新过

    % 标记最优蜜源（解）
BestInd=find(ObjVal==min(ObjVal)); % 找到错误率最小的那个作为最优解
BestInd=BestInd(end);
GlobalMin=ObjVal(BestInd); % 更新全局最优目标函数值
GlobalParams=Foods(BestInd,:); % 更新全局最优参数为最优蜜源
Fit_Curve = zeros(maxCycle,1); %用于绘制适应度曲线

iter=1; % 迭代开始
while ((iter <= maxCycle)) % 循环条件
%%%%%%%%%%%%%%%%%%%%%采蜜蜂搜索解的过程%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:(FoodNumber) % 遍历每个蜜源（解）
        Param2Change=fix(rand*D)+1; % 随机选择需要变异的参数，fix函数是向0的方向对数取整，比如3.2->3，-1.2->-1。
        neighbour=fix(rand*(FoodNumber))+1; % 随机选择相邻蜜源（解）以准备变异
        % 需要保证选择的相邻蜜源不是当前蜜源（i）
        while(neighbour==i)
            neighbour=fix(rand*(FoodNumber))+1;
        end
        sol=Foods(i,:); % 提取当前蜜源（解）对应的的参数
            
        % 参数变异得到新的蜜源：v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
        sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
        % 确保参数取值范围不越界
        ind=find(sol<lb);
        sol(ind)=lb(ind);
        ind=find(sol>ub);
        sol(ind)=ub(ind);
        % 计算变异后蜜源的目标函数值和适应度函数值
%             ObjValSol = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
        ObjValSol = objfun_cross(sol,train_data,train_attack_label);
        FitnessSol=calculateFitness(ObjValSol);
        % 更新当前蜜源的相关信息
        if (FitnessSol>Fitness(i))
            Foods(i,:)=sol;
            Fitness(i)=FitnessSol;
            ObjVal(i)=ObjValSol;
            trial(i)=0; % 如果当前蜜源被更新了，则对应的trial归零
        else
            trial(i)=trial(i)+1; % 如果当前蜜源没有被更新，则trial(i)加1
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%% 观察蜂搜索解的过程 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 计算解（蜜源）的概率
    prob=(0.9.*Fitness./max(Fitness))+0.1;
    % 循环初始化
    i=1;
    t=0;
    while(t<FoodNumber) % 循环条件
        if(rand<prob(i)) % 若随机概率小于当前解（蜜源）的概率
            t=t+1; % 循环计数器加1

            Param2Change=fix(rand*D)+1; % 随机确定需要变异的参数
            neighbour=fix(rand*(FoodNumber))+1; % 随机选择相邻蜜源（解）
            % 需要保证选择的相邻蜜源不是当前蜜源（i）
                while(neighbour==i)
                   neighbour=fix(rand*(FoodNumber))+1;
                end
            sol=Foods(i,:); % 提取当前蜜源i（解）对应的的参数
            % 参数变异得到新的蜜源：v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
            sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
            % 防止参数越界
            ind=find(sol<lb);
            sol(ind)=lb(ind);
            ind=find(sol>ub);
            sol(ind)=ub(ind);
            % 计算变异后蜜源的目标函数值和适应度函数值
%                 ObjValSol = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
            ObjValSol = objfun_cross(sol,train_data,train_attack_label);
            FitnessSol=calculateFitness(ObjValSol);
            % 更新当前蜜源的相关信息
                if (FitnessSol>Fitness(i))
                    Foods(i,:)=sol;
                    Fitness(i)=FitnessSol;
                    ObjVal(i)=ObjValSol;
                    trial(i)=0; % 如果当前蜜源被更新了，则对应的trial归零
                else
                    trial(i)=trial(i)+1; % 如果当前蜜源没有被更新，则trial(i)加1
                end
        end
    
        i=i+1; % 更新i
            if (i==(FoodNumber)+1) % 若值超过蜜源数量，则i重新初始化
                i=1;
            end   
    end 
    % 记住最优蜜源
    ind=find(ObjVal==min(ObjVal));
    ind=ind(end);
    if (ObjVal(ind)<GlobalMin)
        GlobalMin=ObjVal(ind);
        GlobalParams=Foods(ind,:);
    end
        
%%%%%%%%%%%% 侦查蜂搜索解的过程 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 找出连续最多次都没有被更新的蜜源        
    ind=find(trial==max(trial)); 
    ind=ind(end);
    % 如果连续没有更新的次数大于限定次数，则由侦查蜂重新初始化该蜜源
    if (trial(ind)>limit) 
        Bas(ind)=0;
        sol=(ub-lb).*rand(1,D)+lb;
%             ObjValSol = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
        ObjValSol = objfun_cross(sol,train_data,train_attack_label);
        FitnessSol=calculateFitness(ObjValSol);
        Foods(ind,:)=sol;
        Fitness(ind)=FitnessSol;
        ObjVal(ind)=ObjValSol;
    end
        
        iter=iter+1;
    Fit_Curve(iter,1) = calculateFitness(GlobalMin);
end % 一次ABC算法完结

    BestGlobalMins(r)=GlobalMin; % 记录本次ABC算法的最优目标函数值
    BestGlobalParams(r,:)=GlobalParams; % 记录本次ABC算法的最优参数
    %%
    % 
    %  PREFORMATTED
    %  TEXT
    % 

end % end of runs

fprintf('\n*******************end of test********************\n');
load ae_test.mat
load test_label.mat
%% %%%%%%%%%%%%%用ABC算法优化SVM中的参数c和g结束%%%%%%%%%%%%%%%%%%%%
%% 打印参数选择结果，这里输出的是最后一次ABC算法寻优得到的参数
bestc=GlobalParams(1);
bestg=GlobalParams(2);

disp('打印选择结果');
str=sprintf('Best c = %g，Best g = %g',bestc,bestg);
disp(str)
%% 利用最佳的参数进行SVM网络训练
cmd_gwosvm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwosvm = libsvmtrain(train_label,encoded_train,cmd_gwosvm);
%% SVM网络预测
[predict_label,accuracy,~] = libsvmpredict(test_label,encoded_test,model_gwosvm);
% 打印测试集分类准确率
total = length(test_label);
right = sum(predict_label == test_label);
disp('打印测试集分类准确率');
str = sprintf( '\nAccuracy = %g%% (%d/%d)',accuracy(1),right,total);
disp(str);
%% 结果分析
% 测试集的实际分类和预测分类图
figure(1);
hold on;
plot(test_label,'o');
plot(predict_label,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on
snapnow
figure(2);
plot(1:maxCycle,Fit_Curve(2:maxCycle+1,1));
%% 显示程序运行时间
toc