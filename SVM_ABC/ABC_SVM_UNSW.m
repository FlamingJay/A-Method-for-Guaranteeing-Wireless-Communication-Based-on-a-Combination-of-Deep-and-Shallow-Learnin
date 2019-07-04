tic % ��ʱ��
%% ��ջ�����׼������
close all
clear
clc
format compact

load ae_train.mat
load train_label.mat


%% %%%%%%%%%%%%%��ABC�㷨�Ż�SVM�еĲ���c��g��ʼ%%%%%%%%%%%%%%%%%%%%
%% ������ʼ��
NP=10; % ��Ⱥ��ģ
FoodNumber=NP/2; % ��Դ���⣩������ͬʱҲ�ǲ��۷�͹۲�����������SN
limit=50; % ������Դ����û�����µĴ�������limitʱ������Դ�������³�ʼ��
maxCycle=10; % ����������
% ���Ż�������Ϣ
D=2; % ���Ż���������������Ϊc��g�Լ�������ѡ�����0Ϊ��ѡ��1Ϊѡ
ub=[100   100]; % ����ȡֵ�Ͻ磬�˴���c��g���Ͻ���Ϊ100
lb=[0.01 0.01]; % ����ȡֵ�½磬�˴���c��g���½���Ϊ0.01

runtime=1; % ���������ö�����У���ABC�㷨����runtime�Σ��Կ��������Ƚ���

BestGlobalMins=ones(1,runtime); % ȫ����Сֵ��ʼ����������Ż�Ŀ��ΪSVMԤ������׼ȷ�����Ӽ�������һ������
BestGlobalParams=zeros(runtime,D); % ���ڴ��ABC�㷨�Ż��õ������Ų���

for r=1:runtime % ����ABC�㷨runtime��
    % ��ʼ����Դ
fprintf('******************************************************************************\n');
Range = repmat((ub-lb),[FoodNumber 1]); %�ԣ�ub-lb�������ݶѵ��ڣ�FoodNumber*1���ľ���Range�С�Range�ľ����С����[��*FoodNumber ��*1]
Lower = repmat(lb, [FoodNumber 1]);
Foods = rand(FoodNumber,D) .* Range + Lower; %��Դ��rand����������һ��FoodNumber*D�ľ�������ֵ�ķ�Χ��[0,1[]֮�䡣
    
 %% ����ÿ����Դ���⣩��Ŀ�꺯��ֵ
ObjVal=ones(1,FoodNumber); 
    for k = 1:FoodNumber
%         ObjVal(k) = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
        ObjVal(k) = objfun_cross(Foods(k,:),train_data,train_attack_label);
    end
Fitness=calculateFitness(ObjVal); % ������Ӧ��ֵ
    
trial=zeros(1,FoodNumber); % ���ڼ�¼��i����Դ������trail(i)��û�����¹�

    % ���������Դ���⣩
BestInd=find(ObjVal==min(ObjVal)); % �ҵ���������С���Ǹ���Ϊ���Ž�
BestInd=BestInd(end);
GlobalMin=ObjVal(BestInd); % ����ȫ������Ŀ�꺯��ֵ
GlobalParams=Foods(BestInd,:); % ����ȫ�����Ų���Ϊ������Դ
Fit_Curve = zeros(maxCycle,1); %���ڻ�����Ӧ������

iter=1; % ������ʼ
while ((iter <= maxCycle)) % ѭ������
%%%%%%%%%%%%%%%%%%%%%���۷�������Ĺ���%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:(FoodNumber) % ����ÿ����Դ���⣩
        Param2Change=fix(rand*D)+1; % ���ѡ����Ҫ����Ĳ�����fix��������0�ķ������ȡ��������3.2->3��-1.2->-1��
        neighbour=fix(rand*(FoodNumber))+1; % ���ѡ��������Դ���⣩��׼������
        % ��Ҫ��֤ѡ���������Դ���ǵ�ǰ��Դ��i��
        while(neighbour==i)
            neighbour=fix(rand*(FoodNumber))+1;
        end
        sol=Foods(i,:); % ��ȡ��ǰ��Դ���⣩��Ӧ�ĵĲ���
            
        % ��������õ��µ���Դ��v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
        sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
        % ȷ������ȡֵ��Χ��Խ��
        ind=find(sol<lb);
        sol(ind)=lb(ind);
        ind=find(sol>ub);
        sol(ind)=ub(ind);
        % ����������Դ��Ŀ�꺯��ֵ����Ӧ�Ⱥ���ֵ
%             ObjValSol = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
        ObjValSol = objfun_cross(sol,train_data,train_attack_label);
        FitnessSol=calculateFitness(ObjValSol);
        % ���µ�ǰ��Դ�������Ϣ
        if (FitnessSol>Fitness(i))
            Foods(i,:)=sol;
            Fitness(i)=FitnessSol;
            ObjVal(i)=ObjValSol;
            trial(i)=0; % �����ǰ��Դ�������ˣ����Ӧ��trial����
        else
            trial(i)=trial(i)+1; % �����ǰ��Դû�б����£���trial(i)��1
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%% �۲��������Ĺ��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ����⣨��Դ���ĸ���
    prob=(0.9.*Fitness./max(Fitness))+0.1;
    % ѭ����ʼ��
    i=1;
    t=0;
    while(t<FoodNumber) % ѭ������
        if(rand<prob(i)) % ���������С�ڵ�ǰ�⣨��Դ���ĸ���
            t=t+1; % ѭ����������1

            Param2Change=fix(rand*D)+1; % ���ȷ����Ҫ����Ĳ���
            neighbour=fix(rand*(FoodNumber))+1; % ���ѡ��������Դ���⣩
            % ��Ҫ��֤ѡ���������Դ���ǵ�ǰ��Դ��i��
                while(neighbour==i)
                   neighbour=fix(rand*(FoodNumber))+1;
                end
            sol=Foods(i,:); % ��ȡ��ǰ��Դi���⣩��Ӧ�ĵĲ���
            % ��������õ��µ���Դ��v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
            sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
            % ��ֹ����Խ��
            ind=find(sol<lb);
            sol(ind)=lb(ind);
            ind=find(sol>ub);
            sol(ind)=ub(ind);
            % ����������Դ��Ŀ�꺯��ֵ����Ӧ�Ⱥ���ֵ
%                 ObjValSol = objfun(Foods(k,:),train_attack_label,feature_train,test_attack_label,feature_test);
            ObjValSol = objfun_cross(sol,train_data,train_attack_label);
            FitnessSol=calculateFitness(ObjValSol);
            % ���µ�ǰ��Դ�������Ϣ
                if (FitnessSol>Fitness(i))
                    Foods(i,:)=sol;
                    Fitness(i)=FitnessSol;
                    ObjVal(i)=ObjValSol;
                    trial(i)=0; % �����ǰ��Դ�������ˣ����Ӧ��trial����
                else
                    trial(i)=trial(i)+1; % �����ǰ��Դû�б����£���trial(i)��1
                end
        end
    
        i=i+1; % ����i
            if (i==(FoodNumber)+1) % ��ֵ������Դ��������i���³�ʼ��
                i=1;
            end   
    end 
    % ��ס������Դ
    ind=find(ObjVal==min(ObjVal));
    ind=ind(end);
    if (ObjVal(ind)<GlobalMin)
        GlobalMin=ObjVal(ind);
        GlobalParams=Foods(ind,:);
    end
        
%%%%%%%%%%%% ����������Ĺ��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % �ҳ��������ζ�û�б����µ���Դ        
    ind=find(trial==max(trial)); 
    ind=ind(end);
    % �������û�и��µĴ��������޶������������������³�ʼ������Դ
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
end % һ��ABC�㷨���

    BestGlobalMins(r)=GlobalMin; % ��¼����ABC�㷨������Ŀ�꺯��ֵ
    BestGlobalParams(r,:)=GlobalParams; % ��¼����ABC�㷨�����Ų���
    %%
    % 
    %  PREFORMATTED
    %  TEXT
    % 

end % end of runs

fprintf('\n*******************end of test********************\n');
load ae_test.mat
load test_label.mat
%% %%%%%%%%%%%%%��ABC�㷨�Ż�SVM�еĲ���c��g����%%%%%%%%%%%%%%%%%%%%
%% ��ӡ����ѡ��������������������һ��ABC�㷨Ѱ�ŵõ��Ĳ���
bestc=GlobalParams(1);
bestg=GlobalParams(2);

disp('��ӡѡ����');
str=sprintf('Best c = %g��Best g = %g',bestc,bestg);
disp(str)
%% ������ѵĲ�������SVM����ѵ��
cmd_gwosvm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwosvm = libsvmtrain(train_label,encoded_train,cmd_gwosvm);
%% SVM����Ԥ��
[predict_label,accuracy,~] = libsvmpredict(test_label,encoded_test,model_gwosvm);
% ��ӡ���Լ�����׼ȷ��
total = length(test_label);
right = sum(predict_label == test_label);
disp('��ӡ���Լ�����׼ȷ��');
str = sprintf( '\nAccuracy = %g%% (%d/%d)',accuracy(1),right,total);
disp(str);
%% �������
% ���Լ���ʵ�ʷ����Ԥ�����ͼ
figure(1);
hold on;
plot(test_label,'o');
plot(predict_label,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on
snapnow
figure(2);
plot(1:maxCycle,Fit_Curve(2:maxCycle+1,1));
%% ��ʾ��������ʱ��
toc