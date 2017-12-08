%% ML PROJECT: "default of credit card clients Data Set"
% Yann Adjanor & Shawn Ban
% 14 November 2017

% Initialisize
close all; clear; clc;
screensize = get(0, 'Screensize');

%% LOAD DATA
% Skip the first row
ccdata = readtable('default of credit card clients.xls', 'Range','2:30002');

%% CLEAN THE DATA
%Update last variable name
ccdata.Properties.VariableNames(end) = {'DEFAULT'};
%find missing data
if nnz(ismissing(ccdata))
    %do some data interpolation/cleaning
    disp('Missing data: action required');
else
    %do nothing (as it is the case here)
    disp('No missing data');
end

%% DEFINE THE CATEGORICAL VARIABLES (Sex, Education, Marriage) %%%% OR MAYBE NOT
categorical(ccdata.SEX);
categorical(ccdata.EDUCATION);
categorical(ccdata.MARRIAGE);
categorical(ccdata.PAY_0);
categorical(ccdata.PAY_2);
categorical(ccdata.PAY_3);
categorical(ccdata.PAY_4);
categorical(ccdata.PAY_5);
categorical(ccdata.PAY_6);
categorical(ccdata.DEFAULT);

%% BASIC STATISTICS DATA 
fprintf('\nBasic statistics.\n');

varnames = strrep(ccdata.Properties.VariableNames,'_','');
ccdata.Properties.VariableNames = varnames;

stats = calcstats(ccdata);
f = subfig(3,1,1,'Basic Statistics on ALL instances');
t= display_stats(f,stats);

%stats = calcstats(ccdata(ccdata.DEFAULT==0,:));
%f = subfig(3,1,2,'Basic Statistics on DEFAULT=0 instances');
%t= display_stats(f,stats)

%stats = calcstats(ccdata(ccdata.DEFAULT==1,:));
%f = subfig(3,1,3,'Basic Statistics on DEFAULT=1 instances');
%t= display_stats(f,stats)

fprintf('\nDisplay Stats Table.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);

%% PLOT HISTOGRAMS FOR EACH ATTRIBUTE


f = subfig(2,2,1,'First 12 Features');
b = [];
for i = [1:12]
    ax  = subplot(3,4,i);
    b = histogram(ccdata{:,i});
    title(varnames(i));
end
f = subfig(2,2,3,'Second 12 Features');
b = [];
for i = [13:length(varnames) - 1]
    ax  =subplot(3,4,i - 12);
    b = histogram(ccdata{:,i});
    title(varnames(i));
end

clearvars b stat 

fprintf('\nHistograms for each feature.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);


%% LOOKING AT CORRELATIONS 
f = subfig(2,4,2,'Correlation Heatmap Before Features Transformation');
Xcc = ccdata(:,2:end);
img = plot_correlations(Xcc);
fprintf('\nCorrelation heatmap.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);

%% DATA TRANSFORMATIONS
paycols = 7:12;
bamtcols = 13:18;
pamtcols = 19:24;
% Normalise all cols except categoricals
ccdata{:,[2,6, paycols, bamtcols, pamtcols]} = norm_data(ccdata{:,[2,6, paycols, bamtcols, pamtcols]}, 'zscore');
%%  PCA aggregation
pclist = {'PC1','PC2','PC3','PC5','PC5','PC6'};
vargrp = {'PAY','BILLAMT', 'PAYAMT'};
Pca ={};
eig = {};
clr = ['w','w','w'];
[Pca{1}, eig{1}, ~] = reduce_dim(ccdata{:,paycols}, 3, 'PCA');
[Pca{2}, eig{2}, ~] = reduce_dim(ccdata{:,bamtcols}, 3, 'PCA');
[Pca{3}, eig{3}, ~] = reduce_dim(ccdata{:,pamtcols}, 3, 'PCA');
f = subfig(1,5,5,'PCA Analysis');
for i = 1:3
  ax  = subplot(3,1,i);
  grid on;
  [Hp, axp] = pareto(ax, eig{i}, pclist);
  Hp(1).FaceColor ='Flat';
  %set(Hp(1),'FaceColor',[1 1 1]);
  set(Hp(1),'LineWidth',1);
  %Hp(1).CData(1,:) = [1 0 0];
  set(Hp(2),'LineWidth',3);
  title(vargrp(i));
  xlabel('Principal Component')
  ylabel('Variance Explained (%)')
end
%% Remove PAY, and BILLAMT colums, Leave PAYAMT
ccdata(:,[paycols, bamtcols]) = [];
% Replace PAY columns by first 2 PCs
ccdata.PAYPC1 = Pca{1}(:,1);
ccdata.PAYPC2 = Pca{1}(:,2);
% Replace BILLAMT columns by first PC
ccdata.BILLAMTPC1 = Pca{2}(:,1);
%  Move default back to last position
D = ccdata.DEFAULT;
ccdata.DEFAULT = [];
ccdata.DEFAULT = D;

clearvars paycols bamtcols pamtcols pclist pctitle Pca eig D;

fprintf('\nPCA analysis.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);


%% CREATE TEST AND TRAINING SETS
% ratio of training instances over total number
trainratio = 0.7;
% set the random seed so that the random permutation stays the same from one run to the next
rng(42);
% Using Matlab cvpartition function to maintain
% the same ratio of the target classes in both the train and test sets
cvpart  = cvpartition(ccdata.DEFAULT,'Holdout', 1-trainratio);
trainidx = training(cvpart);
testidx = ~ trainidx;
cc_train = ccdata(trainidx,2:end);  % Drop the ID  col
cc_test = ccdata(testidx,2:end);    % Drop the ID  col

%Initiate variables:
nbfeats = size(cc_train,2)-1;
pred_train = cc_train(:,1:nbfeats);
resp_train = cc_train(:,nbfeats+1); 
pred_test = cc_test(:,1:nbfeats);
resp_test = cc_test(:,nbfeats+1);

fprintf('\nTraining set and test sets created.\n');


%% LOOKING AT CORRELATIONS AGAIN
f = subfig(2,4,3,'Correlation Heatmap After Features Transformation');
img = plot_correlations(cc_train);
fprintf('\nCorrelation heatmap.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);

%% COLORS
logC = [0.27 0.50 0.70] %uisetcolor([0.27 0.50 0.70],'Select a color for LR')
rfC = [0.13 0.54 0.13] %uisetcolor([0.13 0.54 0.13],'Select a color for RF')

%% FIRST MODEL LOGISTIC REGRESSION

%Implement the logistic regression:
%LRbasis = 'linear'; %('linear' or 'quadratic')
%logModel = fitglm(cc_train,LRbasis,'Distribution','binomial','Link','logit');
% use the stepwise Matlab function to iteratively add quadratic terms to the initial linear model
dlgTitle    = 'Please Choose';
dlgQuestion = 'Do you wish to run stepwise regression? (Warning: takes 30 min)';
choice = questdlg(dlgQuestion,dlgTitle,'Yes','No', 'No');
%previously optimized formula
logForm = 'logit(DEFAULT) ~ 1 + LIMITBAL*AGE + LIMITBAL*PAYAMT1 + LIMITBAL*PAYAMT2 + LIMITBAL*PAYAMT5 + LIMITBAL*PAYPC1';
logForm = strcat(logForm,'+ SEX*MARRIAGE+ EDUCATION*PAYAMT1 + EDUCATION*PAYAMT2 + EDUCATION*PAYAMT4 + MARRIAGE*PAYAMT5');
logForm = strcat(logForm,'+ MARRIAGE*PAYPC1 + AGE*BILLAMTPC1 + PAYAMT1*PAYPC1 + PAYAMT2*PAYPC1 + PAYAMT3*PAYAMT4 + PAYAMT3*PAYPC1');
logForm = strcat(logForm,'+ PAYAMT3*PAYPC2 + PAYAMT4*PAYAMT5 + PAYAMT5*PAYAMT6 + PAYAMT6*PAYPC2 + PAYPC1*PAYPC2 + PAYPC1*BILLAMTPC1');
logForm = strcat(logForm,'+ PAYPC2*BILLAMTPC1 + EDUCATION^2 + PAYAMT1^2 + PAYAMT5^2 + PAYPC1^2 + PAYPC2^2 + BILLAMTPC1^2');
if strcmp(choice,'Yes')
    logModel = stepwiseglm(cc_train,'linear','upper', 'quadratic','Distribution','Binomial');
else
    logModel = fitglm(cc_train,logForm, 'Distribution','binomial','Link','logit');
end
logPred_train = predict(logModel, pred_train);
logPred_test = predict(logModel, pred_test);
logPredBinary_test = double(logPred_test > 0.5); %Assign 1 if probability >0.5

%% Calculate ROC curves
h = waitbar(0,'Calculating LR ROC Curves, Please wait...');
%[Xlog_train,Ylog_train,Tlog,AUClog_train] = perfcurve(resp_train{:,:},logPred_train,1); 
[Xlog_train,Ylog_train,Tlog,AUClog_train] = perfcurve(resp_train{:,:},logPred_train, 1,'NBoot',1000,'XVals',[0:0.05:1]);
waitbar(0.5,h);
%[Xlog_test,Ylog_test,~,AUClog_test] = perfcurve(resp_test{:,:},logPred_test,1);
[Xlog_test,Ylog_test,Tlog,AUClog_test] = perfcurve(resp_test{:,:},logPred_test, 1,'NBoot',1000,'XVals',[0:0.05:1]);
waitbar(1,h);
delete(h);
%% Error and confusion matrix: we need to compare to TP/ALL which is the
%accuracy of the 'never default' classifier
C = confusionmat(resp_test{:,:},logPredBinary_test);
logacc = (C(1,1)+C(2,2))/sum(sum(C)); %(TP+TN)/ALL 
logrec = C(2,2)/(C(2,1)+C(2,2));       %TP/(TP+FN) VERIFY FORMULAS
logpre = C(2,2)/(C(1,2)+C(2,2));       %TP/(TP+FP) VERIFY FORMULAS
logf1  = 2/(1/logpre + 1/logrec);      % Harmonic mean between precision and recall
fprintf('Accuracy of Logistic Regression model on test set: %.1f%%',logacc*100);

%% Plot regression  importances
imp = sortrows(logModel.Coefficients,'pValue','ascend');
imp = imp(:,4);
imp.pValue = -log(imp.pValue);
f = subfig(2,2,1,strcat('Linear Regression  - First 10 Inverse P-values'));
b = bar(imp{1:10,1},'FaceColor',logC);
b.FaceColor = 'flat';
%b.CData(1,:) = brighten(logC,0.5);
ylabel('-log(p-value)');
xlabel('Predictors');
h = gca;
h.XTick = 1:size(pred_train,2);
h.XTickLabel = imp.Properties.RowNames(1:10);
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';
grid on

fprintf('\nLogistic Regression implemented.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);

%% Implement a cross-validated regularised lasso regression
%[B,FitInfo] = lassoglm(pred_train{:,:}, resp_train{:,:},'binomial','NumLambda',25,'CV',10);
%f = subfig(2,3,2,'Lasso Regression Regularisation');
%ax = gca;
%lassoPlot(B,FitInfo,'PlotType','CV', 'Parent', ax);
%f = subfig(2,3,5,'Lasso Regression Regularisation');
%ax = gca;
%lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log', 'Parent', ax);
%nonzerospreads = sum(B(:,FitInfo.Index1SE) ~= 0);


%% SECOND MODEL RANDOM FOREST
prompt = 'Random Forest: enter number of trees (100 is optimal):';
nTrees = str2num(cell2mat(inputdlg(prompt)));

dlgTitle    = 'Please Choose';
dlgQuestion = 'Do you wish to run Bayesian Hyperparameter Optimisation? (This could take a while)';
choice = questdlg(dlgQuestion,dlgTitle,'Yes','No', 'No');
%Optimized values below
LeafSize = 46; %LeafSize = 46;
nbFeats = 11; %nbFeats = 11;
% Tuning the model - be careful, can take more than one hour
if strcmp(choice,'Yes')
    minLeafSize = 1;
    maxLeafSize = 300;
    rfLeafSize = optimizableVariable('rfLeafSize',[minLeafSize,maxLeafSize],'Type','integer');
    rfnbFeats = optimizableVariable('rfnbFeats',[1,size(pred_train,2)-1],'Type','integer');
    rfHyperparms = [rfLeafSize; rfnbFeats];
    results = bayesopt(@(params)oobErrRF(params,pred_train,resp_train, nTrees),rfHyperparms,...
    'AcquisitionFunctionName','expected-improvement-plus');
    % Fit the best model
    bestOOBErr = results.MinObjective;
    bestrfHyperparms = results.XAtMinObjective;
    LeafSize = bestrfHyperparms.rfLeafSize
    nbFeats = bestrfHyperparms.rfnbFeats
    rfModel = TreeBagger(nTrees,pred_train,resp_train,...
    'Method','regression','OOBPrediction','On','OOBPredictorImportance','on',...
    'MinLeafSize',LeafSize,'NumPredictorstoSample',nbFeats);
else
    rfModel = TreeBagger(nTrees,pred_train,resp_train,...
    'Method','regression','OOBPrediction','On','OOBPredictorImportance','on',...
    'MinLeafSize',LeafSize,'NumPredictorstoSample',nbFeats);   
end


%% Plot features importances
imp = rfModel.OOBPermutedPredictorDeltaError;
f = subfig(2,3,4,'Random Forest Features Importances');
b = bar(imp,'FaceColor',rfC);
b.FaceColor = 'flat';
%b.CData(12,:) = brighten(rfC,0.5);
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTick = 1:size(pred_train,2);
h.XTickLabel = rfModel.PredictorNames;
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';
grid on

%% Plot Classification error as a funtion of number of trees
f = subfig(2,3,5,'Random Forest Classification Error');
oobErrorBaggedEnsemble = oobError(rfModel);
plot(oobErrorBaggedEnsemble);
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% Calculate ROC curves
h = waitbar(0,'Calculating RF ROC Curves, Please wait');
rfPred_train = predict(rfModel, pred_train);
rfPred_test = predict(rfModel, pred_test);
rfPredBinary_test = double(rfPred_test > 0.5); %Assign 1 if probability >0.5

%[Xrf_train,Yrf_train,Trf_train,AUCrf_train] = perfcurve(resp_train{:,:},rfPred_train,1);
[Xrf_train,Yrf_train,Trf_train,AUCrf_train] = perfcurve(resp_train{:,:},rfPred_train, 1,'NBoot',1000,'XVals',[0:0.05:1]);
%[Xrf_test,Yrf_test,Trf_test,AUCrf_test] = perfcurve(resp_test{:,:},rfPred_test,1);
waitbar(0.5,h);
[Xrf_test,Yrf_test,Trf_test,AUCrf_test] = perfcurve(resp_test{:,:},rfPred_test,1,'NBoot',1000,'XVals',[0:0.05:1]);
waitbar(1,h);
delete(h);
%% Confusion matrix
C = confusionmat(resp_test{:,:},rfPredBinary_test);
rfacc = (C(1,1)+C(2,2))/sum(sum(C));
rfrec = C(2,2)/(C(2,1)+C(2,2));
rfpre = C(2,2)/(C(1,2)+C(2,2));
rff1  = 2/(1/rfpre + 1/rfrec);

fprintf('Accuracy of Random Forest on test set: %.1f%%',rfacc*100);

fprintf('\nRandom Forest implemented.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);


%% Plot the ROC curves:
% on Train Set
f = subfig(2,3,5,'ROCs on Training Set');
hold on 
curves = zeros(2,1); labels = cell(2,1);
%curves(1) = plot(Xlog_train,Ylog_train,'Color',[0.1,0.8,1],'LineWidth',2);
%curves(2) = plot(Xrf_train,Yrf_train,'Color',[0.4,0.4,1],'LineWidth',2);
curves(1) = errorbar(Xlog_train(:,1),Ylog_train(:,1),Ylog_train(:,1)-Ylog_train(:,2),Ylog_train(:,3)-Ylog_train(:,1),...
    'Color',logC,'LineWidth',2);
curves(2) = errorbar(Xrf_train(:,1),Yrf_train(:,1),Yrf_train(:,1)-Yrf_train(:,2),Yrf_train(:,3)-Yrf_train(:,1),...
    'Color',rfC,'LineWidth',2);
hline = refline(1,0);
hline.Color = 'r';
labels{1} = sprintf('Logistic Regression - AUC: [%.1f%% - %.1f%%]', AUClog_train(2)*100, AUClog_train(3)*100);
labels{2} = sprintf('Random Forest  - AUC: [%.1f%% - %.1f%%]', AUCrf_train(2)*100, AUCrf_train(3)*100);
legend(curves, labels, 'Location', 'SouthEast');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves Training Set');
hold off

% on Test Set
f = subfig(2,3,5,'ROCs on Test Set');
hold on;
curves = zeros(2,1); labels = cell(2,1);
%curves(1) = plot(Xlog_test,Ylog_test,'Color',[0.1,0.8,1],'LineWidth',2);
%curves(2) = plot(Xrf_test,Yrf_test,'Color',[0.4,0.4,1],'LineWidth',2);
curves(1) = errorbar(Xlog_test(:,1),Ylog_test(:,1),Ylog_test(:,1)-Ylog_test(:,2),Ylog_test(:,3)-Ylog_test(:,1),...
    'Color',logC,'LineWidth',2);
curves(2) = errorbar(Xrf_test(:,1),Yrf_test(:,1),Yrf_test(:,1)-Yrf_test(:,2),Yrf_test(:,3)-Yrf_test(:,1),...
    'Color',rfC,'LineWidth',2);
hline = refline(1,0);
hline.Color = 'r';
labels{1} = sprintf('Logistic Regression - AUC: [%.1f%% - %.1f%%]', AUClog_test(2)*100, AUClog_test(3)*100);
labels{2} = sprintf('Random Forest  - AUC: [%.1f%% - %.1f%%]', AUCrf_test(2)*100, AUCrf_test(3)*100);
legend(curves, labels, 'Location', 'SouthEast')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves - Test');
hold off;

fprintf('\nROC curves plotted.\n');
dlg = warndlg('Press Ok to continue');
waitfor(dlg);

%% LEARNING CURVES
AUC_train=[];
AUC_test =[];
nbtrials = size(pred_train,1);
X = [[200,300,400,500,600,800,900],[1000:1000:nbtrials]];

% RF calculations
h = waitbar(0,'Calculating RF Learning Curves');
for i = 1:size(X,2)
    Mdl = TreeBagger(nTrees,pred_train([1: X(i)],:),resp_train([1:X(i)],:),...
    'Method','regression','OOBPrediction','off','OOBPredictorImportance','off',...
    'MinLeafSize',LeafSize,'NumPredictorstoSample',nbFeats);   
    [~,~,~,AUC_train(i)] = perfcurve(resp_train{1: X(i),:},predict(Mdl, pred_train(1: X(i),:)),1);
    [~,~,~,AUC_test(i)] = perfcurve(resp_test{:,:},predict(Mdl, pred_test),1);
    waitbar(i/size(X,2),h);
end
delete(h);

% plot RF
%f = subfig(2,2,1,'Random Forest Learning Curves');
f = subfig(2,2,1,'Learning Curves');
hold on;
grid on
c1 = plot(X,AUC_train, 'Color', brighten(rfC,0.5),'LineWidth',2);
c2 = plot(X,AUC_test, 'Color', rfC,'LineWidth',2);
h = gca;
set(h,'xscale','log')
ylim([0.5 1]);
xlim([X(1) X(end)])
xlabel('Train Set Size (Log Scale)');
%xticklabels=
ylabel('AUC');
%leg = legend('Training RF', 'Test RF');
%legend('Location', 'SouthEast');
%hold off;

% LR Calculations 
h = waitbar(0,'Calculating LR Learning Curves');
for i = 1:size(X,2)
    Mdl = fitglm(cc_train(1: X(i),:),logModel.Formula, 'Distribution','binomial','Link','logit');
    [~,~,~,AUC_train(i)] = perfcurve(resp_train{1: X(i),:},predict(Mdl, pred_train(1: X(i),:)),1);
    [~,~,~,AUC_test(i)] = perfcurve(resp_test{:,:},predict(Mdl, pred_test),1);
    waitbar(i/size(X,2),h);
end
delete(h);

% plot LR
%f = subfig(2,2,3,'Logistic Regression Learning Curves');
%hold on;
%h = gca;
%set(h,'xscale','log')
c3 = plot(X,AUC_train, 'Color', brighten(logC,0.5),'LineWidth',2);
c4 = plot(X,AUC_test, 'Color', logC,'LineWidth',2);
%xlabel('Train Set Size (Log Scale)');
%ylim([0.5 1]);
%%xlim([X(1) X(end)])
ylabel('AUC');
leg = legend('Training RF', 'Test RF','Training LR', 'Test LR');
legend('Location', 'SouthEast');
hold off;

clearvars Mdl AUC_train AUC_test;

%% Summarize performance measurese
modelNames = {'LogisticR','TreeBagger'};
perfMeasures = {'Accuracy', 'Recall', 'Precision', 'f1 score', 'AUC'};
results = [logacc,rfacc;logrec,rfrec;logpre,rfpre;logf1,rff1;...
    AUClog_test(1)*100,AUCrf_test(1)*100];
restable= array2table(results,'RowNames',perfMeasures,'VariableNames',modelNames);
disp(restable);

%% SAVE CURRENT MODELS DATA
f = strcat('ccdata_',replace(datestr(datetime),':','|'),'.txt');
fid = fopen(f,'w');
writeline(fid,strcat(evalc('logModel')));
writeline(fid,strcat(evalc('rfModel')));
writeline(fid,strcat(evalc('restable')));
fclose(fid);
disp('Data dumped');


%% FUNCTIONS

function stats = calcstats(datatable)
    varnames = datatable.Properties.VariableNames;
    % define list of stats required
    statlist = [{'min'}, {'max'}, {'mean'},{'@median'}, {'std'},{'@(x)prctile(x,25)'},{'@(x)prctile(x,75)'},{'@skewness'}];
    % iterate and build statstable
    for i = 1:length(statlist)
        stat= grpstats(datatable,[],statlist(i));
        stat.Properties.VariableNames = [{'COUNT'} varnames];
        stat.Properties.RowNames = statlist(i);
        if i == 1
            stats = stat;
        else
            stats = [stats ; stat];
        end
    end
end

function t = display_stats(f, stats)
    t = uitable(f, 'Data',stats{:,:}, 'Position',[20, 20, 1300, 200]);
    t.ColumnName = stats.Properties.VariableNames;
    t.RowName = stats.Properties.RowNames;
    t.HandleVisibility = 'on';
end

function img =  plot_correlations(data)
    C = cov([data{:,:}]);   %Covariance matrix
    R = corrcov(C);        %Correlation matrix
    L = data.Properties.VariableNames;
    n = size(R,1);
    h = gca;
    % Display as Heatmap
    img = imagesc(R); % plot the matrix
    set(h, 'XTick', 1:n); % center x-axis ticks on bins
    set(h, 'YTick', 1:n); % center y-axis ticks on bins
    set(h, 'XTickLabel', L); % set x-axis labels
    set(h, 'YTickLabel', L); % set y-axis labels
    pos = ylim;
    rotate_xlabels(h, 90, pos(2)); 
    colormap('jet'); % set the colorscheme
    colorbar(h); % enable colorbar
end


function oobErr = oobErrRF(params,X, Y, nbtrees)
    rForest = TreeBagger(nbtrees,X, Y,'OOBPrediction','On','Method','regression',...
    'MinLeafSize',params.rfLeafSize,'NumPredictorstoSample',params.rfnbFeats);
    oobErr = oobQuantileError(rForest);
    % do not try this -> oobErr = oobError(rForest);
end


function res = norm_data(array, method)
    switch method
        case 'zscore'
            res = zscore(array);
        case 'logzscore'
            res = zscore(log(array));
        otherwise
            res = array;
    end
end

function [reducedarray, eig, vectors] = reduce_dim(array, dims, method)
    switch method
        case 'MDS'
        	D = pdist(array);
            [reducedarray,eig] = mdscale(array,dims);
        case 'PCA'
        	[vectors,reducedarray,~,~,eig] = pca(array);
        otherwise
        	[reducedarray, eig, vectors] = [array, 0, []];
    end
end

%function to simplify drawing on single-window figures
function f = subfig(r,c,x, figTitle)
    screensize = get(0, 'Screensize');
    pad = 1;
    fw = round(pad * screensize(3) / c);
    fh = round(pad * screensize(4) / r);
    if (x>=1) & (x<= r*c)
        fx = mod(x-1,c) * fw;
        fy = (r - floor((x - mod(x-1,c))/c) - 1 * (c~=1)) * fh;
        f = figure('OuterPosition',[fx,fy,fw,fh],'Name',figTitle,'NumberTitle','off');
    else
        dispstr = strcat('Figure index out of range[1,',int2str(r*c),']');
        f = figure('Name',dispstr);
    end
end


function rotate_xlabels(h,rot,y)  %function from Copyright 2005 by Andy Bliss
    a=get(h,'XTickLabel');  %erase current tick labels from figure
    set(h,'XTickLabel',[]);  %get tick label positions
    b=get(h,'XTick');
    c=get(h,'YTick');
    %make new tick labels
    th=text(b,repmat(y,length(b),1),a,'HorizontalAlignment','right','rotation',rot);
end

function writeline(fid, cellarray)
    cellarray = replace(cellarray,'<strong>','');
    cellarray = replace(cellarray,'</strong>','');
    for r=1:size(cellarray,1);
        fprintf(fid,'%s\n',cellarray(r,:));
    end
end

function tp = progress_bar(i, imax, scale);
    maxl = 20;
    if scale == 'log'
        pl = round(maxl * exp(i/imax))
    else
        pl = round(maxl * i/imax)
    end
    tp = strcat(repmat('|',1, pl), repmat('_',1, maxl-pl))
end
