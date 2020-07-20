clear all
clc

% LOAD Dataset
path='Data\';
% the different datasets
datasets={'e','ic','gpcr','nr'};
ds = 4; % Nuclear Receptor dataset
[Y,Sd,St,~,~]=getdata(path,datasets{ds});
Y = Y';

% LOAD WANG Dataset
% load 'Data/WANG'

% CV parameters
m = 5;  % number of n-fold experiments (repetitions)
n = 10; % the 'n' in "n-fold experiment"

% compute Laplacian matrices
Dd = diag(sum(Sd));
Ld = Dd - Sd;
Dt = diag(sum(St));
Lt = Dt - St;

% choose the cross validation setting, cv_drug or cv_target
cv_setting = 'cv_target';

% parameters for cv_drug
p = 1;
r = 0.001;
c1 = 0.01;
c2 = 0.0001;
K = 7;

% fixed seeds (for reproducibility)
% cv_drug setting: all drugs (drug prediction)
if strcmp(cv_setting, 'cv_drug')
    seeds = [6381  1535  9727  9129  6802 ]; 

% cv_target: all targets (target prediction)
else 
    seeds = [8763  7952  402  9781  9541 ];  
end

% cross validation (m repetitions of n-fold experiments)
AUPR = zeros(1,m);
for k=1:m
    seed = seeds(k);
    [num_drugs,num_targets] = size(Y);
    if strcmp(cv_setting,'cv_drug')
        len = num_drugs;
    else
        len = num_targets;
    end
    rng('default')
    rng(seed);
    rand_ind = randperm(len);
    
    AUPRs = zeros(1,n);
    Y_final = zeros(size(Y));
    
    % loop over the n folds
    for i=1:n
        % cv_drug: leave out random entire drugs
        if strcmp(cv_setting,'cv_drug')
            left_out_drugs = rand_ind((floor((i-1)*len/n)+1:floor(i*len/n))');
            test_ind = zeros(length(left_out_drugs),num_targets);
            for j=1:length(left_out_drugs)
                curr_left_out_drug = left_out_drugs(j);
                test_ind(j,:) = ((0:(num_targets-1)) .* num_drugs) + curr_left_out_drug;
            end
            test_ind = reshape(test_ind,numel(test_ind),1);
            left_out = left_out_drugs;

        % cv_target: leave out random entire targets
        else
            left_out_targets = rand_ind((floor((i-1)*len/n)+1:floor(i*len/n))');
            test_ind = zeros(num_drugs,length(left_out_targets));
            for j=1:length(left_out_targets)
                curr_left_out_target = left_out_targets(j);
                test_ind(:,j) = (1:num_drugs)' + ((curr_left_out_target-1)*num_drugs);
            end
            test_ind = reshape(test_ind,numel(test_ind),1);
            left_out = left_out_targets;
        end
        left_out = left_out(:);
        test_ind = test_ind(:);

        % predict with test set being left out
        y2 = Y;
        y2(test_ind) = 0;   % test set = ZERO
       
        % preprocess step
        y2 = preprocess_KNNC(y2,Sd,St,K);
        trIndex = double(y2~=0);
        % predict 
        y3=LRSpNM(y2,Ld,Lt,trIndex,p,r,c1,c2);
        
        % compute evaluation metrics based on obtained prediction scores
        AUPRs(i) = calculate_aupr(y3(test_ind),Y(test_ind));
        Y_final(test_ind) = y3(test_ind);
    end 
    
    % average  AUPRs of the different folds
    AUPR(k) = mean(AUPRs);     
    fprintf('\n AUPR: %g\t\n',AUPR(k));
end
    
% display evaluation results
fprintf('\n FINAL AVERAGED RESULTS\n\n');
fprintf('    AUPR (std): %g\t(%g)\n',   mean(AUPR), std(AUPR)); 
