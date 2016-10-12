clear all
clc
format long;
format compact;
% mex cec13_func.cpp -DWINDOWS
func_num=1;
D= 30;
Xmin= -100;
Xmax= 100;
pop_size= 40;
FEs_max = 10000*D;
iter_max= 10000*D/pop_size;

runs=51;
fhd=str2func('cec13_func');

for i=1:28
    func_num=i;
	GET_GOAL_VALUE;
	
	for run_num=1:runs
        func_num,run_num,
        [gbest, gbestval, FES, CG] = SPSO2007(fhd, D, pop_size, func_num);
        xbest(run_num,:) = gbest;
		result_CG(run_num,1:FEs_max) = CG(1,1:FEs_max);
		result_error_value = gbestval - goal_value;
        fbest(i,run_num) = gbestval;
        fbest(i,run_num) - goal_value
    end
    f_mean(i)=mean(fbest(i,:));
	filename = sprintf('SPSO2007%dfunc%druns_%diterations.mat',func_num,runs,iter_max);
	save(filename,'-mat');
end



