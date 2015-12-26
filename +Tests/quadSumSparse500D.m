%% Initialise

clear
close all
clc

%% Objective function and its gradients

% Load coefficients (all Qa_i are symmetric)
load +Tests/Qa500D

% Store the number of decision variables
nDecVar = size(Qa{1}, 1) - 1;

% Store the number of addends in the stochastic objective
nQa = length(Qa);

% Define the objective function
objFun = @(x) 0.5*([x', 1]*QaAvg*[x; 1]);

% Define the full gradient and the stochastic gradient functions
grad = @(x) ([x; 1]'*QaAvg(:, 1 : 1 : end - 1))';
gradStoch = @(i, x) ([x; 1]'*Qa{i}(:, 1 : 1 : end - 1))';

%% Perform optimisation

% Adamax has problems because of missing gradients: 0/0 = NaN

x0 = ones(nDecVar, 1);
nIter = 500;
idxSG = randi(nQa, 1, nIter);

solvers = {'Adam', 'Adamax', 'AdaGrad', 'AdaGradDecay', 'VanillaSGD'};

xMat.Adam = Adam(gradStoch, x0, 1e-1, idxSG, nIter, 0.8, 0.999);
xMat.Adamax = Adamax(gradStoch, x0, 1e-1, idxSG, nIter, 0.9, 0.999);
xMat.AdaGrad = AdaGrad(gradStoch, x0, 1e-1, idxSG, nIter);
xMat.AdaGradDecay = AdaGradDecay(gradStoch, x0, 1e-1, idxSG, nIter, 0.9);
xMat.VanillaSGD = VanillaSGD(gradStoch, x0, 1e-6, idxSG, nIter);

for i = 1 : 1 : length(solvers)
    objFunMat.(solvers{i}) = ...
        cellfun(objFun, num2cell(xMat.(solvers{i}), 1));
end

%% Plot results -- Convergence plot

figConvergence = figure( ...
    'Name', 'Convergence behaviour of different solvers');
for i = 1 : 1 : length(solvers)
    plot(objFunMat.(solvers{i}));
    hold on
end
hold off
legend(solvers);
