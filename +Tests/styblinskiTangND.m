%% Initialise

clear
close all
clc

%% Objective function and its gradients

% Store the number of addends in the stochastic objective
nQa = 3;

% Define the objective function
objFun = @Tests.StyblinskiTang.ObjFun;

% Define the full gradient and the stochastic gradient functions
gradStoch = @Tests.StyblinskiTang.StochGrad;

%% Perform optimisation
% WARNING: Adamax can go NaN if `x0` contains zeros
x0 = [-1; 0.2; 0.1; 2; 1; 2; 3];
nIter = 1500;
idxSG = randi(nQa, 1, nIter);

solvers = { ...
    'VanillaSGD', ...
    'AdaGrad', ...
    'AdaGradDecay', ...
    'Adadelta', ...
    'Adam', ...
    'Adamax', ...
    };

xMat.VanillaSGD = VanillaSGD(gradStoch, x0, nIter, idxSG, 1.5e-3);
xMat.AdaGrad = AdaGrad(gradStoch, x0, nIter, idxSG, 0.1);
xMat.AdaGradDecay = AdaGradDecay(gradStoch, x0, nIter, idxSG, 0.05, 0.9);
xMat.Adadelta = Adadelta(gradStoch, x0, nIter, idxSG, 0.95);
xMat.Adam = Adam(gradStoch, x0, nIter, idxSG, 0.05, 0.9, 0.999);
xMat.Adamax = Adamax(gradStoch, x0, nIter, idxSG, 0.05, 0.9, 0.999);

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

plot([1, nIter + 1], [1, 1].*(-39.166165).*length(x0), ...
    'Color', 'r', 'LineStyle', '--');
hold off
legend([solvers, {'Global minimum'}]);
xlim([1, nIter + 1]);
