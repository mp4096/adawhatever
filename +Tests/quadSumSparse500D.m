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

%% Perform optimisation -- Purely stochastic gradient

x0 = ones(nDecVar, 1);
nIter = 1000;
idxSG = randi(nQa, 1, nIter);

solvers = { ...
    'VanillaSGD', ...
    'MomentumSGD', ...
    'AdaGrad', ...
    'AdaGradDecay', ...
    'Adadelta', ...
    'Adam', ...
    'Adamax', ...
    };

xMat.VanillaSGD = VanillaSGD(gradStoch, x0, nIter, idxSG, 1e-1);
xMat.MomentumSGD = MomentumSGD(gradStoch, x0, nIter, idxSG, 1e-1, 0.5);
xMat.AdaGrad = AdaGrad(gradStoch, x0, nIter, idxSG, 1e-1);
xMat.AdaGradDecay = AdaGradDecay(gradStoch, x0, nIter, idxSG, 1e-1, 0.9);
xMat.Adadelta = Adadelta(gradStoch, x0, nIter, idxSG, 0.95);
xMat.Adam = Adam(gradStoch, x0, nIter, idxSG, 1e-1, 0.9, 0.999);
xMat.Adamax = Adamax(gradStoch, x0, nIter, idxSG, 1e-1, 0.9, 0.999);


for i = 1 : 1 : length(solvers)
    objFunMat.(solvers{i}) = ...
        cellfun(objFun, num2cell(xMat.(solvers{i}), 1));
end

%% Plot results -- Convergence plot -- Purely stochastic gradient

figure
ax(1) = subplot(1, 2, 1);
for i = 1 : 1 : length(solvers)
    plot(objFunMat.(solvers{i}));
    hold on
end
hold off
grid on
legend(solvers);
title(['Convergence behaviour of different solvers, ', ...
    'stochastic gradient only'])

%% Perform optimisation -- Averaged stochastic gradient

% Average over 10 random gradients
idxSG = randi(nQa, 10, nIter);

avgSG = @(idx, x) AvgGrad(gradStoch, idx, x);

xMat.VanillaSGD = VanillaSGD(avgSG, x0, nIter, idxSG, 1e-1);
xMat.AdaGrad = AdaGrad(avgSG, x0, nIter, idxSG, 1e-1);
xMat.AdaGradDecay = AdaGradDecay(avgSG, x0, nIter, idxSG, 1e-1, 0.9);
xMat.Adadelta = Adadelta(avgSG, x0, nIter, idxSG, 0.95);
xMat.Adam = Adam(avgSG, x0, nIter, idxSG, 1e-1, 0.9, 0.999);
xMat.Adamax = Adamax(avgSG, x0, nIter, idxSG, 1e-1, 0.9, 0.999);


for i = 1 : 1 : length(solvers)
    objFunMat.(solvers{i}) = ...
        cellfun(objFun, num2cell(xMat.(solvers{i}), 1));
end

%% Plot results -- Convergence plot -- Averaged stochastic gradient

ax(2) = subplot(1, 2, 2);
for i = 1 : 1 : length(solvers)
    plot(objFunMat.(solvers{i}));
    hold on
end
hold off
grid on
legend(solvers);
title(['Convergence behaviour of different solvers, ', ...
    'averaged stochastic gradient']);

linkaxes(ax, 'xy');
xlim([1, nIter + 1]);
