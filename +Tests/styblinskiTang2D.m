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

% Store the global minimum
objFunGlobMin = -78.3323;

%% Compute the objective function values (for plotting)

rangeX = linspace(-4, +4, 100);
rangeY = linspace(-4, +4, 100);

[X, Y] = meshgrid(rangeX, rangeY);

Z = zeros(size(X));

for i = 1 : 1 : length(rangeX)
    for j = 1 : 1 : length(rangeY)
        Z(j, i) = objFun([rangeX(i); rangeY(j)]);
    end
end

%% Perform optimisation
% WARNING: Adamax can go NaN if `x0` contains zeros

% Depending on `idxSG`, solutions can converge in either minima in the
% negative x1 half-plane.
x0 = [-1; 0.2];
nIter = 800;
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

xMat.VanillaSGD = VanillaSGD(gradStoch, x0, nIter, idxSG, 1.5e-3);
xMat.MomentumSGD = MomentumSGD(gradStoch, x0, nIter, idxSG, 1.5e-3, 0.5);
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
plot([1, nIter + 1], [1, 1].*objFunGlobMin, ...
    'Color', 'r', 'LineStyle', '--');
hold off
legend([solvers, {'Global minimum'}]);
xlim([1, nIter + 1]);

%% Plot results -- Contour plot of the objective function

figContour = figure('Name', 'Contour plot of the objective function');

surf(X, Y, Z, 'EdgeAlpha', 0.1);
colormap bone
xlabel('x');
ylabel('y');
xlim(rangeX([1, end]));
ylim(rangeY([1, end]));

hold on
for i = 1 : 1 : length(solvers)
    plot3(xMat.(solvers{i})(1, :), xMat.(solvers{i})(2, :), ...
        objFunMat.(solvers{i}), ...
        'LineWidth', 1.4);
end
plot3(-2.903534, -2.903534, objFunGlobMin, 'o', ...
    'Color', 'y');
hold off
legend([{'Obj fun'}, solvers, {'Global minimum'}]);
view([0, 90]);
