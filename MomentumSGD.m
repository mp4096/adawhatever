function xMat = MomentumSGD(sg, x0, nIter, idxSG, stepSize, beta)
%MOMENTUMSGD Momentum stochastic gradient descent solver
%
% This function minimises an objective function `J(x)`, where `x` is an
% n-dimensional column vector containing decision variables. The stochastic
% gradient of the objective is supplied as the function handle `sg`, which
% accepts the index (or indices) of the stochastic gradient as the first
% argument and the value of the decision variable as the second argument,
% i.e. `sg(idx, x)`. `sg` returns an n-dimensional column vector.
%
% Note that both `idx` and `x` must be column vectors. If `idx` is a
% vector, function `sg` should return the averaged stochastic gradient. You
% can use the `AvgGrad` wrapper provided in this repo to do the averaging
% without additional effort.
%
% `idxSG` is a row vector or a matrix which columns specify the indices of
% the stochastic gradient should be used at each iteration. If `idxSG` has
% fewer columns than `nIter`, it is repeating to the required size.
%
% Normally, one would generate `idxSG` with `randi`, e.g. `idxSG =
% randi(<maxIdx>, 1, nIter);`.
%
% References:
%   [1] Zeiler, Matthew D. Adadelta: An Adaptive Learning Rate Method.
%   arXiv preprint: http://arxiv.org/abs/1212.5701
%   [2] Hinton, Geoffrey with Srivastava, Nitish and Swersky, Kevin. Neural
%   Networks for Machine Learning, Lecture 6c: The momentum method.
%
% Input:
%   sg       : function handle to the stochastic gradient
%   x0       : initial guess for the decision variables
%   nIter    : number of iterations to perform
%   idxSG    : indices of the gradients to use
%   stepSize : scalar step size
%   beta     : exponential decay rate for historical updates
%
% Output:
%   xMat     : matrix with decision variables at each iteration step
%

% Allocate output
xMat = zeros(length(x0), nIter + 1);

% Set the initial guess
xMat(:, 1) = x0;

% Repeat `idxSG` if it has fewer columns than `nIter`
if size(idxSG, 2) < nIter
    idxSG = repmat(idxSG, 1, ceil(nIter/size(idxSG, 2)));
    idxSG(:, nIter + 1 : 1 : end) = [];
end

% Initialise the updates variables
upd = zeros(length(x0), 1);

% Run optimisation
for i = 1 : 1 : nIter
    upd = beta.*upd - stepSize.*sg(idxSG(:, i), xMat(:, i));
    
    xMat(:, i + 1) = xMat(:, i) + upd;
end

end
