function xMat = Adadelta(sg, x0, nIter, idxSG, beta, epsilon)
%ADADELTA Adadelta algorithm for SGD optimisation (Zeiler, 2012)
%
% Implemented according to preprint 1212.5701v1, 22 Dec 2012.
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
% the stochastic gradient to be use at each iteration. If `idxSG` has fewer
% columns than `nIter`, it is repeating to the required size.
%
% Normally, one would generate `idxSG` with `randi`, e.g. `idxSG =
% randi(<maxIdx>, 1, nIter);`.
%
% Refer to [1] for a description of solver parameters `delta` and
% `epsilon`.
%
% References:
%   [1] Zeiler, Matthew D. Adadelta: An Adaptive Learning Rate Method.
%   arXiv preprint: http://arxiv.org/abs/1212.5701
%
% Input:
%   sg       : function handle to the stochastic gradient
%   x0       : initial guess for the decision variables
%   nIter    : number of iterations to perform
%   idxSG    : indices of the gradients to use
%   beta     : exponential decay rate for moving averages
%   epsilon  : back-to-numerical-reality addend, default: `sqrt(eps)`
%
% Output:
%   xMat     : matrix with decision variables at each iteration step
%

% Store default value for `epsilon` if there are only 5 input arguments
if nargin == 5
    epsilon = 1.0e-6;
end

% Store the number of decision variables
nDecVar = length(x0);

% Allocate output
xMat = zeros(nDecVar, nIter + 1);

% Set the initial guess
xMat(:, 1) = x0;

% Repeat `idxSG` if it has fewer columns than `nIter`
if size(idxSG, 2) < nIter
    idxSG = repmat(idxSG, 1, ceil(nIter/size(idxSG, 2)));
    idxSG(:, nIter + 1 : 1 : end) = [];
end

% Initialise accumulator variables
accG = zeros(nDecVar, 1); % accumulated gradients
accD = zeros(nDecVar, 1); % accumulated updates (deltas)

% Run optimisation
for i = 1 : 1 : nIter
    % Get gradients w.r.t. stochastic objective at the current iteration
    sgCurr = sg(idxSG(:, i), xMat(:, i));
    
    % Update accumulated gradients
    accG = beta.*accG + (1 - beta).*(sgCurr.^2);
    
    % Compute update
    dCurr = -(sqrt(accD + epsilon)./sqrt(accG + epsilon)).*sgCurr;
    
    % Update accumulated updates (deltas)
    accD = beta.*accD + (1 - beta).*(dCurr.^2);
    
    % Update decision variables
    xMat(:, i + 1) = xMat(:, i) + dCurr;
end

end
