function xMat = Adamax(sg, x0, nIter, idxSG, stepSize, beta1, beta2)
%ADAMAX Adamax algorithm for SGD optimisation (Kingma & Ba, 2015)
%
% Implemented according to preprint 1412.6980v8, 23 July 2015.
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
% WARNING: Adamax can have problems with gradients initialisation if some
% gradient components are exactly zero in the first iteration, leading to
% `NaN`-s in the estimates vector. If you have such (often sparse)
% gradients, try to initialise `u` with a very small strictly positive
% value, e.g. `u = ones(nDecVar, 1).*realmin;`.
%
% Refer to [1] for a description of solver parameters `stepSize`, `beta1`
% and `beta2`.
%
% References:
%   [1] Kingma, Diederik and Ba, Jimmy. Adam: A Method for Stochastic
%       Optimization. arXiv preprint: http://arxiv.org/abs/1412.6980
%
% Input:
%   sg       : function handle to the stochastic gradient
%   x0       : initial guess for the decision variables
%   nIter    : number of iterations to perform
%   idxSG    : indices of the gradients to use
%   stepSize : scalar step size
%   beta1    : exponential decay rate for the 1st moment estimate
%   beta2    : exp. decay rate for the exp. weighted infinity norm
%
% Output:
%   xMat     : matrix with decision variables at each iteration step
%
% See also: ADAM
%

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

% Initialise moment estimate and the exponentially weighted infinity norm
m = zeros(nDecVar, 1);
u = zeros(nDecVar, 1);

% Run optimisation
for i = 1 : 1 : nIter
    % Get gradients w.r.t. stochastic objective at the current iteration
    sgCurr = sg(idxSG(:, i), xMat(:, i));
    
    % Update biased 1st moment estimate
    m = beta1.*m + (1 - beta1).*sgCurr;
    % Update the exponentially weighted infinity norm
    u = max(beta2.*u, abs(sgCurr));
    
    % Compute the bias-corrected 1st moment estimate
    mHat = m./(1 - beta1^i);
    
    % Update decision variables
    xMat(:, i + 1) = xMat(:, i) - stepSize.*mHat./u;
end

end
