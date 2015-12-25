function xMat = Adam(sg, x0, stepSize, idxSG, nIter, beta1, beta2, epsilon)
%ADAM Adam algorithm for SGD optimisation (Kingma & Ba, 2015)
%
% Implemented according to preprint 1412.6980v8, 23 July 2015.
%
% Decision variable `x` is a column vector.
%
% Function handle `sg` to the stochastic gradient accepts the index of the
% stochastic gradient as the first argument and the value of the decision
% variable as the second argument, i.e. `sg(idx, x)`.
%
% References:
%   [1] Kingma, Diederik and Ba, Jimmy. Adam: A Method for Stochastic
%       Optimization. arXiv preprint: http://arxiv.org/abs/1412.6980
%
% Input:
%   sg       : function handle to the stochastic gradient
%   x0       : initial guess for the decision variables
%   stepSize : scalar step size
%   idxSG    : indices of the gradients to use
%   nIter    : number of iterations to perform
%   beta1    : exponential decay rate for the 1st moment estimate
%   beta2    : exponential decay rate for the 2nd moment estimate
%   epsilon  : back-to-numerical-reality addend, default: `sqrt(eps)`
%
% Output:
%   xMat     : matrix with decision variables at each iteration step
%
% See also: ADAMAX
%

% Store default value for `epsilon` if there are only 7 input arguments
if nargin == 7
    epsilon = sqrt(eps);
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

% Initialise moment estimates
m = zeros(nDecVar, 1);
v = zeros(nDecVar, 1);

% Run optimisation
for i = 1 : 1 : nIter
    % Get gradients w.r.t. stochastic objective at the current iteration
    sgCurr = sg(idxSG(:, i), xMat(:, i));
    
    % Update biased 1st moment estimate
    m = beta1.*m + (1 - beta1).*sgCurr;
    % Update biased 2nd raw moment estimate
    v = beta2.*v + (1 - beta2).*(sgCurr.^2);
    
    % Compute bias-corrected 1st moment estimate
    mHat = m./(1 - beta1^i);
    % Compute bias-corrected 2nd raw moment estimate
    vHat = v./(1 - beta2^i);
    
    % Update decision variables
    xMat(:, i + 1) = xMat(:, i) - stepSize.*mHat./(sqrt(vHat) + epsilon);
end

end
