function xMat = Adamax(sg, x0, stepSize, idxSG, nIter, beta1, beta2)
%ADAMAX Adamax algorithm for SGD optimisation (Kingma & Ba, 2015)
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
%	[1]  Kingma, Diederik and Ba, Jimmy. Adam: A Method for Stochastic
%        Optimization. arXiv preprint: http://arxiv.org/abs/1412.6980
%
% Input:
%   sg       : function handle to the stochastic gradient
%   x0       : initial guess for the decision variables
%   stepSize : scalar step size
%   idxSG    : indices of the gradients to use
%   nIter    : number of iterations to perform
%   beta1    : exponential decay rate for the 1st moment estimate
%   beta2    : exp. decay rate for the exp. weighted infinity norm
%
% Output:
%   xMat     : matrix with decision variables at each iteration step
%

% Store the number of decision variables
nDecVar = length(x0);

% Allocate output
xMat = zeros(nDecVar, nIter + 1);

% Set the initial guess
xMat(:, 1) = x0;

% Repeat `idxSG` if it has fewer than `nIter` elements
if length(idxSG) < nIter
    idxSG = repmat(idxSG(:), ceil(nIter/length(idxSG)), 1);
    idxSG(nIter + 1 : 1 : end) = [];
end

% Initialise moment estimate and the exponentially weighted infinity norm
mOld = zeros(nDecVar, 1);
uOld = zeros(nDecVar, 1);

% Run optimisation
for i = 1 : 1 : nIter
    % Get gradients w.r.t. stochastic objective at the current iteration
    sgCurr = sg(idxSG(i), xMat(:, i));
    
    % Update biased 1st moment estimate
    mCurr = beta1.*mOld + (1 - beta1).*sgCurr;
    % Update the exponentially weighted infinity norm
    uCurr = max(beta2.*uOld, abs(sgCurr));
    
    % Compute bias-corrected 1st moment estimate
    mHatCurr = mCurr./(1 - beta1^i);
    
    % Update decision variables
    xMat(:, i + 1) = xMat(:, i) - stepSize.*mHatCurr./uHatCurr;
    
    % Shift the moment estimates
    mOld = mCurr;
    uOld = uCurr;
end

end
