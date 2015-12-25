function xMat = AdaGradDecay(sg, x0, stepSize, idxSG, nIter, beta, epsilon)
%ADAGRADDECAY AdaGrad algorithm with exp. decay of historical gradients
%
% Implemented according to [2] and [3].
%
% Decision variable `x` is a column vector.
%
% Function handle `sg` to the stochastic gradient accepts the index of the
% stochastic gradient as the first argument and the value of the decision
% variable as the second argument, i.e. `sg(idx, x)`.
%
% References:
%   [1] Duchi, John, Hazan, Elad and Singer, Yoram. Adaptive Subgradient
%       Methods for Online Learning and Stochastic Optimization.
%   [2] Dyer, Chris. Notes on AdaGrad.
%   [3] xcorr.net: AdaGrad – Eliminating learning rates in stochastic
%       gradient descent. http://xcorr.net/2014/01/23/adagrad-
%       eliminating-learning-rates-in-stochastic-gradient-descent/
%
% Input:
%   sg       : function handle to the stochastic gradient
%   x0       : initial guess for the decision variables
%   stepSize : scalar step size
%   idxSG    : indices of the gradients to use
%   nIter    : number of iterations to perform
%   beta     : exponential decay rate for historical gradients
%   epsilon  : back-to-numerical-reality addend, default: `sqrt(eps)`
%
% Output:
%   xMat     : matrix with decision variables at each iteration step
%
% See also: ADAGRAD
%

% Store default value for `epsilon` if there are only 6 input arguments
if nargin == 6
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

% First iteration outside of the loop for better initialisation of
% historical gradients

% Get gradients w.r.t. stochastic objective at the current iteration
sgCurr = sg(idxSG(:, 1), xMat(:, 1));

% Initialise historical gradients
sgHist = sgCurr.^2;

% Update decision variables
xMat(:, 2) = xMat(:, 1) - stepSize.*sgCurr./(sqrt(sgHist) + epsilon);

% Run the rest of optimisation
for i = 2 : 1 : nIter
    % Get gradients w.r.t. stochastic objective at the current iteration
    sgCurr = sg(idxSG(:, i), xMat(:, i));
    
    % Update historical gradients
    sgHist = beta.*sgHist + (1 - beta).*sgCurr.^2;
    
    % Update decision variables
    xMat(:, i + 1) = xMat(:, i) - ...
        stepSize.*sgCurr./(sqrt(sgHist) + epsilon);
end

end
