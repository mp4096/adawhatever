function res = VanillaSGD(fun, sg, w0, stepSize, idxSG, nIter)
%VANILLASGD Vanilla stochastic gradient descent solver
%
% Input:
%   fun      : function handle to the objective function
%   sg       : function handle to the stochastic gradient
%   w0       : initial guess for the decision variables
%   stepSize : step size, scalar or a function handle
%   idxSG    : indices of the gradients to use
%   nIter    : number of iterations to perform
%

w = zeros(length(w0), nIter + 1);

% Check if `stepSize` is a scalar, convert it to a function if it is one
if isscalar(stepSize)
    stepSize = @(i) stepSize;
end

% Repeat `idxSG` if it has fewer than `nIter` elements
if length(idxSG) < nIter
    idxSG = repmat(idxSG(:), ceil(nIter/length(idxSG)), 1);
    idxSG(nIter + 1 : 1 : end) = [];
end

% Run optimisation
for i = 1 : 1 : nIter
    w(:, i + 1) = w(:, i) - stepSize(i).*sg(idxSG(i), w(:, i));
end

% Save results
res.w = w;

end
