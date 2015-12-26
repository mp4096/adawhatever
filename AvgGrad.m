function avgSG = AvgGrad(sg, idx, x)
%AVGGRAD Compute an average of stochastic gradients
%
% Input:
%   sg    : function handle to the stochastic gradient
%   idx   : indices of the gradients to use, column vector
%   x     : value of the decision parameters
%
% Output:
%   avgSG : averaged stochastic gradient
%

avgSG = zeros(length(x), length(idx));

for i = 1 : 1 : length(idx)
    avgSG(:, i) = sg(idx(i), x);
end

avgSG = mean(avgSG, 2);

end
