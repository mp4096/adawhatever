function sg = StochGrad(idx, x)
%STOCHGRAD Stochastic gradient of the Styblinski--Tang test function
%
% References:
%   [1] : https://en.wikipedia.org/wiki/Test_functions_for_optimization
%
% Input:
%   idx : index of the gradient, 1..3
%   x   : current decision variables (column or row vector)
% Output:
%   sg  : stochastic gradient, column vector
%
% See also: TESTS.STYBLINSKITANG.OBJFUN
%

switch idx
    case 1
        sg = 2.*x(:).^3;
    case 2
        sg = -16.*x(:);
    case 3
        sg = ones(length(x), 1).*2.5;
    otherwise
        error('Stochastic gradient index out of bounds!');
end

end
