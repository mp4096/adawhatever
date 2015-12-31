function J = ObjFun(x)
%OBJFUN Styblinski--Tang test function
%
% References:
%   [1] : https://en.wikipedia.org/wiki/Test_functions_for_optimization
%
% Input:
%   x : decision variables (column or row vector)
% Output:
%   J : value of the objective function
%
% See also: TESTS.STYBLINSKITANG.STOCHGRAD
%

J = sum(x.^4 - 16.*x.^2 + 5.*x)./2;

end
