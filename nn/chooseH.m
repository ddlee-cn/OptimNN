function [H] = chooseH(s,y)
    H = (s'*y)/(y'*s)*eye(length(s));
end
