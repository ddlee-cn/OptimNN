function [xn,exitflag] = linesearch(fun, x0, f0, g0, d, alphamax, alphamin, rho, c1, c2)
    if nargin<10,c2=0.45;
        if nargin<9,c1=0.2;
            if nargin<8,rho=0.5;
                if nargin<7,alphamin=1e-8;
                    if nargin<6,alphamax=1;
                    end
                end
            end
        end
    end

    exitflag = -1;
    alpha = alphamax;
    while (exitflag==-1 && alpha>alphamin)
        xn = x0 + alpha*d;
        [fn, gn] = fun(xn);
        wc1 = (f0-fn >= -alpha*c1*g0'*d);
        wc2 = (gn'*d >= c2*g0'*d);

        if (wc1 && wc2)
            exitflag = 1;
        else
            alpha = rho * alpha;
        end
    end
end
