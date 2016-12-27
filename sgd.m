function [xstar, f, it] = sgd(fun, x0, dt, lb, options)
    fprintf('SGD\r');
    if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
        maxit = options.MaxIter;
    else
        maxit = 100;
    end
    % const
    tol = 1e-10;
    % learning rate
    lrnrate = 0.1;
    momentum = 0.5;
    v = 0.1;
    % init
    it = 0;
    len = size(dt, 1);
    while(it<maxit)
        it = it +1;
        rdperm = randperm(len);
        rd = rdperm(1:500);
        sdt = dt(rd, :);
        sy = lb(rd);
        [fn, gn] = fun(x0, sdt, sy);
        v = momentum * v - lrnrate*gn;
        xn = x0 + v;
        disp([sprintf('%3d',it), sprintf('  %12.4f',fn)]);
        if (norm(x0-xn)<tol)
            xstar = x0;
            f = fn;
            return;
        end
        x0 = xn;
        % monotone learning rate strategy
        % if it>100
        %     lrnrate = 0.75;
        % elseif it>200
        %     lrnrate = 0.01;
        %     momentum = 0.9;
        % % elseif it>800
        % %     lrnrate = lrnrate*0.999;
        % end
    end
    xstar = x0;
    f = fn;
end
