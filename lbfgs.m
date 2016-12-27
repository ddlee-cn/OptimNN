function [x, f, it] = lbfgs(fun, x0, options)
    fprintf('L-BFGS\r');
    if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
        maxit = options.MaxIter;
    else
        maxit = 100;
    end
    % const
    tol = 1e-10;
    % initial statement
    it = 0; itt = 0;
    xold = x0;
    x=x0+1;
    m = 5; % limited memory
    s = zeros(length(x0),m);
    y = zeros(length(x0),m);
    [f, g] = fun(x0);
    s(:,1)=x0;
    y(:,1)=g;

    while(it<maxit)
        it = it+1;
        itt = itt+1;

        % choose H0
        if itt>1
            H0 = chooseH(s(:,itt-1), y(:,itt-1));
        else
            H0 = eye(length(s(:,itt)));
        end
        % compute H(k)*delta_k and direction p
        p = computeH(H0, s, y, g, itt, m);

        % line search, choose alpha, compute x_k+1
        x = linesearch(fun, xold, f, g, p);

        % free storage
        if itt > m
            s(:,1:end-1) = s(:,2:end);
            y(:,1:end-1) = y(:,2:end);
            itt = itt-1;
        end

        [fn, gn] = fun(x);
        s(:, itt) = x - xold;
        y(:, itt) = gn - g;
        xold = x;
        f = fn;
        g = gn;
        if norm(s(:, itt))<tol
            return;
        end
        % generate report
        disp([sprintf('%3d',it), sprintf('  %12.4f',fn)]);

    end
end





