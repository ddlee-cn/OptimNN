function [xstar, f, it] = sgd(fun, x0, dt, lb, options)
    fprintf('SGD\r');
    assert(all(isfield(options,{'epochs','lrnrate','minibatch'})),...
            'Some options not defined');
    if ~isfield(options,'momentum')
        options.momentum = 0.9;
    end;
    % setup
    epochs = options.epochs;
    lrnrate = options.lrnrate;
    minibatch = options.minibatch;
    mom = 0.5;
    momIncrease = 20;
    velocity = zeros(size(x0));
    % init
    it = 0;
    len = size(dt, 1);
    for e = 1:epochs
        it = it +1;
        rdperm = randperm(len);
        for s=1:minibatch:(len-minibatch+1)
            it = it +1;

            % set momentum
            if it == momIncrease
                mom = options.momentum;
            end;

            % get random data
            mb_data = dt(rdperm(s:s+minibatch-1),:);
            mb_label = lb(rdperm(s:s+minibatch-1));
            [cost, grad] = fun(x0, mb_data, mb_label);

            % update x0
            velocity = mom*velocity +lrnrate*grad;
            x0 = x0 - velocity;

            % generate report
            fprintf('Epoch %d: Cost on iteration %d | %f\n',e,it,cost);

        end
        lrnrate = lrnrate/2.0;
    end
    xstar = x0;
    f = cost;
end
