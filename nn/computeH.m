function [p] = computeH(H0, s, y, g, itt, m)
    % full memory
    if itt>m
        alpha=zeros(1,m);
        for i = itt-1:-1:itt-m
            alpha(i) = 1/((y(:,i))'*s(:,i)) * (s(:,i))' * g;
            g = g - alpha(i)*y(:,i);
        end
        r = H0 * g;
        for i = itt-m:itt-1
            beta = 1/((y(:,i))'*s(:,i)) * (y(:,i))' * r;
            r = r + s(:,i) * (alpha(i) - beta);
        end
    else
    % efficient memory
        alpha=zeros(1:itt);
        for i = itt-1:-1:1
            alpha(i) = 1/((y(:,i))'*s(:,i)) * (s(:,i))' * g;
            g = g - alpha(i)*y(:,i);
        end
        r = H0 * g;
        for i = 1:itt-1
            beta = 1/((y(:,i))'*s(:,i)) * (y(:,i))' * r;
            r = r + s(:,i) * (alpha(i) - beta);
        end
    end
    p = -r;
end
