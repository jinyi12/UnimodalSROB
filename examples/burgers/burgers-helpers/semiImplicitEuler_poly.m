%% semi-implicit Euler scheme for integrating learned model from zero initial condition
function s_hat = semiImplicitEuler_poly(Chat, Ahat, Fhat, Bhat, Phat, dt, u_input, IC)
    K = size(u_input,1);
    r = size(Ahat,1);
    s_hat = zeros(r,K+1); % initial state is zeros everywhere
    
    s_hat(:,1) = IC;

    % disp('Size of Ahat')
    % disp(size(Ahat))
    % disp('Size pf Fhat')
    % disp(size(Fhat))
    % disp('Size pf Phat')
    % disp(size(Phat))

    % disp(Ahat)
    % disp(Fhat)
    % disp(Phat)

    ImdtA = eye(r) - dt*Ahat;
    for i = 1:K
        ssq = get_x_sq(s_hat(:,i)')';
        gs = gen_poly(s_hat(:, i)', 0)';

        if isempty(Bhat)
            s_hat(:,i+1) = ImdtA\(s_hat(:,i) + dt*Fhat*ssq + dt*Phat*gs + dt*Chat);
            % disp((s_hat(:,i) + dt*Fhat*ssq + dt*Phat*gs))
            % disp(dt * Fhat * ssq)
            
        else
            % disp(size(Bhat))
            % disp(size(u_input(i)))
            % disp(size(Fhat*ssq))
            % disp(size(Phat))
            % disp(size(gs))
            Bhat_term = dt*Bhat*u_input(i);
            s_hat(:,i+1) = ImdtA\(s_hat(:,i) + dt*Fhat*ssq + Bhat_term + dt*Phat*gs + dt*Chat);
        end        

        
        if any(isnan(s_hat(:,i+1)))
            warning(['ROM unstable at ',num2str(i),'th timestep'])
            break
        end
    end