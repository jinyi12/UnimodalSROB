%% semi-implicit Euler scheme for integrating learned model from zero initial condition
function s_hat = semiImplicitEuler_poly(Ahat, Fhat, Bhat, Phat, dt, u_input, IC)
    K = size(u_input,1);
    r = size(Ahat,1);
    s_hat = zeros(r,K+1); % initial state is zeros everywhere
    
    s_hat(:,1) = IC;

    ImdtA = eye(r) - dt*Ahat;
    for i = 1:K
        ssq = get_x_sq(s_hat(:,i)')';
        gs = gen_poly(s_hat(:, i)', 2)';
        disp("ImdtA");
        disp(ImdtA);
        disp("shat");
        disp(size(s_hat(:, i)));

        disp("dtFhat");
        disp(size(dt*Fhat*ssq))

        disp("dtBhat")
        disp(size(dt*Bhat*u_input(i)))

        disp("dtPhat")
        disp(size(dt*Phat*gs));
        % disp((s_hat(:,i) + dt*Fhat*ssq + dt*Bhat*u_input(i) + dt*Phat*gs));

        s_hat(:,i+1) = ImdtA\(s_hat(:,i) + dt*Fhat*ssq + dt*Bhat*u_input(i) + dt*Phat*gs);
        if any(isnan(s_hat(:,i+1)))
            warning(['ROM unstable at ',num2str(i),'th timestep'])
            break
        end
    end