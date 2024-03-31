function run_burgersOpInf()

    clear; close all; clc;

    datapath = '/data1/jy384/research/Data/UnimodalSROB/Burgers';
    s_ref_all_path = fullfile(datapath, 'referenceState');
    reconstructedState_path = fullfile(datapath, 'reconstructedState');
    ROBs_path = fullfile(datapath, 'ROBs');
    addpath(ROBs_path);
    addpath(datapath);
    addpath(s_ref_all_path);
    addpath('/home/jy384/projects/UnimodalSROB/examples/burgers/burgers-helpers')

    data = jsondecode(fileread('config.json'));

    N = data.N;
    dt = data.dt;
    T_end = data.T_end;
    mus = eval(data.mus);  % Assuming mus is saved as a string that represents a MATLAB expression
    Mp = data.Mp;
    K = data.K;
    DS = data.DS;
    params = data.params;  % This will be a struct in MATLAB
    robparams = data.robparams;  % This will be a struct in MATLAB

    r = robparams.r;

    mu_start = mus(1);
    mu_step = mus(2) - mus(1);
    mu_end = mus(end);

    % reference state for different mus
    s_ref_all = load(sprintf('s_ref_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end));
    s_ref_all = s_ref_all.s_ref_all;

    % Define cell to store reconstructed states
    s_rec_all = cell(Mp, 1);

    % Define array to store reconstructed error for each parameter mu
    err_inf_all = zeros(Mp, 1);

    % Inputs:
    %   - params: parameters for the operator inference, including:
    %       - modelform: model form of the operator
    %       - modeltime: model time of the operator
    %       - dt: timestep to compute state time derivative
    %       - ddt_order: explicit 1st order timestep scheme
    %   - mus: parameter values
    %   - r: POD basis order
    %   - X_all: state data
    %   - R_all: RHS data (residual)
    %   - U_all: input data
    %   - Vr_all: truncated POD basis

    % Define the reference input
    u_ref = ones(K,1);

    % Vr_all = load('Vr_all.mat'); % all truncated POD basis of order r
    Vr_all = load(sprintf('Vr_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end)); % all truncated POD basis of order r
    Vr_all = Vr_all.Vr_all;

    [infop_all] = burgersOpInf(params, Mp, mus, r);
    % operators_path = fullfile(datapath, 'operators');
    % infop_all = load(fullfile(operators_path, sprintf('operators_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end)));
    % infop_all = infop_all.infop_all;
    IC = zeros(N,1);

    for i = 1:length(mus)
        disp('Solving for mu = ' + string(mus(i)))
        mu = mus(i);
        Vr = Vr_all(:, :, i);

        operators = infop_all{i};
        s_ref = s_ref_all{i};

        Chat = operators.C;
        Ahat = operators.A;
        Bhat = operators.B;
        Fhat = operators.F;
        Hhat = operators.H;
        % Fhat = extractF(Fhat, r);

        Phat = operators.P;

        % Check if Bhat is empty
        if isempty(Bhat)
            disp('Bhat is empty')
        else
            disp('Bhat is not empty')
        end

        s_hat = semiImplicitEuler_poly(Chat, Ahat(1:r, 1:r), Fhat, Bhat(1:r,:), Phat, dt, u_ref, Vr'*IC);
        % s_hat = semiImplicitEuler(Ahat(1:r,1:r),Fhat,Bhat(1:r,:),dt,u_ref,Vr'*IC);
        s_rec = Vr*s_hat;
        s_rec_all{i} = s_rec;

        err_inf = norm(s_rec - s_ref, 'fro')/norm(s_ref, 'fro');
        err_inf_all(i) = err_inf;
        
    end

    % Save the datas
    reconstructedState_filename = sprintf('s_rec_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end);
    save(fullfile(reconstructedState_path, reconstructedState_filename), 's_rec_all');

end