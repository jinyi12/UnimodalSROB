% This function infers the operators of the Burgers equation using the
% state data, input data, and the POD basis. The inferred operators are
% stored in a cell array and saved in the data path.
%
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


function [infop_all, rmax] = burgersOpInf(params, Mp, mus, r, config, X_all, R_all, U_all, Vr_all)

    infop_all = cell(Mp,1);

    % load data from a path
    datapath = '/data1/jy384/research/Data/UnimodalSROB/Burgers';
    trajectories_path = fullfile(datapath, 'trajectories');
    s_ref_all_path = fullfile(datapath, 'referenceState');
    snapshot_path = fullfile(datapath, 'snapshots');
    snpashotsDerivatives_path = fullfile(datapath, 'snapshotsDerivatives');

    addpath(datapath);
    addpath(trajectories_path);
    addpath(s_ref_all_path);
    addpath(snapshot_path);
    addpath(snpashotsDerivatives_path);

    mu_start = mus(1);
    mu_step = mus(2) - mus(1);
    mu_end = mus(end);

    if nargin < 6
        if ~isfield(config, 'perturbations')
            Vr_all = load(sprintf('Vr_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end)); % all POD basis
            X_all = load(sprintf('X_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end)); % all state data
            R_all = load(sprintf('R_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end)); % all RHS data
            U_all = load(sprintf('U_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end)); % all input data
        else
            uid = config.uid;
            Vr_all = load(sprintf('Vr_all_nominalmu_%s_perturb_%s.mat', config.nominal_mu, uid))
            X_all = load(sprintf('X_all_nominalmu_%s_perturb_%s.mat', config.nominal_mu, uid));
            R_all = load(sprintf('R_all_nominalmu_%s_perturb_%s.mat', config.nominal_mu, uid));
            U_all = load(sprintf('U_all_nominalmu_%s_perturb_%s.mat', config.nominal_mu, uid));
        end
    end

    rmax = r;

    for i = 1:length(mus)
        
        mu = mus(i);
        Vr = Vr_all.Vr_all(:, :, i);
        U = U_all.U_all(:, i);
        X = X_all.X_all{i};
        R = R_all.R_all{i};
        
        
        % Inferred operators with stability check
        % while true
        [operators] = inferOperators(X, U, Vr, params, R);
        disp("Operators are inferred.");
        Ahat = operators.A;
        Fhat = operators.F;
        Bhat = operators.B;
        Phat = operators.P;

        % % Check if the inferred operator is stable 
        % lambda = eig(Ahat);
        % Re_lambda = real(lambda);
        % if all(Re_lambda(:) < 0)
        %     infop_all{i} = operators;  % store operators
        %     disp("Operators are stable. Storing operators.");
        %     break;
        % else
        %     warning("For mu = %f, order of r = %d is unstable. Decrementing max order.\n", mu, rmax);
        %     rmax = rmax - 1;
        %     Vr = Vr(:,1:rmax);
        % end
        infop_all{i} = operators;  % store operators
        % end
    end

    % save at the data path
    operators_path = fullfile(datapath, 'operators');

    if isfield(config, 'perturbations')
        operators_filename = sprintf('operators_nominalmu_%s_perturb_%s.mat', config.nominal_mu, config.uid);
    else
        operators_filename = sprintf('operators_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end);

    save(fullfile(operators_path, operators_filename), 'infop_all');
    rmpath(datapath);
end
