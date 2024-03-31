function runburgers(N, dt, T_end, mus, Mp, DS, u_ref)

    logFile = fopen('/data1/jy384/research/Data/UnimodalSROB/Burgers/runburgers_log.txt', 'w');
    
    % Add src path which is one level above the current directory
    addpath(genpath('../'));

    % disp(srcpath);

    disp('Running Burgers equation data generation');

    disp('Parameters:');
    disp(['N = ', num2str(N)]);
    disp(['dt = ', num2str(dt)]);
    disp(['T_end = ', num2str(T_end)]);
    disp(['mus = ', num2str(mus)]);
    disp(['Mp = ', num2str(Mp)]);
    disp(['DS = ', num2str(DS)]);

    % Get parameters from config file
    data = jsondecode(fileread('config.json'));
    params = data.params;  % This will be a struct in MATLAB

    % Initialize parameters
    K = T_end/dt; % num time steps
    IC = zeros(N,1);

    % If u_ref is not provided, use ones
    % Check if u_ref is provided
    if nargin < 7 
        u_ref = ones(K,1);
    end

    % Initialize cell arrays to hold X and R for all mu
    X_all = cell(1, length(mus));
    R_all = cell(1, length(mus));
    
    % Initialize storage for the solution U
    % U_all = zeros(K*Mp, length(mus));  % Adjusted size based on vectorization of U
    U_all = cell(1, length(mus));

    % Store values
    s_ref_all = cell(Mp,1);
    
    % LEARN AND ANALYZE TRAINING DATA
    for i = 1:length(mus)
        mu = mus(i);
        disp(['mu = ', num2str(mu)]);
        [A,B,F] = getBurgers_ABF_Matrices(N,1/(N-1),dt,mu);
        s_ref = semiImplicitEuler(A,F,B,dt,u_ref,IC);
        s_ref_all{i} = s_ref;
        
        % Collect data for a series of trajectories with random inputs
        U_rand = rand(K,Mp);
        x_all = cell(Mp,1);
        xdot_all = cell(Mp,1);

        for k = 1:Mp
            s_rand = semiImplicitEuler(A,F,B,dt,U_rand(:,k),IC);
            foo = s_rand(:,2:end); % throw away IC
            [bar, ind] = ddt(foo,dt, params.ddt_order);
            % x_all{k} = foo(:,1:DS:end);  % down-sample and store
            foo = foo(:,ind);  % retain the indices after central differencing
            x_all{k} = foo(:,1:DS:end);  % down-sample and store
            % bar = (s_rand(:,2:end)-s_rand(:,1:end-1))/dt;
            xdot_all{k} = bar(:,1:DS:end);  % down-sample and store
        end

        %  Concatenate and form the data matrix and the right-hand side matrix
        X = cat(2,x_all{:});  % concatenate data from random trajectories
        R = cat(2,xdot_all{:}); % concatenate derivatives from random trajectories, R for rhs

        % Store the X and R data for each mu
        X_all{i} = X;
        R_all{i} = R;

        % Concatenate and down-sample U
        U = U_rand(1:DS:end,:);  % Down-sample
        U = U(ind,:);  % Retain the indices after central differencing
        
        % U_all(:, i) = U(:);  % Vectorize and store in the matrix for all mus
        U_all{i} = U(:);  % Store the U matrix for each mu

        % Save the data for each mu
        filename = sprintf('/data1/jy384/research/Data/UnimodalSROB/Burgers/data_mu_%g.mat', mu);
        save(filename, 'X', 'R', 'U');
    end

    U_all = cat(2, U_all{:});
    disp(size(U_all));
    
    % X_tensor = cat(3, X_all{:});
    % R_tensor = cat(3, R_all{:});

    % save X and R data and name according to the mu values
    mu_start = mus(1);
    mu_end = mus(end);
    mu_step = mus(2) - mus(1);
    X_all_filename = sprintf('/data1/jy384/research/Data/UnimodalSROB/Burgers/snapshots/X_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end);
    R_all_filename = sprintf('/data1/jy384/research/Data/UnimodalSROB/Burgers/snapshotsDerivatives/R_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end);

    % Save the computed U matrix to a file
    U_all_filename = sprintf('/data1/jy384/research/Data/UnimodalSROB/Burgers/trajectories/U_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end);
    s_ref_all_filename = sprintf('/data1/jy384/research/Data/UnimodalSROB/Burgers/referenceState/s_ref_all_mu_%g_%g_%g.mat', mu_start, mu_step, mu_end);
    

    save(U_all_filename, 'U_all');
    save(s_ref_all_filename, 's_ref_all');
    save(X_all_filename, 'X_all');
    save(R_all_filename, 'R_all');

    % Write information to log file
    fprintf(logFile, 'Data generated for Burgers equation with N = %d, dt = %g, T_end = %g, mus = [%s], Mp = %d, DS = %d\n', N, dt, T_end, num2str(mus), Mp, DS);
    % Write dimensions of X and R to log file
    fprintf(logFile, 'Size of X: %d x %d x %d\n', size(X_all));
    fprintf(logFile, 'Size of R: %d x %d x %d\n', size(R_all));
    % Write information about U to log file
    fprintf(logFile, 'Size of U: %d x %d\n', size(U_all));
    fprintf(logFile, 'Columns of U is K*Mp, where K = %d and Mp = %d\n', K, Mp);
    % Close log file
    fclose(logFile);

end