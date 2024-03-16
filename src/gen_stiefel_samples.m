% Run the main function with default parameters

% Call the parseArguments function with the command-line arguments
% parseArguments(argv{:})
% main(p.Results.rank, p.Results.nROB, p.Results.N, p.Results.N_samples);


function gen_stiefel_samples(rank, nROB, N, N_samples)
    % Main function to run the script
    % clc; clear;
    addpath('./toolbox/');

    % Load data
    load Data/rob.mat
    disp(size(rob))

    % Project on the tangential space
    U0 = rob(:, :, end);
    Vs = projectOnTangentSpace(rob, U0, nROB);

    % Solve the quadratic programming problem
    beta = solveQuadraticProgramming(Vs, N, rank, nROB);

    % Sampling
    rng(41);
    [w0, w, tangential_samples, maxI] = sample(N_samples, beta, Vs, N, rank);

    % Save data
    saveData(N_samples, w0, w, tangential_samples, maxI, rob, U0, N, rank);

    function Vs = projectOnTangentSpace(rob, U0, nROB)
        % Project the samples on the tangential space
        Vs = zeros(size(rob));
        tau = 1e-4;
        for i = 1:nROB
            disp("Calculating the " + i + "th tangent vector");
            Vs(:,:,i) = real(stiefel_log(U0, rob(:,:,i), tau, 'print'));
        end
    end
    
    function beta = solveQuadraticProgramming(Vs, N, rank, nROB)
        % Solve the quadratic programming problem
        X = reshape(Vs(:,:,1:nROB), [N*rank,nROB])';
        H = X*X'; f = zeros(nROB,1); Aeq = ones(1,nROB); beq = 1; lb=zeros(1,nROB); ub=ones(1,nROB);
        beta = quadprog(H, f, [], [], Aeq, beq, lb, ub);
        beta = beta*2;
        disp("The coefficients are: " + beta);
    end
    
    function [w0, w, tangential_samples, maxI] = sample(N_samples, beta, Vs, N, rank)
        % Generate samples
        w0 = gamrnd(repmat(beta',N_samples,1), 1, N_samples, length(beta));
        w = w0 ./ sum(w0,2);
        tangential_samples = w*reshape(Vs(:,:,1:length(beta)), [N*rank,length(beta)])';
        [~, I] = sort(w, 2, "ascend");
        maxI = I(:,end);
        disp("MC samples: " + N_samples);
    end
    
    function saveData(N_samples, w0, w, tangential_samples, maxI, rob, U0, N, rank)
        % Save the generated data
        save('data/w0-mc' + string(N_samples) + '.mat', 'w0');
        save('data/w-mc' + string(N_samples) + '.mat', 'w');
    
        % Project back to Stiefel manifold
        tangential_samples = reshape(tangential_samples', [N,rank,N_samples]);
        stiefel_samples = zeros(size(tangential_samples));
        for i = 1:N_samples
            delta = tangential_samples(:,:,i);
            stiefel_samples(:,:,i) = stiefel_exp(U0, delta);
        end
    
        % Save Stiefel samples
        stiefel_samples = cat(3, stiefel_samples, rob);
        path2save = strcat('data/mc-stiefel_samples_', num2str(N_samples), '.mat');
        save(path2save, 'stiefel_samples', 'maxI', 'w');
    
        % % Save rob samples
        % for i = 1:N_samples
        %     writematrix(stiefel_samples(:,:,i), strcat('data/rob_samples/phi', num2str(i), '.txt'), 'Delimiter', ' ');
        % end
    
        disp("Saved in data!") 
    end
end


% % Function to parse command-line arguments and call the main function
% function parseArguments(varargin)
%     % Parse input arguments
%     p = inputParser;
%     addOptional(p, 'rank', 5);
%     addOptional(p, 'nROB', 6);
%     addOptional(p, 'N', 816);
%     addOptional(p, 'N_samples', 2000);
%     parse(p, varargin{:});

%     % Call the main function with the parsed arguments
%     main(p.Results.rank, p.Results.nROB, p.Results.N, p.Results.N_samples);
% end
