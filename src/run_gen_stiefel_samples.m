function run_gen_stiefel_samples(rank, nROB, N, N_samples)
    % Convert input arguments to numbers
    rank = str2double(rank);
    nROB = str2double(nROB);
    N = str2double(N);
    N_samples = str2double(N_samples);

    % Call the main function with the parsed arguments
    gen_stiefel_samples(rank, nROB, N, N_samples);
end