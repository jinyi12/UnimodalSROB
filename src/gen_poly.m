function poly_features = gen_poly(X, p)
    [N, r] = size(X);
    % Precompute all powers of X up to 2*p, stored in a cell array for quick access
    powers = cell(2*p, 1);
    for deg = 1:(2*p)
        powers{deg} = X.^deg;
    end

    % Calculate total number of terms to initialize the output matrix correctly
    total_terms = calculate_combinatorial(r, p);
    poly_features = zeros(N, total_terms);

    current_index = 1; % MATLAB indexing starts at 1

    % Generate single-variable monomials
    for i = 1:r
        for degree = 3:(2*p)
            if degree <= 2*p && current_index <= total_terms
                poly_features(:, current_index) = powers{degree}(:, i);
                current_index = current_index + 1;
            end
        end
    end

    % Precompute powers of X
    powers = zeros(N, r*p);
    for i = 1:p
        powers(:, (i-1)*r+1:i*r) = X.^i;
    end
    
    current_index = 1; % Assuming poly_features is initialized and passed as argument
    
    % Generate combinations of indices for two variables
    [i_indices, j_indices] = find(triu(ones(r, r), 1));
    
    % Initialize poly_features if not passed as an argument
    poly_features = zeros(N, total_terms);
    
    % For each degree combination, use broadcasting equivalent to apply
    for deg_i = 1:p
        for deg_j = 1:p
            if (deg_i + deg_j <= 2*p) && (deg_i + deg_j >= 3)
                for k = 1:length(i_indices)
                    if current_index <= total_terms
                        i_idx = i_indices(k);
                        j_idx = j_indices(k);
                        % Access precomputed powers directly
                        poly_features(:, current_index) = ...
                            powers(:, (deg_i-1)*r + i_idx) .* powers(:, (deg_j-1)*r + j_idx);
                        current_index = current_index + 1;
                    end
                end
            end
        end
    end
end



function total_unique_monomials = calculate_combinatorial(r, p)
    % Count for single-variable monomials
    single_variable_count = r * (2*p - 2);

    % Initialize count for two-variable monomials
    two_variable_count = 0;
    
    % Calculate two-variable monomial counts
    for d = 3:(2*p + 1)
        for i = 1:min(p, d-1) + 1
            if d - i <= p
                % Choose any two variables out of r for the monomial
                two_variable_count = two_variable_count + nchoosek(r, 2);
            end
        end
    end
    
    % Total unique monomials is the sum of single and two-variable counts
    total_unique_monomials = single_variable_count + two_variable_count;
end
