function data = read_snapshots(filename, num_T, num_atoms, num_states, headlines)
fid = fopen(filename, 'r');
str = fgets(fid);
line = 1; % line number in the data matrix plus headlines
idx = 1; % line number in the data matrix
T = 1; % number of snapshot
data = zeros(num_atoms, num_states, num_T);
tot_lines = num_atoms + headlines;
while str ~= -1
    if line > headlines
        data(idx, :, T) = str2num(str);
        idx = idx+1;
    end
    line = line + 1;
    if line > tot_lines
        line = 1;
        idx = 1; % replace the idx for data matrix
        [~,I] = sort(data(:,1,T), 'ascend');
        data(:,:,T) = data(I,:,T);
        T = T+1; % move to the next snapshot
    end
    if T > num_T
        break
    end
    str = fgets(fid);

end

end