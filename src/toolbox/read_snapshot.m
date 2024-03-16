function data = read_snapshot(filename, T, num_atoms, num_states, headlines)
fid = fopen(filename, 'r');
line = 0; % line number in the data matrix plus headlines
idx = 1; % line number in the data matrix
data = zeros(num_atoms, num_states);
tot_lines = num_atoms + headlines;
% skip the headlines until T
for i = 1:((T-1)*tot_lines)
    str = fgets(fid);
end
% read data
while line < tot_lines
    str = fgets(fid);
    if line >= headlines
        data(idx, :) = str2num(str);
        idx = idx+1;
    end
    line = line + 1;
end
% sort the index
[~,I] = sort(data(:,1), 'ascend');
data = data(I,:);

end