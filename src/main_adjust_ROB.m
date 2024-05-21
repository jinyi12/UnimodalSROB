clc; clear;
load rob.mat 
rank = 5;

%% adjust ROBs
rob1 = rob(:,:,1);
% idx = 5;
num_atom = 272;
for idx = 1:6
    rob_idx = rob(:,:,idx);
for i = 1:rank
    dist1 = norm(rob1(:,i) - rob_idx(:,i));
    dist2 = norm(rob1(:,i) + rob_idx(:,i));
    if dist2 < dist1
        rob(:,i,idx) = -rob(:,i,idx);
    end
end
end




