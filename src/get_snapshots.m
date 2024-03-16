% calculate the snapshots at tow different simulation time after stochastic ROM

clc; clear;
addpath('/data1/jy384/research/Data/SROB/toolbox/')

% Change your root here:
%% read data
root = "/data1/jy384/research/Data/SROB/Airebo/";
%%num snapshots; num_states-id, x, y, x (num columns); headlines - numlines to skip in the beginning
num_ss = 1000; num_atom = 272; num_states = 10; headlines = 9;   
file = "dump.waveload.rob.lammpstrj";

disp("Reading from file: " + root + file + " ...")
disp("Data with num_ss = " + num_ss + " ...")
% runtime
tic

dir = strcat(root);
% disp(dir)
ss = read_snapshots(strcat(dir, file), num_ss, num_atom, num_states, headlines);
% init_qmin is the initial position of (the first ss)
init_qmin = reshape(ss(:,2:4, :), [num_atom*3, num_ss]);
init_qmin = init_qmin(:,1);
ss_q = reshape(ss(:,2:4, :), [num_atom*3, num_ss]);
% ss_q_dis = ss_q;
ss_q_dis = ss_q - init_qmin;

toc
% disp(saving data)
disp("Saving data")
% save ss_q_dis and init_qmin
save(root + "ss_q.mat", "ss_q", "ss_q_dis", "init_qmin")
disp("Data saved: concatenated as x(t), y(t), z(t), x(t+1), y(t+1), z(t+1)...")


%% save data
% T = [1, 4, 5];
% idx = num_atom+1:num_atom*2; % selects the y displacements
% ss6 = load("data/ss_6.mat"); ss6=ss6.ss_q;
% data_6 = squeeze(ss6(idx, T, :));
% data_s = ss_q(idx, T, :);
% save("data_nested/ROM_" + int2str(N_sample) + "samples_multipot.mat", "data_s", "data_6")

% disp("Save in data nested folder!")







