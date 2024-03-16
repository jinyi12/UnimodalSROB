import numpy as np
from scipy.io import savemat
from toolbox import read_snapshots
import argparse


if __name__ == "__main__":

    # filepath
    parser = argparse.ArgumentParser(description='Read snapshots from lammps trajectory file')
    parser.add_argument('--root', type=str, default="/data1/jy384/research/Data/SROB/Airebo/")
    parser.add_argument('--file', type=str, default="dump.waveload.rob.lammpstrj")
    
    
    # # Change your root here:
    # root = "/data1/jy384/research/Data/SROB/Airebo/"
    # file = "dump.waveload.rob.lammpstrj"
    
    args = parser.parse_args()
    
    # if no argument is given, use the default root and file
    if not args.root:
        root = "/data1/jy384/research/Data/SROB/Airebo/"
    else:
        root = args.root
    
    if not args.file:
        file = "dump.waveload.rob.lammpstrj"
    else:
        file = args.file

    # num snapshots; num_states-id, x, y, x (num columns); headlines - numlines to skip in the beginning
    num_ss = 1000
    num_atom = 272
    num_states = 10
    headlines = 9

    print("Reading from file: " + root + file + " ...")
    print("Data with num_ss = " + str(num_ss) + " ...")

    # Read snapshots
    ss = read_snapshots(root + file, num_ss, num_atom, num_states, headlines)

    # init_qmin is the initial position of (the first ss)
    init_qmin = np.reshape(ss[:,1:4, :], [num_atom*3, num_ss], order='F')
    init_qmin = init_qmin[:,0]
    ss_q = np.reshape(ss[:,1:4, :], [num_atom*3, num_ss], order='F')

    # ss_q_dis = ss_q;
    ss_q_dis = ss_q - np.tile(init_qmin[:, np.newaxis], (1, num_ss))

    print("Saving data")

    # save ss_q_dis and init_qmin
    savemat(root + "ss_q.mat", {"ss_q": ss_q, "ss_q_dis": ss_q_dis, "init_qmin": init_qmin})

    print("Data saved: concatenated as x(t), y(t), z(t), x(t+1), y(t+1), z(t+1)...")

