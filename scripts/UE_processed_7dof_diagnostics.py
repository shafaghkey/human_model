#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import quaternion_matrix, euler_from_matrix, euler_matrix

#%% Input data

# Initialize variables
n_dof = 8    
dof_names = ['Shoulder Abduction', 'Shoulder Rotation', 'Shoulder Extension', 'Elbow Flexion', \
             'Elbow Supination', 'Wrist Deviation', 'Wrist Supination', 'Wrist Flexion']
dof_codes = ['sh_abd', 'sh_rot', 'sh_ext', 'el_fle', 'el_sup', 'wr_dev', 'wr_sup', 'wr_fle']

n_subjects = 5
subject_labels = ['Subject 0', 'Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'All Subjects']

# Open npz file for each subject
npz_file_0 = './data/processed_exports/subject0_7dof_eul_angles.npz'
npz_file_1 = './data/processed_exports/subject1_7dof_eul_angles.npz'
npz_file_2 = './data/processed_exports/subject2_7dof_eul_angles.npz'
npz_file_3 = './data/processed_exports/subject3_7dof_eul_angles.npz'
npz_file_4 = './data/processed_exports/subject4_7dof_eul_angles.npz'

npz_data_0, npz_data_1, npz_data_2, npz_data_3, npz_data_4 = \
    np.load(npz_file_0), np.load(npz_file_1), np.load(npz_file_2), np.load(npz_file_3), np.load(npz_file_4)

for i in range(5):
    exec(f"time_{i}, eul_right_{i}, eul_left_{i} = \
         npz_data_{i}['time'], npz_data_{i}['eul_right'], npz_data_{i}['eul_left']")

min_angles = np.zeros((n_subjects, n_dof))
max_angles = np.zeros((n_subjects, n_dof))
for s in range(n_subjects):
    for dof_code in dof_codes:
        exec(f"global {dof_code}_{s}")

def calculate_arm_angles(s, eul_right, eul_left, dof_codes, side):

    # Arm angles (+ and - for right and left arm consistency)
    exec(f"R_sh_abd_{s} = -np.around(eul_right[:,3,0] * 180/np.pi, 2)")
    exec(f"R_sh_rot_{s} =  np.around(eul_right[:,3,1] * 180/np.pi, 2)")
    exec(f"R_sh_ext_{s} = -np.around(eul_right[:,3,2] * 180/np.pi, 2)")
    exec(f"R_el_fle_{s} =  np.around(eul_right[:,4,0] * 180/np.pi, 2)")
    exec(f"R_el_sup_{s} =  np.around(eul_right[:,4,1] * 180/np.pi, 2)")
    exec(f"R_wr_dev_{s} = -np.around(eul_right[:,5,0] * 180/np.pi, 2)")
    exec(f"R_wr_sup_{s} =  np.around(eul_right[:,5,1] * 180/np.pi, 2)")
    exec(f"R_wr_fle_{s} = -np.around(eul_right[:,5,2] * 180/np.pi, 2)")

    exec(f"L_sh_abd_{s} =  np.around(eul_left[:,3,0] * 180/np.pi, 2)")
    exec(f"L_sh_rot_{s} =  np.around(eul_left[:,3,1] * 180/np.pi, 2)")
    exec(f"L_sh_ext_{s} =  np.around(eul_left[:,3,2] * 180/np.pi, 2)")
    exec(f"L_el_fle_{s} = -np.around(eul_left[:,4,0] * 180/np.pi, 2)")
    exec(f"L_el_sup_{s} =  np.around(eul_left[:,4,1] * 180/np.pi, 2)")
    exec(f"L_wr_dev_{s} =  np.around(eul_left[:,5,0] * 180/np.pi, 2)")
    exec(f"L_wr_sup_{s} =  np.around(eul_left[:,5,1] * 180/np.pi, 2)")
    exec(f"L_wr_fle_{s} =  np.around(eul_left[:,5,2] * 180/np.pi, 2)")

    # Healthy arm angles
    for d, dof_code in enumerate(dof_codes):
        # exec(f"global {dof_code}_{s}, min_angles, max_angles")
        exec(f"{dof_code}_{s} = {side}_{dof_code}_{s}")
        exec(f"min_angles[s, {d}] = min({dof_code}_{s})")
        exec(f"max_angles[s, {d}] = max({dof_code}_{s})")

    return [f"{dof_code}_{s}" for dof_code in dof_codes], min_angles, max_angles


#%% Data diagnostics
# time_4 = time_4[0:time_4.shape[0]-800]
# eul_right_4 = eul_right_4[0:eul_right_4.shape[0]-800]
# eul_left_4 = eul_left_4[0:eul_left_4.shape[0]-800]

# eul_right_0 = eul_right_0[np.r_[0:12700, 22300:34000, 37000:54500, 56700:58500, 60800:62400, 65400:75000, 79000:87000],:,:]
# eul_left_0 = eul_left_0[np.r_[0:12700, 22300:34000, 37000:54500, 56700:58500, 60800:62400, 65400:75000, 79000:87000],:,:]
# time_0 = time_0[np.r_[0:12700, 22300:34000, 37000:54500, 56700:58500, 60800:62400, 65400:75000, 79000:87000]]

# time_2 = time_2[np.r_[0:15100, 15800:]]
# eul_right_2 = eul_right_2[np.r_[0:15100, 15800:],:,:]
# eul_left_2 = eul_left_2[np.r_[0:15100, 15800:],:,:]

#%% min and max values for healthy arm angles
# min_angles = np.zeros((n_subjects, n_dof))
# max_angles = np.zeros((n_subjects, n_dof))

print("{:^9} {:^16} {:^16} {:^20} {:^17} {:^20} {:^17} {:^16} {:^16}".format\
      ("", "Shoulder", "Shoulder", "Shoulder", "Elbow", "Elbow", "Wrist", "Wrist", "Wrist"))
print("{:^9} {:^16} {:^16} {:^20} {:^17} {:^20} {:^17} {:^16} {:^16}".format\
      ("", "Abduction", "Rotation", "Extension", "Flexion", "Supination", "Deviation", "Supination", "Flexion"))
print("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12}".format\
      ("Subject", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max"))

#%% Plot the data vs time for diagnostics
i_start = 0 
i_end = 6000
i_end_values = np.zeros(n_subjects, dtype=int)
colors = ['b', 'g', 'r', 'c', 'm', 'orange', 'purple', 'brown']
# n_dof = 4 

for s in range(n_subjects):   
    side = 'R' if s == 0 or s == 4 else 'L'

    # Arm angles (+ and - for right and left arm consistency)
    exec(f"R_sh_abd_{s} = -np.around(eul_right_{s}[:,3,0] * 180/np.pi, 2)")
    exec(f"R_sh_rot_{s} =  np.around(eul_right_{s}[:,3,1] * 180/np.pi, 2)")
    exec(f"R_sh_ext_{s} = -np.around(eul_right_{s}[:,3,2] * 180/np.pi, 2)")
    exec(f"R_el_fle_{s} =  np.around(eul_right_{s}[:,4,0] * 180/np.pi, 2)")
    exec(f"R_el_sup_{s} =  np.around(eul_right_{s}[:,4,1] * 180/np.pi, 2)")
    exec(f"R_wr_dev_{s} = -np.around(eul_right_{s}[:,5,0] * 180/np.pi, 2)")
    exec(f"R_wr_sup_{s} =  np.around(eul_right_{s}[:,5,1] * 180/np.pi, 2)")
    exec(f"R_wr_fle_{s} = -np.around(eul_right_{s}[:,5,2] * 180/np.pi, 2)")

    exec(f"L_sh_abd_{s} =  np.around(eul_left_{s}[:,3,0] * 180/np.pi, 2)")
    exec(f"L_sh_rot_{s} =  np.around(eul_left_{s}[:,3,1] * 180/np.pi, 2)")
    exec(f"L_sh_ext_{s} =  np.around(eul_left_{s}[:,3,2] * 180/np.pi, 2)")
    exec(f"L_el_fle_{s} = -np.around(eul_left_{s}[:,4,0] * 180/np.pi, 2)")
    exec(f"L_el_sup_{s} =  np.around(eul_left_{s}[:,4,1] * 180/np.pi, 2)")
    exec(f"L_wr_dev_{s} =  np.around(eul_left_{s}[:,5,0] * 180/np.pi, 2)")
    exec(f"L_wr_sup_{s} =  np.around(eul_left_{s}[:,5,1] * 180/np.pi, 2)")
    exec(f"L_wr_fle_{s} =  np.around(eul_left_{s}[:,5,2] * 180/np.pi, 2)")
    
    # Healthy arm angles:
    for i, dof_code in enumerate(dof_codes[:n_dof]):
        exec(f"{dof_code}_{s} = {side}_{dof_code}_{s}")
        exec(f"min_angles[{s}, {i}], max_angles[{s}, {i}] = min({dof_code}_{s}), max({dof_code}_{s})")

    exec(f"ang_range_{s} = max_angles[s,:] - min_angles[s,:]")
    exec(f"ax_min_{s}, ax_max_{s} = np.min(min_angles[s,:]), np.max(max_angles[s,:])")

    print("{:<9} {:<8} {:<8} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12}".format("#"+str(s), \
      str(min_angles[s,0]), str(max_angles[s,0]), str(min_angles[s,1]), str(max_angles[s,1]), str(min_angles[s,2]), str(max_angles[s,2]), str(min_angles[s,3]), str(max_angles[s,3]), \
      str(min_angles[s,4]), str(max_angles[s,4]), str(min_angles[s,5]), str(max_angles[s,5]), str(min_angles[s,6]), str(max_angles[s,6]), str(min_angles[s,7]), str(max_angles[s,7])))

    # plot data
    n_ax, i_start, i_end = 3, 0, 30000

    i_end_values[s] = min(i_end, len(eval(f"time_{s}"))-1)
    fig1, ax1 = plt.subplots(n_ax, 1, sharex=True, figsize=(n_ax*6, 10))
    ax1[0].set(position=(0.1, 0.5, 0.8, 0.45), ylim=(eval(f"ax_min_{s}")-10, eval(f"ax_max_{s}")+10), ylabel=f'Angles (deg)', title=f'{subject_labels[s]}')
    ax1[1].set(position=(0.1, 0.27, 0.8, 0.2), ylim=(-90, 90), ylabel=f'Angles_First derivative (deg/s)')
    ax1[2].set(position=(0.1, 0.05, 0.8, 0.2), ylim=(-90, 90), ylabel=f'Angles_Second derivative (deg/s^2)', xlabel='Time (s)')
    for n in range(n_ax):
        if i_start > len(eval(f"time_{s}")):
            break
        ax1[n].set(xlim=(eval(f"time_{s}")[i_start]-1, eval(f"time_{s}")[i_start]+(i_end-i_start)/100+10))
        for d in range(n_dof):
            ax1[n].plot(eval(f"time_{s}")[n:-1], np.diff(eval(f"{dof_codes[d]}_{s}")[:-1], n=n), label=f'{dof_names[d]}', color=colors[d])    
    
    # Jumps, Switches, and Filters in the data
    exec(f"take_ind_{s} = [0]")
    exec(f"jump_ind_{s} = []")
    exec(f"switch_ind_{s} = []")
    exec(f"filt_ind_{s} = []")
    for t in range(2, len(eval(f"time_{s}"))-1):
        if eval(f"time_{s}")[t] - eval(f"time_{s}")[t-1] < 0.001:
            exec(f"take_ind_{s}.append(t)")
            [exec(f"{dof_codes[d]}_{s}[t] = {dof_codes[d]}_{s}[t+1]") for d in range(n_dof)]
            continue
        ang_dot = np.array([eval(f"{dof_codes[d]}_{s}[t]-{dof_codes[d]}_{s}[t-1]") for d in range(n_dof)])
        ang_dot_prev = np.array([eval(f"{dof_codes[d]}_{s}[t-1]-{dof_codes[d]}_{s}[t-2]") for d in range(n_dof)])
        ang_ddot = np.abs(ang_dot - ang_dot_prev)    # second derivative
        ang_dot = np.abs(ang_dot)    # first derivative
        if sum((ang_dot > 20)) > 2:
            exec(f"jump_ind_{s}.append(t)")
            t += 1
            # print(f"Time: {t}, Ang_dot: {sum(ang_dot)}, Ang_ddot: {sum(ang_ddot)}")
            continue
        if sum((ang_dot > eval(f"ang_range_{s}")/4)) > 1:
            exec(f"switch_ind_{s}.append(t)")
            # t += 1
            continue
        if (ang_dot > eval(f"ang_range_{s}")/4).any():
            # only update the dof that has a jump
            for d in range(n_dof):
                if ang_dot[d] > 10:
                    exec(f"{dof_codes[d]}_{s}[t] = {dof_codes[d]}_{s}[t-1]")
            exec(f"filt_ind_{s}.append(t)")
        # if sum((ang_ddot > 2)) > 3 and sum(ang_dot)>5:

    for i, dof_code in enumerate(dof_codes[:n_dof]):
        exec(f"min_angles[{s}, {i}], max_angles[{s}, {i}] = min({dof_code}_{s}), max({dof_code}_{s})")

    print("{:<9} {:<8} {:<8} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12}".format("#"+str(s), \
      str(min_angles[s,0]), str(max_angles[s,0]), str(min_angles[s,1]), str(max_angles[s,1]), str(min_angles[s,2]), str(max_angles[s,2]), str(min_angles[s,3]), str(max_angles[s,3]), \
      str(min_angles[s,4]), str(max_angles[s,4]), str(min_angles[s,5]), str(max_angles[s,5]), str(min_angles[s,6]), str(max_angles[s,6]), str(min_angles[s,7]), str(max_angles[s,7])))
    print()

    for n in range(n_ax):
        if i_start > len(eval(f"time_{s}")):
            ax1[n].axis('off')
            break
        for d in range(n_dof):
            ax1[n].plot(eval(f"time_{s}")[n:-1], np.diff(eval(f"{dof_codes[d]}_{s}")[:-1], n=n), label=f'{dof_names[d]}_filtered', linestyle='dashed', color=colors[d])
        for take in eval(f"take_ind_{s}"):
            ax1[n].annotate('Take', xy=(eval(f"time_{s}")[take], 90), xytext=(eval(f"time_{s}")[take], 150), arrowprops=dict(facecolor='black', shrink=0.05))
        for jump in eval(f"jump_ind_{s}"):
            ax1[n].annotate('Jump', xy=(eval(f"time_{s}")[jump], 90), xytext=(eval(f"time_{s}")[jump], 150), arrowprops=dict(facecolor='black', shrink=0.05))
            # ax1[n].fill_between(eval(f"time_{s}")[jump]-10, eval(f"time_{s}")[jump]+10, ax1[n].get_ylim()[0], ax1[n].get_ylim()[1], color='gray', alpha=0.5)
        for switch in eval(f"switch_ind_{s}"):
            ax1[n].annotate('Switch', xy=(eval(f"time_{s}")[switch], 90), xytext=(eval(f"time_{s}")[switch], 150), arrowprops=dict(facecolor='black', shrink=0.05))
        for filt in eval(f"filt_ind_{s}"):
            ax1[n].fill_between(eval(f"time_{s}")[filt], -200, 200, color='yellow', alpha=0.5)
            
        # ax1[n].legend(loc='right')
        ax1[n].grid()
        ax1[n].xaxis.set_major_locator(plt.MultipleLocator(20))
        ax1[n].xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax1[0].legend(loc='right')
    
    # print(f"Subject {s}, axis min: {eval(f'ax_min_{s}')}, axis max: {eval(f'ax_max_{s}')}")

dof_min, dof_max = np.min(min_angles, axis=0), np.max(max_angles, axis=0)

for s in range(n_subjects):
    print(f"Subject {s}, switches: {eval(f'switch_ind_{s}')}, filtered: {eval(f'len(filt_ind_{s})')}")
    # print(f"Subject {s}, switches: {eval(f'switch_ind_{s}')}")

plt.show()

# if sum((ang_dot > 10)) > 4:
#     exec(f"jump_ind_{s}.append(t)")
# Subject 0, jumps: [22046, 22212, 24121, 78625, 88119, 88120], filtered: 757
# Subject 1, jumps: [16570, 16572], filtered: 1
# Subject 2, jumps: [13240, 15413, 19731], filtered: 9
# Subject 3, jumps: [10290], filtered: 0
# Subject 4, jumps: [16608], filtered: 13

# if sum((ang_dot > 10)) > 5:
#     exec(f"jump_ind_{s}.append(t)")
# Subject 0, jumps: [24122, 78625, 88119, 88121], filtered: 861
# Subject 1, jumps: [16570, 16573], filtered: 2
# Subject 2, jumps: [15414], filtered: 15
# Subject 3, jumps: [], filtered: 4
# Subject 4, jumps: [16635], filtered: 40

# if sum((ang_dot > 5)) > 6:
#     exec(f"jump_ind_{s}.append(t)")
# Subject 0, jumps: [24294, 78625, 88119, 88121], filtered: 1033
# Subject 1, jumps: [16570, 16572], filtered: 1
# Subject 2, jumps: [13240, 15414, 19731], filtered: 10
# Subject 3, jumps: [], filtered: 4
# Subject 4, jumps: [16613], filtered: 18

# if sum((ang_dot > 2)) > 7:
#     exec(f"jump_ind_{s}.append(t)")
# Subject 0, jumps: [4, 6, 24136, 78625, 78628, 88119, 88122], filtered: 876
# Subject 1, jumps: [16570, 16574], filtered: 3
# Subject 2, jumps: [13240, 15422, 19733], filtered: 20
# Subject 3, jumps: [2, 10290], filtered: 0
# Subject 4, jumps: [8265, 8271, 16622], filtered: 27    
