#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import quaternion_matrix, euler_from_matrix, euler_matrix

#%% Input data: full

PLOT = True     # Plot the data
n_ax = 3        # angles, first derivative, second derivative

# Initialize variables
n_dof = 11    
dof_names = ['Clavicle Protraction', 'Clavicle Elevation', 'Clavicle Axial Rotation', \
             'Shoulder Abduction', 'Shoulder Rotation', 'Shoulder Extension', 'Elbow Flexion', \
             'Elbow Supination', 'Wrist Deviation', 'Wrist Supination', 'Wrist Flexion']
dof_codes = ['cl_x', 'cl_y', 'cl_z', 'sh_abd', 'sh_rot', 'sh_ext', 'el_fle', 'el_sup', 'wr_dev', 'wr_sup', 'wr_fle']

n_subjects = 5
subject_labels = ['Subject 0', 'Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'All Subjects']

# Open npz file for each subject
npz_file_0 = './data/processed_exports/subject0_eul_angles.npz'
npz_file_1 = './data/processed_exports/subject1_eul_angles.npz'
npz_file_2 = './data/processed_exports/subject2_eul_angles.npz'
npz_file_3 = './data/processed_exports/subject3_eul_angles.npz'
npz_file_4 = './data/processed_exports/subject4_eul_angles.npz'

npz_data_0, npz_data_1, npz_data_2, npz_data_3, npz_data_4 = \
    np.load(npz_file_0), np.load(npz_file_1), np.load(npz_file_2), np.load(npz_file_3), np.load(npz_file_4)

for s in range(n_subjects):
    exec(f"time_{s}, eul_right_{s}, eul_left_{s} = \
         npz_data_{s}['time'], npz_data_{s}['eul_right'], npz_data_{s}['eul_left']")


print("{:^9} {:^16} {:^16} {:^20} {:^17} {:^20} {:^17} {:^16} {:^16}".format\
      ("", "Shoulder", "Shoulder", "Shoulder", "Elbow", "Elbow", "Wrist", "Wrist", "Wrist"))
print("{:^9} {:^16} {:^16} {:^20} {:^17} {:^20} {:^17} {:^16} {:^16}".format\
      ("", "Abduction", "Rotation", "Extension", "Flexion", "Supination", "Deviation", "Supination", "Flexion"))
print("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12}".format\
      ("Subject", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max", "min", "max"))

#%% Plot the data vs time for diagnostics
i_end_s = np.zeros(n_subjects, dtype=int)
colors = ['pink', 'olive', 'gray', 'b', 'g', 'r', 'c', 'm', 'orange', 'purple', 'brown']

min_angles = np.zeros((n_subjects, n_dof))
max_angles = np.zeros((n_subjects, n_dof))

for s in range(n_subjects):   
    side = 'R' if s == 0 or s == 4 else 'L'

    # Arm angles (+ and - for right and left arm consistency)
    exec(f"R_cl_x_{s} =  -np.around(eul_right_{s}[:,3,0] * 180/np.pi, 2)")
    exec(f"R_cl_y_{s} =  np.around(eul_right_{s}[:,3,1] * 180/np.pi, 2)")
    exec(f"R_cl_z_{s} =  -np.around(eul_right_{s}[:,3,2] * 180/np.pi, 2)")
    exec(f"R_sh_abd_{s} = -np.around(eul_right_{s}[:,4,0] * 180/np.pi, 2)")
    exec(f"R_sh_rot_{s} =  np.around(eul_right_{s}[:,4,1] * 180/np.pi, 2)")
    exec(f"R_sh_ext_{s} = -np.around(eul_right_{s}[:,4,2] * 180/np.pi, 2)")
    exec(f"R_el_fle_{s} =  np.around(eul_right_{s}[:,5,0] * 180/np.pi, 2)")
    exec(f"R_el_sup_{s} =  np.around(eul_right_{s}[:,5,1] * 180/np.pi, 2)")
    exec(f"R_wr_dev_{s} = -np.around(eul_right_{s}[:,6,0] * 180/np.pi, 2)")
    exec(f"R_wr_sup_{s} =  np.around(eul_right_{s}[:,6,1] * 180/np.pi, 2)")
    exec(f"R_wr_fle_{s} = -np.around(eul_right_{s}[:,6,2] * 180/np.pi, 2)")

    exec(f"L_cl_x_{s} =  np.around(eul_left_{s}[:,3,0] * 180/np.pi, 2)")
    exec(f"L_cl_y_{s} =  np.around(eul_left_{s}[:,3,1] * 180/np.pi, 2)")
    exec(f"L_cl_z_{s} =  np.around(eul_left_{s}[:,3,2] * 180/np.pi, 2)")
    exec(f"L_sh_abd_{s} =  np.around(eul_left_{s}[:,4,0] * 180/np.pi, 2)")
    exec(f"L_sh_rot_{s} =  np.around(eul_left_{s}[:,4,1] * 180/np.pi, 2)")
    exec(f"L_sh_ext_{s} =  np.around(eul_left_{s}[:,4,2] * 180/np.pi, 2)")
    exec(f"L_el_fle_{s} = -np.around(eul_left_{s}[:,5,0] * 180/np.pi, 2)")
    exec(f"L_el_sup_{s} =  np.around(eul_left_{s}[:,5,1] * 180/np.pi, 2)")
    exec(f"L_wr_dev_{s} =  np.around(eul_left_{s}[:,6,0] * 180/np.pi, 2)")
    exec(f"L_wr_sup_{s} =  np.around(eul_left_{s}[:,6,1] * 180/np.pi, 2)")
    exec(f"L_wr_fle_{s} =  np.around(eul_left_{s}[:,6,2] * 180/np.pi, 2)")
    
    # Healthy arm angles:
    for d, dof_code in enumerate(dof_codes):
        exec(f"{dof_code}_{s} = {side}_{dof_code}_{s}")
        exec(f"min_angles[s, d], max_angles[s, d] = min({dof_code}_{s}), max({dof_code}_{s})")

    exec(f"ang_range_{s} = max_angles[s,:] - min_angles[s,:]")
    # print("ang_range: ", eval(f"ang_range_{s}")) 

    exec(f"ax_min_{s}, ax_max_{s} = np.min(min_angles[s,:]), np.max(max_angles[s,:])")

    print("{:<9} {:<8} {:<8} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12}".format("#"+str(s), \
    #   str(min_angles[s,0]), str(max_angles[s,0]), str(min_angles[s,1]), str(max_angles[s,1]), str(min_angles[s,2]), str(max_angles[s,2]), \
      str(min_angles[s,3]), str(max_angles[s,3]), str(min_angles[s,4]), str(max_angles[s,4]), str(min_angles[s,5]), str(max_angles[s,5]), \
      str(min_angles[s,6]), str(max_angles[s,6]), str(min_angles[s,7]), str(max_angles[s,7]), \
      str(min_angles[s,8]), str(max_angles[s,8]), str(min_angles[s,9]), str(max_angles[s,9]), str(min_angles[s,10]), str(max_angles[s,10])))

    # plot data
    i_start, i_end = 0, len(eval(f"time_{s}"))-1

    i_end_s[s] = min(i_end, len(eval(f"time_{s}"))-1)
    fig1, ax1 = plt.subplots(n_ax, 1, sharex=True, figsize=(n_ax*6, 10))
    ax1[0].set(position=(0.1, 0.5, 0.8, 0.45), ylim=(eval(f"ax_min_{s}")-10, eval(f"ax_max_{s}")+10), ylabel=f'Angles (deg)', title=f'{subject_labels[s]}')
    ax1[1].set(position=(0.1, 0.27, 0.8, 0.2), ylim=(-10, 10), ylabel=f'Angles_First derivative (deg/s)')
    ax1[2].set(position=(0.1, 0.05, 0.8, 0.2), ylim=(-10, 10), ylabel=f'Angles_Second derivative (deg/s^2)', xlabel='Time (s)')
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
    for t in range(2, len(eval(f"time_{s}"))):
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

    for d, dof_code in enumerate(dof_codes):
        exec(f"min_angles[{s}, {d}], max_angles[{s}, {d}] = min({dof_code}_{s}), max({dof_code}_{s})")

    print("{:<9} {:<8} {:<8} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<11} {:<8} {:<8} {:<8} {:<8} {:<8} {:<12}".format("#"+str(s), \
    #   str(min_angles[s,0]), str(max_angles[s,0]), str(min_angles[s,1]), str(max_angles[s,1]), str(min_angles[s,2]), str(max_angles[s,2]), \
      str(min_angles[s,3]), str(max_angles[s,3]), str(min_angles[s,4]), str(max_angles[s,4]), str(min_angles[s,5]), str(max_angles[s,5]), \
      str(min_angles[s,6]), str(max_angles[s,6]), str(min_angles[s,7]), str(max_angles[s,7]), \
      str(min_angles[s,8]), str(max_angles[s,8]), str(min_angles[s,9]), str(max_angles[s,9]), str(min_angles[s,10]), str(max_angles[s,10])))
    print()

    for n in range(n_ax):
        if i_start > len(eval(f"time_{s}")):
            ax1[n].axis('off')
            break
        for d in range(n_dof):
            ax1[n].plot(eval(f"time_{s}")[n:-1], np.diff(eval(f"{dof_codes[d]}_{s}")[:-1], n=n), label=f'{dof_names[d]}_filtered', linestyle='dashed', color=colors[d])
        
        annotations = {'Take': 'black', 'Jump': 'red', 'Switch': 'purple'}
        for label, color in annotations.items():
            for index in eval(f"{label.lower()}_ind_{s}"):
                if index < i_end_s[s]:
                    ax1[0].annotate(label, xy=(eval(f"time_{s}")[index], 90), xytext=(eval(f"time_{s}")[index], 150), arrowprops=dict(facecolor=color, shrink=0.05))

        for filt in eval(f"filt_ind_{s}"):
            if filt < i_end_s[s]:
                ax1[n].fill_between(eval(f"time_{s}")[filt], -200, 200, color='yellow', alpha=0.5)

        ax1[n].grid()
        ax1[n].xaxis.set_major_locator(plt.MultipleLocator(20))
    ax1[0].legend(loc='right')
    
    # npz_file = f'../data/processed_exports/filtered/subject{s}_{side}h_filtered_full_eul_angles.npz'
    # kwargs = {dof_codes[d]: eval(f"{dof_codes[d]}_{s}") for d in range(n_dof)}
    # kwargs['time'] = eval(f"time_{s}")
    # np.savez(npz_file, **kwargs)
    
    # print(f"Subject {s}, axis min: {eval(f'ax_min_{s}')}, axis max: {eval(f'ax_max_{s}')}")

dof_min, dof_max = np.min(min_angles, axis=0), np.max(max_angles, axis=0)

for s in range(n_subjects):
    print(f"Subject {s}, takes: {eval(f'len(take_ind_{s})')}, jumps: {eval(f'jump_ind_{s}')}, \
          switches: {eval(f'switch_ind_{s}')}, filtered: {eval(f'len(filt_ind_{s})')}")
    # print(f"Subject {s}, switches: {eval(f'switch_ind_{s}')}")

plt.show()

  
