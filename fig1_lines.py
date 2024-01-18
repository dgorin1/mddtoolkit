
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
# from utils.get_plot_name import get_plot_name


# Drew is gonna hardcode in his data because he already wrote the code to deal with the uncertainty-weighted chisq in matlab.. saved at
# /Users/drew/Documents/MATLAB_fromPC/Diffusion-Calculations/SyntheticDataExperiments/Initial DTAA experiments/ArArSynthExperiments


best_fit_exampleA = pd.read_csv("/Users/drew/Documents/MATLAB_fromPC/Diffusion-Calculations/SyntheticDataExperiments/Initial DTAA experiments/ArArSynthExperiments/best_fit_exampleA.csv", header=None)
slopes_exampleA = pd.read_csv("/Users/drew/Documents/MATLAB_fromPC/Diffusion-Calculations/SyntheticDataExperiments/Initial DTAA experiments/ArArSynthExperiments/slopes_example_A.csv", header=None)
best_fit_exampleB = pd.read_csv("/Users/drew/Documents/MATLAB_fromPC/Diffusion-Calculations/SyntheticDataExperiments/Initial DTAA experiments/ArArSynthExperiments/best_fit_exampleB.csv", header = None)
slopes_exampleB = pd.read_csv("/Users/drew/Documents/MATLAB_fromPC/Diffusion-Calculations/SyntheticDataExperiments/Initial DTAA experiments/ArArSynthExperiments/slopes_exampleB.csv", header = None)


fig = plt.figure()
gridspec.GridSpec(2,4)
ax1 = plt.subplot2grid((2,4), (0,0), colspan=1, rowspan=1)

# Plot all the goodness-of-fit data
plt.plot(
    range(2,len(best_fit_exampleA[:])+2),
    best_fit_exampleA[:],
    '--o',
    markersize = 11,
    color = (0.69, 0.69, 0.69),
    linewidth = 1,
    mec = 'black'
    )

# Single out the value that would be chosen by old algorithm
# I've hard-coded the data here so I can plot just the value that would have been chosen
plt.plot(
    6,
    best_fit_exampleA.loc[4],
    '--o',
    markersize = 11,
    color = (0, 0, 0),
    linewidth = 1,
    mec = 'black'
    )

 #format the plot
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Number of Points Included")
plt.ylabel("GOF * N")
plt.ylim(-.5,6.5)
plt.xlim(1.2,12.7)
ax1.set_box_aspect(1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.15)


# Add plot of the values
ax2 = plt.subplot2grid((2,4),(0,1),colspan=1,rowspan=1)

plt.plot(range(2,len(slopes_exampleA[:])+2),
         slopes_exampleA[:]*-83.14,
        '--o',
        markersize = 11,
        color = (0.69, 0.69, 0.69),
        linewidth = 1,
        mec = 'black',
        zorder = 100)

plt.plot(
    6,
    slopes_exampleA.loc[4]*-83.14,
    '--o',
    markersize = 11,
    color = (0, 0, 0),
    linewidth = 1,
    mec = 'black',
    zorder = 10
    )

# Show activation Energy on Plot

y = np.ones(100)*200
plt.plot(range(100),
          y,
          '-r',
          zorder = 1)



plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Number of Points Included")
plt.ylabel("Ea (kJ/mol)")
ax2.set_box_aspect(1)
plt.xlim(1.2,12.7)
plt.ylim(100,206)

fig.set_size_inches(w=15,h=7)
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.15)


#plt.savefig("/Users/drew/Documents/Berkeley_Fall_2023/MDD_Paper/Figure_1_Data/Actual_Figure_Plots/example_A_subplot.pdf")
#lt.close(fig)


# Repeat all steps for part B!!!!
# 
# 
#
#
#
# 
# 
#
#
#


ax1 = plt.subplot2grid((2,4), (1,0), colspan=1, rowspan=1)

# Plot all the goodness-of-fit data
plt.plot(
    range(2,len(best_fit_exampleB[:])+2),
    best_fit_exampleB[:],
    '--o',
    markersize = 11,
    color = (0.69, 0.69, 0.69),
    linewidth = 1,
    mec = 'black'
    )

# Single out the value that would be chosen by old algorithm
# I've hard-coded the data here so I can plot just the value that would have been chosen
plt.plot(
    3,
    best_fit_exampleB.loc[1],
    '--o',
    markersize = 11,
    color = (0, 0, 0),
    linewidth = 1,
    mec = 'black'
    )

 #format the plot
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Number of Points Included")
plt.ylabel("GOF * N")
plt.ylim(-.5,6.5)
plt.xlim(1.2,12.7)
ax1.set_box_aspect(1)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.15)


# Add plot of the values
ax2 = plt.subplot2grid((2,4),(1,1),colspan=1,rowspan=1)

plt.plot(range(2,len(slopes_exampleB[:])+2),
         slopes_exampleB[:]*-83.14,
        '--o',
        markersize = 11,
        color = (0.69, 0.69, 0.69),
        linewidth = 1,
        mec = 'black',
        zorder = 10)

plt.plot(
    3,
    slopes_exampleB.loc[1]*(-83.14),
    '--o',
    markersize = 11,
    color = (0, 0, 0),
    linewidth = 1,
    mec = 'black',
    zorder = 100
    )

# Show activation Energy on Plot

y = np.ones(100)*200
plt.plot(range(100),
          y,
          '-r',
          zorder = 0)



plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Number of Points Included")
plt.ylabel("Predicted Ea (kJ/mol)")
ax2.set_box_aspect(1)
plt.xlim(1.2,12.7)
plt.ylim(100,206)


fig.tight_layout()

for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.15)

plt.savefig("/Users/drew/Documents/Berkeley_Fall_2023/MDD_Paper/Figure_1_Data/Actual_Figure_Plots/test.pdf")
plt.close(fig)
