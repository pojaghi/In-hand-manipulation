import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import os
import sys
from warnings import simplefilter
from decimal import Decimal


simplefilter(action='ignore', category=FutureWarning)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

experiment_ID = "handmotorservo_Oct_28"
data="Result_data"
experiment="Experiment" 
if not os.path.exists("./logs/{}/{}".format(experiment_ID,experiment)):
        os.makedirs("./logs/{}/{}".format(experiment_ID,experiment))

data_dir = "./logs/{}/{}".format(experiment_ID,data)
output_dir= "./logs/{}/{}".format(experiment_ID,experiment)

# Reading the data
data=np.load(os.path.join(data_dir,'data_h.npy'))


learning_episode= ["250", "500", "750","1k","1.25k","1.5k","1.75k","2k"]
sensory_info = ['No Force', '1D Force','3D Force']
N=8
x = np.arange(N)    # the x locations for the groups

positions_p0 = 4*np.arange(N)-0.5
positions_p1 = 4*np.arange(N)
positions_p2 = 4*np.arange(N)+0.5
positions_p3 = 4*np.arange(N)+1


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)


total_timesteps = 2000000
episode_timesteps = 1000
total_episodes = int(total_timesteps/episode_timesteps)
x0=range(total_episodes)

sensory0=data[:,0,:]
sensory1=data[:,1,:]
sensory2=data[:,2,:]
sensory3=data[:,3,:]


#No sensory information 
for i in range(0,N):
	if i==10:
		globals()['s0_%s' % i] = sensory0[:,0:1] #No sensory
		globals()['s1_%s' % i] = sensory1[:,0:1] #1D Force 
		globals()['s2_%s' % i] = sensory2[:,0:1] #3D Forc
		globals()['s3_%s' % i] = sensory3[:,0:1] #Binary
	else:	
		globals()['s0_%s' % i] = sensory0[:,249*(i+1)+(i-1)*1] #No sensory
		globals()['s1_%s' % i] = sensory1[:,249*(i+1)+(i-1)*1] #1D Force 
		globals()['s2_%s' % i] = sensory2[:,249*(i+1)+(i-1)*1] #3D Force
		globals()['s3_%s' % i] = sensory3[:,249*(i+1)+(i-1)*1] #Binaray  Force
	

fig = plt.figure(figsize= (14,14))
ax = plt.axes()



p0_0 = plt.boxplot(
	[s0_0,s0_1,s0_2,s0_3,s0_4,s0_5,s0_6,s0_7],
	positions=positions_p0,
	notch=True,
	patch_artist=True,
	vert=True,
	showfliers=False,
	widths =0.4)
set_box_color(p0_0,'lightskyblue') 

for b in p0_0['boxes']:
    b.set_edgecolor('k') # or try 'black'
    b.set_linewidth(1)

p1_0 = plt.boxplot(
	[s1_0,s1_1,s1_2,s1_3,s1_4,s1_5,s1_6,s1_7],
	positions=positions_p1,
	notch=True,
	patch_artist=True,
	vert=True,
	showfliers=False,
	widths =0.4)

for b in p1_0['boxes']:
    b.set_edgecolor('k') # or try 'black'
    b.set_linewidth(1)



p2_0 = plt.boxplot(
	[s2_0,s2_1,s2_2,s2_3,s2_4,s2_5,s2_6,s2_7],
	positions=positions_p2,
	notch=True,
	patch_artist=True,
	vert=True,
	showfliers=False,
	widths =0.4)
set_box_color(p2_0,'darkblue') 

for b in p2_0['boxes']:
    b.set_edgecolor('k') # or try 'black'
    b.set_linewidth(1)

p3_0 = plt.boxplot(
	[s3_0,s3_1,s3_2,s3_3,s3_4,s3_5,s3_6,s3_7],
	positions=positions_p3,
	notch=True,
	patch_artist=True,
	vert=True,
	showfliers=False,
	widths =0.4)
set_box_color(p3_0,'blue') 

for b in p3_0['boxes']:
    b.set_edgecolor('k') # or try 'black'
    b.set_linewidth(1)



ax.set_xlabel('Episode #',fontsize=12)
ax.set_ylabel('Final Mean Height (when lifting and rotating is reward)',fontsize=12)
plt.legend([p0_0["boxes"][0], p1_0["boxes"][0],p2_0["boxes"][0],p3_0["boxes"][0]], ['No Force','1D Force','3D Force','Binary'], loc='upper left', fontsize='small',prop={'size':10})
ax.autoscale()
ax.set(xticks=positions_p2, xticklabels=learning_episode)
fig.subplots_adjust(left=.06, right=.95, bottom=.3)
plt.axhline(y=25, color='r', linestyle='--')
plt.axhline(y=40, color='r', linestyle='--')
plt.title('Box Plot: Final Mean Height Vs. Episode plot [60 Monte Carlo runs]', fontsize=14)
fig.savefig(output_dir+'/exp1_vs_height_boxplot.pdf')
plt.show()



# for i in range(0,N):
# 	globals()['F0_%s' % i],globals()['PV0_%s' % i] = f_oneway(globals()['s0_%s' % i][:,-1],globals()['s1_%s' % i][:,-1]) #No sensory
# 	globals()['F1_%s' % i],globals()['PV1_%s' % i] = f_oneway(globals()['s0_%s' % i][:,-1],globals()['s2_%s' % i][:,-1]) 
# 	globals()['F2_%s' % i],globals()['PV2_%s' % i] = f_oneway(globals()['s1_%s' % i][:,-1],globals()['s2_%s' % i][:,-1])
# 	globals()['F3_%s' % i],globals()['PV3_%s' % i] = f_oneway(globals()['s1_%s' % i][:,-1],globals()['s3_%s' % i][:,-1])
# 	globals()['F4_%s' % i],globals()['PV4_%s' % i] = f_oneway(globals()['s2_%s' % i][:,-1],globals()['s3_%s' % i][:,-1])
# 	globals()['PV0_%s' % i]=round(globals()['PV0_%s' % i],5)
# 	globals()['PV1_%s' % i]=round(globals()['PV1_%s' % i],5)
# 	globals()['PV2_%s' % i]=round(globals()['PV2_%s' % i],5)
# 	globals()['PV3_%s' % i]=round(globals()['PV3_%s' % i],5)
# 	globals()['PV4_%s' % i]=round(globals()['PV4_%s' % i],5)


# the_table = plt.table(cellText=[[PV0_0,PV1_0,PV2_0,PV3_0,PV4_0],[PV0_1,PV1_1,PV2_1,PV3_1,PV4_1],
# 	                            [PV0_2,PV1_2,PV2_2,PV3_2,PV4_2],[PV0_3,PV1_3,PV2_3,PV3_3,PV4_3],
# 	                            [PV0_4,PV1_4,PV2_4,PV3_4,PV4_4],[PV0_5,PV1_5,PV2_5,PV3_5,PV4_5],
# 	                            [PV0_6,PV1_6,PV2_6,PV3_6,PV4_6],[PV0_7,PV1_7,PV2_7,PV3_7,PV4_7]],
#                       rowLabels=learning_episode,
#                       colLabels=['P-Value:No Force and 1D Force ', 'P-Value:No Force and 3D Force ','P-Value:1D Force and 3D Force','P-Value:1D Force and binary','P-Value:3D Force and binaray'],
#                       loc='bottom',
#                       cellLoc="center",
#                       bbox=[0, -0.4, 1.0, 0.3])

# the_table.auto_set_font_size(False)
# the_table.set_fontsize(10)

# plt.show()
# fig.savefig(output_dir+'/exp2_vs_height_boxplot.pdf')
 


