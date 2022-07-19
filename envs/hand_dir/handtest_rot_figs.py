import json
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import colorsys
import pickle
from warnings import simplefilter
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'cm'

extended_view = False
sensory_versions = 4
select_sensory = range(sensory_versions)

RL_method = "PPO1"
# NUMBER of Monte Carlo Runs
total_MC_runs = 60
experiment_ID = "handmotorservo_C1"
data="Result_data"

if not os.path.exists("./logs/{}/{}".format(experiment_ID,data)):
    os.makedirs("./logs/{}/{}".format(experiment_ID,data))


total_timesteps = 2000000
episode_timesteps = 1000
total_episodes = int(total_timesteps/episode_timesteps)

#Reward 
episode_rewards_all = np.zeros([total_MC_runs, sensory_versions, total_episodes])

#Height
episode_height = np.zeros([total_MC_runs, sensory_versions, total_episodes,episode_timesteps])


# Between height 18.75 to 31.25

episode_height_range = np.zeros([total_MC_runs, sensory_versions, total_episodes])

#Degree
episode_degree = np.zeros([total_MC_runs, sensory_versions, total_episodes,episode_timesteps])

# Height at last 3 seconds
episode_height_3sec = np.zeros([total_MC_runs, sensory_versions, total_episodes,300])


#Reward rotation
episode_rewards_rot_all = np.zeros([total_MC_runs, sensory_versions, total_episodes])

#Reward height
episode_rewards_height_all = np.zeros([total_MC_runs, sensory_versions, total_episodes])


mc_cntrr=0
for sensory_value in range(sensory_versions):
	sensory_value_str = "sensory_{}".format(sensory_value)
	for mc_cntr in range(1,total_MC_runs+1):
			if sensory_value==0:
				
				if mc_cntr<=20:
					mc_cntrr=0
				elif 20<mc_cntr<=40:
					mc_cntrr=4

				elif 40<mc_cntr<61:	
					mc_cntrr=8
				else:
				    mc_cntrr=12	
          
			elif sensory_value==1:
				
				if mc_cntr<=20:
					mc_cntrr=1
				elif 20<mc_cntr<=40:
					mc_cntrr=5

				elif 40<mc_cntr<61:	
					mc_cntrr=9
				else:
				    mc_cntrr=13
		    
			elif sensory_value==2:
				if mc_cntr<=20:
					mc_cntrr=2
				elif 20<mc_cntr<=40:
					mc_cntrr=6

				elif 40<mc_cntr<61:	
					mc_cntrr=10
				else:
				    mc_cntrr=14

			elif sensory_value==3:
				if mc_cntr<=20:
					mc_cntrr=3
				elif 20<mc_cntr<=40:
					mc_cntrr=7

				elif 40<mc_cntr<61:	
					mc_cntrr=11
				else:
				    mc_cntrr=15

		    

			jsonFile = open("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/PPO1/{}/PPO1_{}Monitor/openaigym.episode_batch.{}.Monitor_info.stats.json".format(experiment_ID,sensory_value_str,mc_cntr,mc_cntrr))
			jsonString = jsonFile.read()
			jsonData = json.loads(jsonString)
			print("sensory_info: ", sensory_value , "mc_cntr: ", mc_cntr)
			# print(np.array(jsonData['episode_rewards']).shape,np.array(jsonData['episode_height']).shape)
			episode_rewards_all[mc_cntr-1, sensory_value] = np.array(jsonData['episode_rewards'])
			episode_rewards_rot_all[mc_cntr-1, sensory_value] = np.array(jsonData['episode_rotation_rewards'])
			episode_rewards_height_all[mc_cntr-1, sensory_value] = np.array(jsonData['episode_height_rewards'])
			episode_height[mc_cntr-1, sensory_value,:,:] = np.array(jsonData['episode_height'])
			episode_height_3sec[mc_cntr-1, sensory_value,:,:]=np.array(jsonData['episode_last3sec_height'])
			episode_degree[mc_cntr-1, sensory_value,:,:] = np.array(jsonData['episode_degree'])

reward_to_displacement_coeficient  = 1
episode_displacement_all = episode_rewards_all*reward_to_displacement_coeficient

#Height
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'e_h.npy'), episode_height)
print("episode_height.shape:",episode_height.shape)
episode_height_all=np.mean(1000*(episode_height)-34,axis=3)
episode_height_b=1000*(episode_height)-34

episode_height_3sec_all=np.mean(1000*(episode_height_3sec)-34,axis=3)
print("episode_height_3sec.shape:",episode_height_3sec.shape,"episode_height_all.shape:", episode_height_all.shape)



for i in range(total_MC_runs):
	for l in range(sensory_versions):
		for j in range(total_episodes):
				episode_height_range[i,l,j]=100*(np.count_nonzero((18.75 < episode_height_b[i,l,j,:]) & (episode_height_b[i,l,j,:]< 31.25)))/1000



np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'data.npy'), episode_displacement_all)
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'data_h.npy'), episode_height_all)
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'degree.npy'), episode_degree)
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'data_h_range.npy'), episode_height_range)
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'data_h_3sec.npy'), episode_height_3sec_all)
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'data_rotation.npy'), episode_rewards_rot_all)
np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'data_height_reward.npy'), episode_rewards_height_all)

np.save(os.path.join("/Users/ucsc/Desktop/custom_gym/envs/hand_dir/logs/{}/{}".format(experiment_ID,data), 'degree.npy'), episode_degree)

print(episode_displacement_all.shape,"episode_displacement_all.shape")
episode_displacement_average = episode_displacement_all.mean(0)
episode_displacement_std = episode_displacement_all.std(0)

#episode_displacement_std = episode_displacement_all.std(2)
# Smean = episode_rewards_all.mean(2)


final_displacement = np.zeros([total_MC_runs, sensory_versions])
print (final_displacement.shape,"final_displacement.shape")




sensory_info_full = ['none', 'one SD', 'Normal Force','one SD', '3D Force', 'one SD']
sensory_info = ['No Force', '1D Force','3D Force','Binary Force']



## Figure 1
fig = plt.figure(figsize= (7,4))

for sensory_value in select_sensory:
	x0=range(total_episodes)
	y0=episode_displacement_average[sensory_value, :]
	std0 = episode_displacement_std[sensory_value,:]
	print(std0)
	plt.plot(x0, y0, alpha=.75)
	plt.fill_between(x0, y0-std0/2, y0+std0/2, color=colorsys.hsv_to_rgb((8.75-sensory_value)/14,1,.75), alpha=0.20)
	plt.legend(sensory_info, fontsize='small',loc='upper left')

plt.xlabel('Episode #', fontsize=10)
plt.ylabel('Mean Reward', fontsize=10)
plt.title('Task: Mean Reward vs. Episode Plots', fontsize=12)
plt.axhline(y=500, color='b', linestyle='--')
plt.grid()
plt.show()
 

for sensory_value in range(sensory_versions):
	final_displacement[:,sensory_value] = episode_displacement_all[:,sensory_value,-1]
	print(np.shape(final_displacement))
	



# boxplot figure
s0 = final_displacement[:,0]
# print(s0.shape,"parmi")
# s0=[x for x in s0 if x > -300]
# s0 = np.array(s0)
# print('final_displacement_s0:',s0,sum(s0 > -300))


s1 = final_displacement[:,1]
# s1=[x for x in s1 if x > -300]
# s1 = np.array(s1)
# print('final_displacement_s1:',s1,sum(s1 > -300))



s2 = final_displacement[:,2]
# s2=[x for x in s2 if x > -300]
# s2 = np.array(s2)
# print('final_displacement_s2:',s2,sum(s2 > -300))


s3 = final_displacement[:,3]

s = [s0,s1,s2,s3]


meanlineprops = dict(linestyle='--', linewidth=2, color='green')
medianlineprops =dict(linestyle='-', linewidth=2, color='red')
plt.ylim(-3000, 4000)
plt.boxplot(s, whis='range', showfliers=True, showmeans=True, meanline= False, notch=True, patch_artist=True, medianprops = medianlineprops, meanprops=meanlineprops, widths = 0.25)
plt.xticks([1,2,3,4], sensory_info, fontsize=8)
plt.title('Final Reward', fontsize=8)
#plt.grid()

plt.show()

print ('s0 std:', s0.std(), 's1 std:', s1.std())



# average plot
x2=range(sensory_versions)
y2 = final_displacement.mean(0)
std2 = final_displacement.std(0)
plt.plot(x2, y2, '--',color='black',alpha=.1)
# plt.fill_between(x2, y2-std2/2, y2+std2/2, alpha=0.25, edgecolor='C9', facecolor='C9')
plt.errorbar(x2, y2,yerr=std2/2,color='black',alpha=.2,animated=True)
for sensory_value in range(sensory_versions):
	plt.plot(x2[sensory_value], y2[sensory_value], 'o',alpha=.9, color=colorsys.hsv_to_rgb((8.75-sensory_value)/14,1,.75))
plt.xlabel('Sensory', fontsize=8)
plt.ylabel('Reward', fontsize=8)
plt.xticks(range(sensory_versions), sensory_info, fontsize=8)
plt.yticks( fontsize=8)
plt.title('Average Final Reward', fontsize=8)
plt.show()



