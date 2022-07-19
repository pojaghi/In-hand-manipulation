import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import os
import sys
from warnings import simplefilter
from decimal import Decimal
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches


simplefilter(action='ignore', category=FutureWarning)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

experiment_ID = "handmotorservo_C2"
data="Result_data"
experiment="Experiment" 

if not os.path.exists("./logs/{}/{}".format(experiment_ID,experiment)):
        os.makedirs("./logs/{}/{}".format(experiment_ID,experiment))

data_dir = "./logs/{}/{}".format(experiment_ID,data)
output_dir= "./logs/{}/{}".format(experiment_ID,experiment)


## Reading the data of reward
data=np.load(os.path.join(data_dir,'e_h.npy'))

data=np.mean(data,axis=3)
data2=np.load(os.path.join(data_dir,'degree.npy'))
data3=np.load(os.path.join(data_dir,'data_rotation.npy'))

reward=np.load(os.path.join(data_dir,'data.npy'))
data1=np.zeros([60, 4, 2000])
for i in range(2000):
  data1[:,:,i]=data2[:,:,i,999]



data=1000*data-34
##Theta (angular displacment) 
data3=data3*0.05
# print(data3)
data1=np.degrees(data1)/360

# ======================================== 
# construct cmap
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
#No_sensory
df1 = pd.DataFrame(data1[:,0,1999], columns=['y1'])
df1['y2'] = data[:,0,1999]
df1['y3'] = reward[:,0,1999]
df1=df1.sort_values('y3')
df1= df1.reset_index(drop=True)


#1D_sensory
df2 = pd.DataFrame(data1[:,1,1999], columns=['y1'])
df2['y2'] = data[:,1,1999]
df2['y3'] = reward[:,1,1999]
df2=df2.sort_values('y3')
df2= df2.reset_index(drop=True)


#3D_sensory
df3 = pd.DataFrame(data1[:,2,1999], columns=['y1'])
df3['y2'] = data[:,2,1999]
df3['y3'] = reward[:,2,1999]
df3=df3.sort_values('y3')
df3= df3.reset_index(drop=True)

#Binary_sensory
df4 = pd.DataFrame(data1[:,3,1999], columns=['y1'])
df4['y2'] = data[:,3,1999]
df4['y3'] = reward[:,3,1999]
df4=df4.sort_values('y3')
df4= df4.reset_index(drop=True)


colormap = cm.jet
normalize = mcolors.Normalize(vmin=0, vmax=240)
color=[colormap(normalize(r)) for r in range(0,240)]

dfc = pd.DataFrame(color, columns=['color1','color2','color3','color4'])



dfl=pd.concat([df1['y3'],df2['y3'],df3['y3'],df4['y3']],axis=0,ignore_index=True)
dfl=dfl.sort_values()


dfc.index=dfl.index


dfl=pd.merge(dfl, dfc, how = 'inner', right_index = True, left_index = True)




dfc1=dfl.loc[dfl.index<60]
dfc1=dfc1.drop('y3',axis=1)



dfc2 = dfl[dfl.index.to_series().between(60,120)]
dfc2=dfc2.drop('y3',axis=1)



dfc3 = dfl[dfl.index.to_series().between(120,180)]
dfc3=dfc3.drop('y3',axis=1)


dfc4=dfl.loc[dfl.index>=180]
dfc4=dfc4.drop('y3',axis=1)


records = dfc1.to_records(index=False)
result1 = list(records)


records2 = dfc2.to_records(index=False)
result2 = list(records2)

records3 = dfc3.to_records(index=False)
result3 = list(records3)

records4 = dfc4.to_records(index=False)
result4 = list(records4)


df1=df1


#graph = sns.jointplot(x=df1.y1, y=df1.y2, color='w',cmap=my_cmap,height=9,marginal_kws=dict(bins=[0,18.75,32.25,70]),marginal_ticks=False)

sns.set(style="ticks", color_codes=False)

graph = sns.JointGrid(x=df1.y1, y=df1.y2,height=11)
graph = graph.plot_joint(plt.scatter, color='r', edgecolor="white",linewidths=30)
graph.fig.set_figwidth(12)
graph.fig.set_figheight(10)

graph.ax_marg_x.hist(df1.y1,color='lightpink',bins=[-10,5,10,15,20,25,30,35],edgecolor='black',linewidth=2)
graph.ax_marg_y.hist(df1.y2,color='lightpink',bins=[0,18.75,32.25,70],orientation="horizontal",edgecolor='black',linewidth=2)



plt.tight_layout()

graph.ax_marg_x.set_ylim([-10,60])
graph.ax_marg_y.set_xlim([-10,60])



graph.x = df1.y1
graph.y = df1.y2

nValues = np.arange(60)

print(nValues.min(),nValues.max())
# setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=0, vmax=240)

colormap = cm.jet


#color=[colormap(normalize(r)) for r in range(0,60)]
#graph.plot_joint(plt.scatter, marker='x', s=60,color=[colormap(normalize(r)) for r in range(0,60)])

graph.plot_joint(plt.scatter, marker='x', s=220,color=[result2[r] for r in range(0,60)],linewidth=2.5)




# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(nValues)



plt.xlabel("# of completed rounds of rotation",fontsize=22)
plt.ylabel("# Mean height",fontsize=22)
plt.xlim([0,35])
#plt.xlim([0,20])
plt.ylim([-1,65])
from pylab import *
ax = gca()
fontsize = 14
spines = ax.spines
[i.set_linewidth(3) for i in spines.values()]



graph.ax_marg_y.tick_params(labeltop=True,labelsize=22,length=10)
graph.ax_marg_y.grid(True, axis='x', ls=':',linewidth=2.5,color='k')
graph.ax_marg_y.xaxis.set_major_locator(MaxNLocator(2))
plt.tick_params(direction='out', length=6, width=2, colors='k',
               grid_color='k', grid_alpha=0.5)





plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()

graph.ax_marg_x.tick_params(labelleft=True,labelsize=22,length=10)
graph.ax_marg_x.grid(True, axis='y', ls=':',linewidth=2.5,color='k')
graph.ax_marg_x.yaxis.set_major_locator(MaxNLocator(2))





left, bottom, width, height = (0, 18.75, 35, 12.5)


#left, bottom, width, height = (0, 18.75, 20, 12.5)

rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="darkgreen",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect)

# xy=(22, 32)
ann = plt.annotate("$±25\%$ of the target height (25mm)",
                   xy=(15, 31.5), xycoords='data',
                   xytext=(15, 31.5), textcoords='offset points',
                   horizontalalignment="center",
                   # arrowprops=dict(arrowstyle="->", lw=1,color='green'),
                   size=18,
                   color='darkgreen'
                   )
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
# make new ax object for the cbar
cbar_ax = graph.fig.add_axes([.85, .2, .03, .5])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.pcolor(np.arange(20).reshape(4,5))
cbar=plt.colorbar(scalarmappaple,cax=cbar_ax)


cbar.set_ticks([0,80,160,240])
# cbar.set_ticklabels(['-2820','-2585','-2348','-2111', '-1874','-1631'])
cbar.ax.get_yaxis().labelpad = 30
ticklabs = ['-2820','-2424','-2028','-1630']
cbar.ax.set_yticklabels(ticklabs, fontsize=20)
cbar.outline.set_linewidth(1.5)


cbar.set_label('Monte Carlo Run (Final Reward)', rotation=270,fontsize=22)
#cbar.set_ticks([s.colorbar.vmin + t*(s.colorbar.vmax-s.colorbar.vmin) for t in cbar.ax.get_yticks()])


plt.savefig(output_dir+'/rot_height_Binary.png',dpi=2000,bbox_inches='tight')
plt.show()





# import seaborn as sns
# import matplotlib.pylab as plt
# a=data[:,2,1999]
# b=data1[:,2,1999]


# df=pd.DataFrame([a,b])
# df=df.T
# print(df.reset_index(drop=True))

# sns.heatmap(df, linewidths = 0.30, annot = True)
# # ax = sns.heatmap([data[:,3,1999],data1[:,3,1999]], linewidth=0.5)
# plt.show()


# for n in range(0,60):
# 	plt.plot(data1[n,2,0:200], 100*data[n,2,:200],'o-')

# plt.xlabel("rotation")
# plt.ylabel("height (mm)")
# plt.ylim(0, 100)
# plt.show()