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
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

from PIL import Image

from pylab import *



simplefilter(action='ignore', category=FutureWarning)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def height_rotation(experiment_ID):

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
    data3=data3*0.05
    data1=np.degrees(data1)/360


    df1 = pd.DataFrame(data1[:,0,1999], columns=['y1'])
    df1['y2'] = data[:,0,1999]
    df1['y3'] = reward[:,0,1999]
 

    #1D_sensory
    df2 = pd.DataFrame(data1[:,1,1999], columns=['y1'])
    df2['y2'] = data[:,1,1999]
    df2['y3'] = reward[:,1,1999]


    #3D_sensory
    df3 = pd.DataFrame(data1[:,2,1999], columns=['y1'])
    df3['y2'] = data[:,2,1999]
    df3['y3'] = reward[:,2,1999]
  

    #Binary_sensory
    df4 = pd.DataFrame(data1[:,3,1999], columns=['y1'])
    df4['y2'] = data[:,3,1999]
    df4['y3'] = reward[:,3,1999]
    



    Num_NO_C2=np.count_nonzero((18.75 <= df1['y2']) & (df1['y2'] <= 31.25))
    Num_1D_C2=np.count_nonzero((18.75 <= df2['y2']) & (df2['y2'] <= 31.25))
    Num_3D_C2=np.count_nonzero((18.75 <= df3['y2']) & (df3['y2'] <= 31.25))
    Num_B_C2=np.count_nonzero((18.75 <= df4['y2']) & (df4['y2'] <= 31.25))

    Num_C2=Num_NO_C2+Num_1D_C2+Num_3D_C2+Num_B_C2
    Num_CS=Num_NO_C2

    Ave_NO_C2=df1['y1'][df1['y2'].between(18.75,31.25)].mean()
    Ave_1D_C2=df2['y1'][df2['y2'].between(18.75,31.25)].mean()
    Ave_3D_C2=df3['y1'][df3['y2'].between(18.75,31.25)].mean()
    Ave_B_C2 =df4['y1'][df4['y2'].between(18.75,31.25)].mean()
    Ave_C2=np.floor(np.nanmean([Ave_NO_C2,Ave_1D_C2,Ave_3D_C2,Ave_B_C2]))
    Ave_CS=np.floor(np.nanmean([Ave_NO_C2]))
    

    return Ave_C2,Num_C2,Num_CS,Ave_CS




experiment_ID = "handmotorservo_C2"
Ave_C2,Num_C2,Num_CS2,Ave_CS2=height_rotation(experiment_ID)


experiment_ID = "handmotorservo_C3"
Ave_C3,Num_C3,Num_CS3,Ave_CS3=height_rotation(experiment_ID)


experiment_ID = "handmotorservo_C4"
Ave_C4,Num_C4,Num_CS4,Ave_CS4=height_rotation(experiment_ID)

experiment_ID = "handmotorservo_C5"
Ave_C5,Num_C5,Num_CS5,Ave_CS5=height_rotation(experiment_ID)


i=0

if i==0:
        print(Ave_C2,Num_C2,Ave_C3,Num_C3,Ave_C4,Num_C4,Ave_C5,Num_C5)
        fig = plt.figure(figsize= (15,10))
        ax = plt.axes()


        colors=['dodgerblue','r','purple','green','gold']

        plt.xlim([-1,25])
        plt.ylim([-5,200])
        plt.plot([-1,Ave_C4,Ave_C4,Ave_C5,Ave_C5,Ave_C3+1,Ave_C3+1],[Num_C4,Num_C4,Num_C5,Num_C5,Num_C3,Num_C3,-3],linewidth=8,color='b')
        plt.scatter([0,Ave_C2,Ave_C3+1,Ave_C4,Ave_C5],[0,Num_C2,Num_C3,Num_C4,Num_C5],marker='x', s=600, color=[colors[r] for r in range(0,5)],linewidth=16)

        plt.fill_between(
                x= [-1,Ave_C4,Ave_C4,Ave_C5,Ave_C5,Ave_C3+1,Ave_C3+1], 
                y1= [Num_C4,Num_C4,Num_C5,Num_C5,Num_C3,Num_C3,-5], 
                y2=-5,
                color= "b",
                alpha= 0.1)


        ###C4
        im = Image.open('C4.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_C4, Num_C4)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(-0.8, 0.4),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)


        ##C5
        im = Image.open('C5.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_C5, Num_C5)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(0.9,-0.6),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)



        ##C2
        im = Image.open('C2.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_C2, Num_C2)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(0, -0.9),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)


        ##C1
        im = Image.open('C1.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (0, 0)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(-0.5, -0.5),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)


        ##C3
        im = Image.open('C3.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_C3+1, Num_C3)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(-1, -0.5),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)

        ax = gca()
        fontsize = 14
        spines = ax.spines
        [i.set_linewidth(3) for i in spines.values()]

        # plt.xticks(rotation=45)
        ax.set_ylabel('# of data points inside target height range',fontsize=30)
        ax.set_xlabel('Mean rotation for data points inside target height range',fontsize=30)



        plt.xticks(np.arange(0, 30, 5),fontsize=32)
        plt.yticks(np.arange(0, 250, 50),fontsize=32)



        plt.tight_layout()

        experiment="Experiment" 
        output_dir= "./logs/{}/{}".format(experiment_ID,experiment)

        plt.savefig(output_dir+'/pareto_plot.pdf',dpi=2000)
        plt.show()

elif i==1:

        fig = plt.figure(figsize= (15,10))
        ax = plt.axes()


        colors=['dodgerblue','r','purple','green','gold']

        plt.xlim([-1,20])
        plt.ylim([-3,40])
        plt.plot([-1,Ave_CS4,Ave_CS4,Ave_CS5,Ave_CS5,Ave_CS3,Ave_CS3],[Num_CS4,Num_CS4,Num_CS5,Num_CS5,Num_CS3,Num_CS3,-3],linewidth=8,color='b')
        plt.scatter([0,Ave_CS2,Ave_CS3,Ave_CS4,Ave_CS5],[0,Num_CS2,Num_CS3,Num_CS4,Num_CS5],marker='x', s=600, color=[colors[r] for r in range(0,5)],linewidth=16)


        plt.fill_between(
                x= [-1,Ave_CS4,Ave_CS4,Ave_CS5,Ave_CS5,Ave_CS3,Ave_CS3], 
                y1= [Num_CS4,Num_CS4,Num_CS5,Num_CS5,Num_CS3,Num_CS3,-3], 
                y2=-3,
                color= "b",
                alpha= 0.1)


        ###C4
        im = Image.open('C4.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_CS4, Num_CS4)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(-0.8, 0.2),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)


        ##C5
        im = Image.open('C5.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_CS5, Num_CS5)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(-0.2,-1),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)



        ##C2
        im = Image.open('C2.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_CS2, Num_CS2)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(1, -0.9),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)


        ##C1
        im = Image.open('C1.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (0, 0)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(-0.1, -0.3),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)


        ##C3
        im = Image.open('C3.png')

        im = OffsetImage(im, zoom=0.25)
        im.image.axes = ax
        xy = (Ave_CS3, Num_CS3)
        # ax.plot(xy[0], xy[1], ".r")

        ab = AnnotationBbox(im, xy,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            box_alignment=(0.2, -0.5),
                            arrowprops=dict(arrowstyle="->",lw=6,color='k'),
                            bboxprops=dict(facecolor = "none", edgecolor='k', 
                                      lw = 6))

        ax.add_artist(ab)

        ax = gca()
        fontsize = 14
        spines = ax.spines
        [i.set_linewidth(3) for i in spines.values()]

        # plt.xticks(rotation=45)
        ax.set_ylabel('# of data points inside target height range',fontsize=30)
        ax.set_xlabel('Mean rotation for data points inside target height range',fontsize=30)



        plt.xticks(np.arange(0, 25, 5),fontsize=32)
        plt.yticks(np.arange(0, 40, 10),fontsize=32)



        plt.tight_layout()

        experiment="Experiment" 
        output_dir= "./logs/{}/{}".format(experiment_ID,experiment)

        plt.savefig(output_dir+'/pareto_plot_No_sensory.pdf',dpi=2000)
        plt.show()


