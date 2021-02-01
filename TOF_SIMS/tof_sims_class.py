import numpy as np
import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import ceil,sqrt
from functools import lru_cache
import plotly.graph_objects as go
import plotly.express as px
import os
#from threading import Thread,RLock

#remember the . before thread_classes !
#from .thread_classes import Flattener, give_threads, start_batch

cache_size = 3 #value for lru_cache

class TOF_SIMS :
    """
    Class for manipulation of tof_sim datasets
    """
    #class attribute to count the number of datasets opened
    datasets_opened = 0

    dim = {"x":1,"y":2,"z":0,"n":3} #to find correspondence between letters and axes


    def __init__(self,filename):
        """
        filename : str
          filepath of TOF-SIM dataset
        fibimages : 2d-numpy array
          contains SEM image before FIB-ing
        file_name : str
          file name
        file_path : str
          path of file
        buf_times : ?
          function unknown
        TPS2: 3D numpy array [z,x,y]
          function unknown
        peak_data: 4D numpy array [z,x,y,250]
          contains three spatial dimensions (x,y,z) and isotope mass (n = 250)
        peak_table: 1D numpy array
          contains reference for peak integration
        """


        self.colormap = "viridis"
        #open hdf5 file
        print("Cache size = ",cache_size)
        f = h5py.File(filename,'r')
        self._file_name = filename.split("/")[-1]
        print("Opening {}".format(self._file_name))
        self.file_path = filename[0:-len(filename.split("/")[-1])] #remove characters corresponding to filename
        self.fibimage = f['FIBImages']['Image0000']['Data'][()]
        self._buf_times = f['TimingData']['BufTimes']
        self._TPS2 = f['TPS2']["TwData"]

        #  The [()] means save it as a numpy array, not reference in h5py
        self._peak_data = f['PeakData']["PeakData"][()]
        self._peak_table = f['PeakData']["PeakTable"]
        self._sum_spectrum = f['FullSpectra']['SumSpectrum'][()] #original working
        print("Extraction of sum_spectrum done")

        #folowing operation very long, removed for debugging faster
        #self._event_list = f['FullSpectra']['EventList'][()]
        #print("Extraction of event_list done")
        print("Extraction of event_list done removed for debugging, remember to put it back")

        self._mass_axis = f['FullSpectra']['MassAxis'][()]
        print("Extraction of mass_axis done")

        self._sum_mass = pd.DataFrame({"Sum Spectrum":self._sum_spectrum,"Mass Axis":self._mass_axis})

        #increment dataset_opened each time a new dataset is opened (to limit memory usage at some point; not implemented)
        TOF_SIMS.datasets_opened +=1

    def plot_FIBImage(self, cmap = 'gray',figsize = (10,10)):
        """
        Method displaying FIBImages
        """
        fig = plt.figure(figsize = figsize)
        plt.imshow(self.fibimage,cmap=cmap)
        plt.colorbar()
        #plt.show()
        #plt.close() to prevent jupyter from displaying image when saving it
        # then return figure to be able to save it
        plt.close()
        return fig


    def opened(TOF_SIMS):
        """
        Class method displaying how many datasets have been opened
        """
        print("{} datasets have been opened".format(TOF_SIMS.datasets_opened))
    opened = classmethod(opened)


    def plot_buf_times(self, cmap = "viridis" ):
        """
        Plots BufTimes
        """
        plt.imshow(self._buf_times,cmap=cmap)
        plt.colorbar()
        plt.show()


    def plot_section_of_3D_dataset(self,three_D_array,section,mode,cmap = "viridis"):
        """
        Method ploting a given section of a 3D dataset

        three_D_array : 3D numpy array
        section : int
          index of frame to plot
        mode : str
          "x" , "y" or "z" : axis onto which selection will be applied
        cmap : "str
          pyplot color maps

        """
        def x(section):
            plt.imshow(three_D_array[:,section,:],cmap=cmap)
        def y(section):
            plt.imshow(three_D_array[:,:,section],cmap=cmap)
        def z(section):
            plt.imshow(three_D_array[section,:,:],cmap=cmap)

        plot_mode = {"x":x,"y":y,"z":z} #store the functions defined above in a dictionnary
        plot_mode[mode](section)  #run the corresponding functionwith section as parameter
        plt.colorbar()
        plt.show()


    def plot_sections_of_4D_dataset(self, four_D_array, cmap ,x ,y ,z ,n , mode, start , n_plotx, n_ploty, figsize ):
        """
        Function displaying the PeakData dataset
        four_D_array : 4D numpy array
          data to display

        mode : str
          dimension to parse, can be x, y, z or n.   Must provide two dimensions,
          the first one is a fixed integer, the second is the dimension to parse through.
          exemple xn would have x fixed at a given value (e.g. x = 0) and dimension n parsed through

        start : value to start parsing with

        nplot_x and nplot_y : number of plots in x/y dimensions
        """
        fig, axs = plt.subplots(n_plotx , n_ploty, figsize=(figsize , figsize))

        for i in range(n_plotx):
            for j in range(n_ploty):
                try:
                    if mode == "xn":
                        axs[i, j].imshow(four_D_array[ : , x ,  : , start ],cmap=cmap)
                    elif mode == "nx":
                        axs[i, j].imshow(four_D_array[ : , start ,  : , n ],cmap=cmap)
                    elif mode == "xy":
                        axs[i, j].imshow(four_D_array[ : , x , start  , : ],cmap=cmap)
                    elif mode == "yx":
                        axs[i, j].imshow(four_D_array[ : , start , y  , : ],cmap=cmap)
                    elif mode == "xz":
                        axs[i, j].imshow(four_D_array[ start , x ,  : , : ],cmap=cmap)
                    elif mode == "zx":
                        axs[i, j].imshow(four_D_array[ z , start , :  , : ],cmap=cmap)
                    elif mode == "yz":
                        axs[i, j].imshow(four_D_array[ start , : ,  y , : ],cmap=cmap)
                    elif mode == "zy":
                        axs[i, j].imshow(four_D_array[ z , : ,  start , : ],cmap=cmap)
                    elif mode == "ny":
                        axs[i, j].imshow(four_D_array[ : , : ,  start , n ],cmap=cmap)
                    elif mode == "yn":
                        axs[i, j].imshow(four_D_array[ : , : ,  y , start ],cmap=cmap)
                    elif mode == "nz":
                        axs[i, j].imshow(four_D_array[ start , : ,  : , n ],cmap=cmap)
                    elif mode == "zn":
                        axs[i, j].imshow(four_D_array[ z , : ,  : , start ],cmap=cmap)
                    #set the title
                    axs[i , j ].set_title(start)
                    axs[i , j ].axis('off')
                    start +=1
                except:
                    pass
        plt.show()

    def plot_peak_data_sections(self,cmap = "viridis" ,x=0,y=0,z=0, mass = 1 ,fixed_dimension = "x" , parsed_dimension = "m",start = 0,n_plotx = 4, n_ploty = 4, figsize = 18):
        """
        """
        if fixed_dimension == "m":
            fixed_dimension = "n"
        if parsed_dimension == "m":
            parsed_dimension = "n"

        if (fixed_dimension not in "xyzn") or (parsed_dimension not in "xyzn") :
            raise ValueError("fixed_dimension and parsed_dimension variables must be x,y,z or m(mass) ")

        if fixed_dimension == parsed_dimension:
            raise ValueError("fixed_dimension must be different to parsed_dimension")

        if (not fixed_dimension.isalpha()) or not (parsed_dimension.isalpha()):
            raise TypeError("fixed_dimension and parsed_dimension variables must be strings (either x,y,z or m(mass)) ")

        mode = fixed_dimension + parsed_dimension

        self.plot_sections_of_4D_dataset(self._peak_data,cmap ,x,y,z,mass,mode,start,n_plotx , n_ploty, figsize)


    def plot_TSP2_section(self,cmap = "viridis", mode = "z", section = 0 ):
        """
        plot TSP2
        """
        #assertions to be added here
        self.plot_section_of_3D_dataset(self._TPS2,section,cmap,mode = "mz")

    def plot_unique_section_4D_dataset(self, four_D_array, cmap ,x ,y ,z ,n , mode ):
        """
        Function displaying the PeakData dataset
        four_D_array : 4D numpy array
          data to display
        mode : str
          dimension to display must be two combination of  x, y, z or n.
        """
        if mode == "xn" or mode == "nx":
            plt.imshow(four_D_array[ z , : , : , n ],cmap=cmap)
        elif mode == "ym" or mode == "my":
            plt.imshow(four_D_array[ z , x , : , : ],cmap=cmap)
        elif mode == "zm" or mode == "mz":
            plt.imshow(four_D_array[ : , x , y , : ],cmap=cmap)
        elif mode == "zy" or mode == "yz":
            plt.imshow(four_D_array[ : , x , : , n ],cmap=cmap)
        elif mode == "zx" or mode == "xz":
            plt.imshow(four_D_array[ : , : , y , n ],cmap=cmap)
        elif mode == "xy" or mode == "yx":
            plt.imshow(four_D_array[ z , : , : , n ],cmap=cmap)

        plt.colorbar()
        plt.show()


    def plot_peak_data__single_frame(self,mode = "xy", cmap="viridis",x=0,y=0,z=0,m=1):
        """
        Plot single peak_data frame
        mode : str
          xy, mz or any two-combination, which axis to plot
        x,y,z,m :   int
          defines which frame to plot
        """
        mode = self.trans_mode(mode) #change m to n
        self.plot_unique_section_4D_dataset(self._peak_data, cmap ,x ,y ,z ,m , mode )



    def transpose_then_max_proj(self,mode,fourD_array):
        """
        transposes and transforms the array
        """
        dim = TOF_SIMS.dim
        #print("dim[mode[0]],dim[mode[1]],dim[mode[2]],dim[mode[3]]",dim[mode[0]],dim[mode[1]],dim[mode[2]],dim[mode[3]])
        new_array = np.transpose(fourD_array,axes=(dim[mode[0]],dim[mode[1]],dim[mode[2]],dim[mode[3]] ))
        return np.sum(new_array,3) #sum over the last axis


    def max_proj(self,fourD_array, mode , figsize = (13,13),dpi = 100,cmap = "viridis" ):
        """
        Create a max projection with first two letters from mode as axes to be displayed, third is the one to be fixed, last the one to be summd
        """
        #transposes the dataset to match order of mode
        proj = self.transpose_then_max_proj(mode,fourD_array)

        print("Projection shape:",proj.shape)

        n_plotx = n_ploty = ceil(sqrt( proj.shape[2]))

        #f = plt.figure(figsize = (10,8), dpi=100) ##added

        fig, axs = plt.subplots(n_plotx , n_ploty, figsize = figsize)
        index = 0
        for i in range ( ceil(sqrt( proj.shape[2])) ):
            for j in range ( ceil(sqrt( proj.shape[2])) ):
                try:
                    axs[i, j].imshow(proj[ : , : , index ],cmap=cmap)
                    axs[i , j ].set_title(index)
                    axs[i , j ].axis('off')
                except:
                    pass
                index +=1

        plt.show()
        #return fig object to allow user to save it using .savefig() method from matplotlib
        return fig


    def max_proj_peak_data(self, axes_displayed = "xy", axis_parsed = "z", axis_max_projection = "n", figsize = 20, cmap = "viridis" ):
        """
        Create a max projection with first two axes to be displayed,
           third axis to be parsed, last is axis to be summed
        mode :  str
          must be a four combination of x,y,z,m
        figsize : pyplot argument for  figure size
        """
        axes_displayed = axes_displayed[::-1]
        print("Subplot represent individual " + axis_parsed + " with sub plots row = " + axes_displayed[0] + "-axis, sub plots column = " + axes_displayed[1] + "-axis projected over " + axis_max_projection + "-axis" )
        mode = self.trans_mode(axes_displayed + axis_parsed + axis_max_projection) #add axes in correct order and changes n to m
        self.max_proj(self._peak_data,mode,figsize,cmap = cmap)


    def trans_mode(self,mode):
        """
        Sub-function used by max_proj_peak_data to change m to n, should be removed later on
        """
        returned_mode = ""
        for i in mode:
            if i == "m":
                returned_mode+="n"
            else:
                returned_mode += i
        return returned_mode



    def max_proj_isotope(self,four_D_array, mode , isotope  ):
        """
        Create a max projection with first two letters from mode as axes to be summed.
        Returns an array.
        This function is cached for every masses and three projections possible
        """
        proj = np.sum(four_D_array,TOF_SIMS.dim[mode]) #sum over the corresponding axis

        #sum the between start and end
        if isotope != "all":
            return proj[:,:,isotope]
        else:
            return proj[:,:,:]


    def plot_max_proj_peak_data(self,mode = "z", mass = "all",cmap = "viridis" , figsize=(5,5) ):
        projection = self.max_proj_isotope(self.peak_data,mode,mass)
        fig = plt.figure(figsize=figsize)
        plt.imshow(projection,cmap=cmap)
        plt.colorbar()
        #plt.show()
        #plt.close() to prevent jupyter from displaying twice then return fig to be able to save it
        plt.close()
        return fig



    def overlay_max_proj(self,  alpha = 0.5 , **isotopes):
      """
      Overlay max projection (z axis) for any number of isotopes
      """
      flag = True
      for isotope,list_value in isotopes.items():
          #value[0] :  list_value[0] : mass and  list_value[1] : color map
          print(isotope,list_value[0],list_value[1])
          proj = self.max_proj_isotope(self.peak_data,"z",list_value[0])
          if flag:
              plt.imshow(proj,cmap=list_value[1])
              flag = False
          else:
              plt.imshow(proj,cmap=list_value[1],alpha=alpha)
      plt.colorbar()
      plt.show()



    @lru_cache(maxsize = cache_size)
    def filter_sum_spectrum_vs_mass_axis(self,mass_min,mass_max,sum_spectrum_min,sum_spectrum_max):
        df = self._sum_mass
        #print(df.head())
        #print(df.columns.values)
        df = df[df["Mass Axis"] > mass_min]
        df = df[df["Mass Axis"] < mass_max]
        df = df[df['Sum Spectrum'] > sum_spectrum_min]
        df = df[df['Sum Spectrum'] < sum_spectrum_max]
        return df



    def plot_sum_spectrum_vs_mass_axis(self, mass_min = 0.5, mass_max = 250, sum_spectrum_min = 0 , sum_spectrum_max = 1E10, title = "",figsize=(6.5, 6.5), cmap = "viridis"):
        """
        """
        #filter dataset, use lru_cache to speed up
        filtered_sum_mass = self.filter_sum_spectrum_vs_mass_axis(mass_min,mass_max,sum_spectrum_min,sum_spectrum_max)

        fig, ax = plt.subplots(figsize = figsize)
        sns.despine(fig, left=True, bottom=True)
        splot = sns.lineplot(x="Mass Axis", y="Sum Spectrum", sizes=(1, 8),
                             linewidth=1, data= filtered_sum_mass  , ax=ax).set_title(title)

        plt.close()
        return fig


    def convert_to_flat(self,four_D_array, mass_threshold ):
        """
        Convert 3D numpy array to 4 columns
        mass_threshold is a tuple (mass,threshold)
        """
        #create empty lists
        x = []
        y = []
        z = []
        v = []
        isotope_mass = []
        #print(mass)
        for mt in mass_threshold:
            print("mass",mt[0],"threshold",mt[1])
            for i in range(four_D_array.shape[0]):
                for j in range(four_D_array.shape[1]):
                    #print("J",j)
                    for k in range(four_D_array.shape[2]):
                        if (four_D_array[i,j,k,mt[0]] >= mt[1]):
                            #print(i,j,k,four_D_array[i,j,k,mass])
                            x.append(j)
                            y.append(k)
                            z.append(i)
                            v.append(four_D_array[i,j,k,mt[0] ] )
                            isotope_mass.append(mt[0])

        #convert to dataframe
        df = pd.DataFrame({'x': x, 'y': y,'z': z,'v':v,'mass':isotope_mass})
        print(df.shape[0],"points")
        print(df.groupby('mass').count())
        return df


    def three_D_plot_isotope(self, mass_threshold = ((27 , 0.9 )), figsize_x=15 , figsize_y=12 ,cmap = "viridis",size = 2,depthshade=True, opacity = 0.5):
        """
        Create a 3D plot using peakData.
        """
        #flaten array
        df = self.convert_to_flat( self.peak_data , mass_threshold)
        #plot
        self.three_D_plot(df,figsize_x , figsize_y ,cmap,size,depthshade, opacity=opacity)


    def three_D_plot(self,df, figsize_x , figsize_y, cmap = "viridis",size = 2 ,depthshade=True  ,opacity = 0.7):
        """
        Non-interactive plots using matplotlib, fast
        """
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = fig.add_subplot(111, projection="3d")
        #colormap = np.linspace(df['v'].min(),df['v'].max())
        ax.scatter(df['x'], df['y'], df['z'], c = df['mass'] , alpha=opacity, cmap = cmap, s = size, depthshade=depthshade  )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


    def plot_3D_scatter_plotly (self, mass_threshold = ((27 , 1.2 )) , size = 1 , opacity = 0.7,colorscale = 'Viridis',mode = 'markers'):
        """
        Interactive plot using plotly
        """
        #flaten array
        df = self.convert_to_flat(self.peak_data, mass_threshold)
        #plot
        fig = go.Figure(data = [go.Scatter3d(x = df['x'],
                                             y = df['y'],
                                             z= df['z'],
                                             mode = mode,
                                             marker = dict(
                                                 size=size ,
                                                 color = df['mass'],
                                                 colorscale = colorscale,
                                                 opacity = opacity))])
        fig.update_layout(margin =dict(l=0, r=0, b=0, t=0))
        fig.show()

    '''
    def plot_peak_data_3Dscatter(self,mass_threshold = ((27 , 1.2 )), size = 1 , opacity=0.5 , colorscale = 'Viridis',mode = 'markers'):
        """
        Interactive plot using plotly
        """
        self.plot_3D_scatter_plotly(self._peak_data, mass_threshold , size , opacity , colorscale , mode )
    '''


    @lru_cache(maxsize = cache_size)
    def sum_abundance(self,a,b):
        """
        used by plot_abundance
        """
        return np.sum(self._peak_data,axis = (TOF_SIMS.dim[a],TOF_SIMS.dim[b]))


    def plot_abundance(self, projection_axis = "z",mass = [1]):
        """
        Plots the sum of abundance per frame given an axis (x,y or z)
        axis : str
          x,y or z
        mass : list of ints
          masses to plot
        """

        mass = set(mass) #remove any duplicates in mass

        dims = "xyz".replace(projection_axis,"") #remve the axis to parse as we'll sum over the other two
        sum = self.sum_abundance(dims[0],dims[1])

        # Data for plotting
        x = np.arange(0.0, sum.shape[0], 1) #can be changed later to take into account real thickness/dimension

        fig, ax = plt.subplots()
        for m in mass:
            ax.plot(x,sum[:,m],label = m)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            #ax.legend()

        ax.set(xlabel= projection_axis, ylabel='sum over '+ dims,
              title='abundance over ' + projection_axis + '-axis')
        ax.grid()
        #fig.savefig("test.png")
        #plt.show()
        plt.close()
        return fig



    def format_axes(self, fig, text):
        """
        Function used for grid_proj_isotope
        """
        for i, ax in enumerate(fig.axes):

            ax.text(0.0 , 0.5, "%s" % text[i], va="bottom", ha="left")
            ax.tick_params(labelbottom=False, labelleft=False)


    def grid_proj_isotope(self, isotope=12, color = "gray", size = 12, cmap = "viridis"):
        """Creates gridspec with every max projections for a given isotope"""

        fig = plt.figure(constrained_layout=False,figsize=(size,size))

        gs = GridSpec(2, 2, figure=fig , left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        ax1 = fig.add_subplot(gs[:1, :1])
        ax1.imshow(self.max_proj_isotope(self.peak_data,"x",isotope),aspect='auto',cmap=cmap)


        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
        ax2 = fig.add_subplot(gs[0:1,1:])
        ax2.imshow(self.max_proj_isotope(self.peak_data,"y",isotope),aspect='auto',cmap=cmap)


        ax3 = fig.add_subplot(gs[1:,:1])
        ax3.imshow(self.max_proj_isotope(self.peak_data,"z",isotope),aspect='auto',cmap=cmap)


        fig.suptitle("Max projections for isotope " + str(isotope))
        self.format_axes(fig, ['xy projection','xz projection','yz projection'])

        #plt.show()
        #if save == True:
        #    fig.savefig("grid_proj_isotope_isotope " + str(isotope) , dpi = 100)

        #return figure to be able to save it using .savefig() method from matplotlib
        plt.close()
        return fig




    ##_________________________PROPERTIES

    #          file_name
    def _get_filename(self):
        """
        accessor for _file_name attribute
        """
        return self._file_name
    def _set_filename(self,string):
        """
        mutator for _file_name attribute
        """
        #assertion here
        try:
            self._file_name = string
        except AssertionError :
            print("{} is not valid name".format(string))
    file_name = property(_get_filename,_set_filename)

    #           BufTimes
    def _get_buf_times(self):
        """
        accessor for _BufTimes
        """
        return self._buf_times

    def _set_Buf_times(self,*arg,**kwarg):
        """
        mutator
        """
        print("buf_times is protected") #perhaps in latter versions
    buf_times = property(_get_buf_times,_set_Buf_times)

    #           TSP2
    def _get_TPS2(self):
        """
        accessor for _BufTimes
        """
        return self._TPS2

    def _set_TPS2(self,*arg,**kwarg):
        """
        mutator for _TPS2
        """
        print("TPS2 is protected") #perhaps in latter versions
    TPS2 = property(_get_TPS2,_set_TPS2)

    #           PeakData
    def _get_peak_data(self):
        """
        accessor for _peak_data
        """
        return self._peak_data

    def _set_peak_data(self,new_peak_data):
        """
        mutator for _peak_data
        """
        self._peak_data = new_peak_data
        #print("peak_data is protected") #perhaps in latter versions
    peak_data = property(_get_peak_data,_set_peak_data)


    #           peak_table
    def _get_peak_table(self):
        """
        accessor for _peak_table
        """
        return self._peak_table

    def _set_peak_table(self,*arg,**kwarg):
        """
        mutator for _peak_table
        """
        print("peak_table is protected") #perhaps in latter versions
    peak_table = property(_get_peak_table,_set_peak_table)

    #           sum_spectrum
    def _get_sum_spectrum(self):
        """
        accessor for _sum_spectrum
        """
        return self._sum_spectrum

    def _set_sum_spectrum(self,*arg,**kwarg):
        """
        mutator for _sum_spectrum
        """
        print("sum_spectrum is protected") #perhaps in latter versions
    sum_spectrum = property(_get_sum_spectrum,_set_sum_spectrum)

    #           event_list
    def _get_event_list(self):
        """
        accessor for _event_list
        """
        return self._event_list

    def _set_event_list(self,*arg,**kwarg):
        """
        mutator for _event_list
        """
        print("event_list is protected") #perhaps in latter versions
    event_list = property(_get_event_list,_set_event_list)

    #           mass_axis
    def _get_mass_axis(self):
        """
        accessor for _mass_axis
        """
        return self._mass_axis

    def _set_mass_axis(self,*arg,**kwarg):
        """
        mutator for _mass_axis
        """
        print("_mass_axis is protected") #perhaps in latter versions
    mass_axis = property(_get_mass_axis,_set_mass_axis)

    #           _sum_mass
    def _get_sum_mass(self):
        """
        accessor for _sum_mass
        """
        return self._sum_mass

    def _set_sum_mass(self,*arg,**kwarg):
        """
        mutator for _sum_mass
        """
        print("_sum_mass is protected") #perhaps in latter versions
    sum_mass = property(_get_sum_mass,_set_sum_mass)
