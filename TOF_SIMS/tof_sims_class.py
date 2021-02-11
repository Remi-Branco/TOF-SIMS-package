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
from sklearn.linear_model import LinearRegression

#remember the . before thread_classes !
from .multivariate_analysis import kPCA, incPCA, t_SNE
from .multivariate_analysis import PCA_pd as PCA
from .clustering import k_mean_voxels,filter_df_KMeans,plot_Clustered , k_mean_mini_batch_voxels,single_cluster_composition, silouhette


cache_size = 3 #value for lru_cache

class TOF_SIMS :
    """
    Class for manipulating tof-sim datasets

    filename : str
        filepath of TOF-SIM dataset
    """
    #to count the number of datasets opened
    datasets_opened = 0
    #to find correspondence between letters and axes
    dim = {"x":1,"y":2,"z":0,"n":3}


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
        #print("Cache size = ",cache_size)
        f = h5py.File(filename,'r')
        self._file_name = filename.split("/")[-1]
        print("Opening {}".format(self._file_name))
        self.file_path = filename[0:-len(filename.split("/")[-1])] #remove characters corresponding to filename
        self.fibimage = f['FIBImages']['Image0000']['Data'][()]
        self._buf_times = f['TimingData']['BufTimes']
        self._TPS2 = f['TPS2']["TwData"]

        #  The [()] means save it as a numpy array, not as reference in h5py
        self._peak_data = f['PeakData']["PeakData"][()]
        self._peak_data = np.flip(self._peak_data, axis = 0)  #flip x axis to have first slice as sample surface
        self._peak_table = f['PeakData']["PeakTable"]
        self._sum_spectrum = f['FullSpectra']['SumSpectrum'][()]
        print("Extraction of sum_spectrum done")

        #folowing operation removed until needed when peak reprocessing implemented
        #self._event_list = f['FullSpectra']['EventList'][()]
        #print("Extraction of event_list done")
        #print("Extraction of event_list done removed for debugging, remember to put it back")


        self._mass_axis = f['FullSpectra']['MassAxis'][()]
        print("Extraction of mass_axis done")
        self._sum_mass = pd.DataFrame({"Sum Spectrum":self._sum_spectrum,"Mass Axis":self._mass_axis})

        #increment dataset_opened each time a new dataset is opened (to limit memory usage at some point; not implemented)
        TOF_SIMS.datasets_opened +=1
        print("TOF-SIMS file imported successfully")


    def fit_detection(self):
        """
        Function for development,
        tests relationship between peakdata and mass spectrum graph.
        """
        #sum peak data over all dimensions to have total of detection per mass
        #self.pd_summed has dimension (250,)
        self.pd_summed = self.peak_data.sum(0).sum(0).sum(0)
        #get a copy of _sum_mass to simplify name
        sm = self._sum_mass
        #sum detected events between intervals [-0.5,0.5] for isotope 0, ...
        self.sum_mass_summed = [sm[(sm['Mass Axis']> i+0.5) & (sm['Mass Axis']<i+1.5)].sum()[0] for i in range(-1,249,1) ]
        # self.sum_mass_summed has shape (250,)
        self.comp_sum = pd.DataFrame({'detect graph': self.sum_mass_summed,'peak_data':self.pd_summed})

        X = np.array(self.comp_sum['detect graph'],dtype = 'float32').reshape(-1,1)
        y = np.array(self.comp_sum['peak_data']).reshape(-1,1)

        reg = LinearRegression().fit(X[1:], y[1:])
        print("R-square:",reg.score(X, y))
        print("y=ax+b","\n", "a: ",reg.coef_[0][0], "\n", "b: ",reg.intercept_[0])


    def plot_FIBImage(self, cmap = 'gray',figsize = (10,10)):
        """
        Method displaying FIBImages

        cmap : str
            color map (pyplot)
        figsize : tuple
            figure size (pyplot)
        """
        fig = plt.figure(figsize = figsize)
        plt.imshow(self.fibimage,cmap=cmap)
        plt.colorbar()
        #plt.show()
        #plt.close() is to prevent jupyter from displaying image when saving it
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
        cmap : str
            colormap (pyplot)
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
        cmap : str
            colormap (pyplot)
        """
        #assertions to be added here
        self.plot_section_of_3D_dataset(self._TPS2,section,cmap,mode = "mz")

    def plot_unique_section_4D_dataset(self, four_D_array, cmap ,x ,y ,z ,n , mode ):
        """
        Function displaying the PeakData dataset.
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
        plt.tight_layout()
        plt.show()
        #return fig object to allow user to save it using .savefig() method from matplotlib
        #curently not working
        return plt


    def projection(self, axes_displayed = "xy", axis_parsed = "z", axis_max_projection = "n", figsize = (22,22), cmap = "viridis" ):
        """
        Create a max projection with first two axes to be displayed as
        row/column in each subplot, axis to be parsed will be displayed in each
        subplot, the last axis will be the one summed (projected).
        axis summed.

        figsize : tuple
            figure size (pyplot)
        cmap : str
            color map (pyplot)
        axes_displayed :  str
          row/col, must be a combination of two of the following
          axes: x,y,z or n (e.g. 'xy')
        axis_parsed : str
            axis displayed in each subplot,
            must be one of the following axes: x,y,z or n (e.g. 'z')
        axis_max_projection : str
            axis summed in each subplot,
            must be one of the following axes: x,y,z or n (e.g. 'n')
        """
        axes_displayed = axes_displayed[::-1]
        print("Individual subplot represents " + axis_parsed + " axis with row = " + axes_displayed[0] + "-axis, column = " + axes_displayed[1] + "-axis projected over " + axis_max_projection + "-axis" )
        mode = self.trans_mode(axes_displayed + axis_parsed + axis_max_projection) #add axes in correct order and changes n to m
        self.max_proj(self._peak_data,mode,figsize,cmap = cmap)


    def trans_mode(self,mode):
        """
        Sub-function used by function "projection" to change m to n, should be removed later on
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
        plt.close()
        return fig



    def overlay_max_proj(self,  alpha = 0.5 , **isotopes):
      """
      Overlay max projection (z axis) for any number of isotopes
      """
      flag = True
      for isotope,list_value in isotopes.items():
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
        df = df[df["Mass Axis"] > mass_min]
        df = df[df["Mass Axis"] < mass_max]
        df = df[df['Sum Spectrum'] > sum_spectrum_min]
        df = df[df['Sum Spectrum'] < sum_spectrum_max]
        return df



    def mass_spec(self, interactive = True, mass_min = 0.5, mass_max = 250, sum_spectrum_min = 0 , sum_spectrum_max = 1E10, figsize=(10, 10)):
        """
        Plot mass spectrum (aundance vs m/z) representing the distribution of detected ions by mass.

        mass_min : int
            minimal mass to display
        mass_max : int
            maximal mass to display
        sum_spectrum_min : int
            minimal intensity/abundance to display
        sum_spectrum_max : int
            maximal intensity/abundance to display
        figsize : tuple
            figure size (pyplot) (if interactive == True)
        interactive : bool
            make the graph interactive
        """
        filtered_sum_mass = self.filter_sum_spectrum_vs_mass_axis(mass_min,mass_max,sum_spectrum_min,sum_spectrum_max)

        if interactive:
            fig = px.line(filtered_sum_mass, x="Mass Axis", y="Sum Spectrum", title='Mass spectrum', labels={"Mass Axis": "m/z","Sum Spectrum": "Intensity"})
            fig.show()

        else:
            fig, ax = plt.subplots(figsize = figsize)
            sns.despine(fig, left=True, bottom=True)
            splot = sns.lineplot(x="Mass Axis", y="Sum Spectrum", sizes=(1, 8),
                                 linewidth=1, data= filtered_sum_mass  , ax=ax).set_title(title)
            ax.set_xlabel('m/z')
            ax.set_ylabel('Intensity')
            ax.set_title('Mass spectrum')
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
        for mt in mass_threshold:
            print("mass",mt[0],"threshold",mt[1])
            for i in range(four_D_array.shape[0]):
                for j in range(four_D_array.shape[1]):
                    for k in range(four_D_array.shape[2]):
                        if (four_D_array[i,j,k,mt[0]] >= mt[1]):
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
        plt.close()
        return fig



    def format_axes(self, fig, text):
        """
        Function used for grid_proj_isotope
        """
        for i, ax in enumerate(fig.axes):

            ax.text(0.0 , 0.5, "%s" % text[i], va="bottom", ha="left")
            ax.tick_params(labelbottom=False, labelleft=False)



    def grid_proj_isotope(self, mass, figsize = (12,12) , cmap = "viridis"):
        """
        Creates gridspec with every max projections for a given isotope
        mass : int
            mass to display
        figsize : tuple , default=(12,12)
            figure size (pyplot)
        cmap : str , default='viridis'
            color map (pyplot)
        """
        #need to change max_proj_isotope name to max_proj_mass to avoid confusion in future releases
        fig = plt.figure(constrained_layout=False, figsize = figsize)
        gs = GridSpec(2, 2, figure=fig , left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        ax1 = fig.add_subplot(gs[:1, :1])
        ax1.imshow(self.max_proj_isotope(self.peak_data,"x",isotope = mass ),aspect='auto',cmap=cmap)
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
        ax2 = fig.add_subplot(gs[0:1,1:])
        ax2.imshow(self.max_proj_isotope(self.peak_data,"y",mass),aspect='auto',cmap=cmap)
        ax3 = fig.add_subplot(gs[1:,:1])
        ax3.imshow(self.max_proj_isotope(self.peak_data,"z",mass),aspect='auto',cmap=cmap)
        fig.suptitle("Max projections for mass " + str(mass))
        self.format_axes(fig, ['front projection','side projection','top projection'])
        #return figure to be able to save it using .savefig() method from matplotlib
        plt.close()
        return fig



    def PCA_masses(self , x_min = 0 , x_max = 1 , y_min = 0 , y_max = 1, z_min = 0, z_max = 1, principal_components = 3 , mass_start = 1 , mass_stop = 249 ):
        """
        perform PCA (Principal Component Analysis) on masses, voxel-wise.
        linear dimensionality reduction

        x_min : int
            minimal x value
        y_min : int
            minimal y value
        z_min : int
            minimal z value
        x_max : int
            maximal x value
        y_max : int
            maximal y value
        z_max : int
            maximal z value
        mass_start : int
            minimal mass
        mass_stop : int
            maximal mass
        principal_components : int
            number of principal components for PCA
        """
        #comment adapted from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA
        self.principalDF, self.df_PCA, self.voxels = PCA(self.peak_data, x_min,x_max , y_min,y_max,z_min, z_max, principal_components , mass_start , mass_stop, with_std = False , with_mean = True )



    def kPCA_masses(self ,kernel = 'linear', gamma = None , x_min = 0 , x_max = 10 , y_min = 0 , y_max = 10, z_min = 0, z_max = 10, principal_components = 3 , mass_start = 1 , mass_stop = 249 ):
        """
        perform kPCA (kernel Principal Component Analysis) on masses, voxel-wise.
        Non-linear dimensionality reduction through the use of kernels

        x_min : int
            minimal x value
        y_min : int
            minimal y value
        z_min : int
            minimal z value
        x_max : int
            maximal x value
        y_max : int
            maximal y value
        z_max : int
            maximal z value
        mass_start : int
            minimal mass
        mass_stop : int
            maximal mass
        principal_components : int
            number of principal components for PCA
        kernel : str
            {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
            Kernel used for PCA.
        gamma : float
            default=None
            Kernel coefficient for rbf, poly and sigmoid kernels.
            Ignored by other kernels. If gamma is None, then it is set to 1/n_features.
        """
        #comment adapted from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html?highlight=kpca
        self.principalDF, self.df_PCA, self.voxels = kPCA(self.peak_data,kernel,gamma, x_min,x_max , y_min,y_max,z_min, z_max, principal_components , mass_start , mass_stop)



    def incPCA_masses(self, n_batches = None , x_min = 0 , x_max =  10 , y_min = 0 , y_max =10 , z_min = 0 , z_max = 10 , principal_components = 2 , mass_start = 1 , mass_stop = 250 ):
        """
        perform iPCA (Incremental Principal Component Analysis) on masses, voxel-wise.
        Linear dimensionality reduction using Singular Value Decomposition of
        the data, keeping only the most significant singular vectors to project
        the data to a lower dimensional space. The input data is centered but not
        scaled for each feature before applying the SVD.

        x_min : int
            minimal x value
        y_min : int
            minimal y value
        z_min : int
            minimal z value
        x_max : int
            maximal x value
        y_max : int
            maximal y value
        z_max : int
            maximal z value
        mass_start : int
            minimal mass
        mass_stop : int
            maximal mass
        principal_components : int
            number of principal components for PCA
        n_batches : int
            The number of samples to use for each batch. Only used when calling fit.
            If batch_size is None, then batch_size is inferred from the data and
            set to 5 * n_features, to provide a balance between approximation
             accuracy and memory consumption.
        """
        #comment adapted from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html?highlight=ipca
        self.principalDF, self.df_PCA, self.voxels = incPCA(self.peak_data , n_batches , x_min , x_max , y_min , y_max , z_min , z_max , principal_components , mass_start , mass_stop , with_std = False , with_mean = True)



    def TSNE_masses(self, n_components = 2, perplexity = 30.0 ,n_iter = 1000 ,early_exaggeration = 12.0, n_iter_without_progress = 300,min_grad_norm = 1e-7,metric = 'euclidean',init = 'random',verbose = 0,random_state = None,method = 'barnes_hut', x_min = 0 , x_max = 10 , y_min = 0 , y_max = 10 , z_min = 0 , z_max = 10  , mass_start = 1 , mass_stop = 250 , learning_rate = 200.0):
        """
        perform incremental t-SNE (t-distributed Stochastic Neighbor Embedding)
        on masses, voxel-wise.

        t-SNE is a tool to visualize high-dimensional data.
        It converts similarities between data points to joint probabilities and
        tries to minimize the Kullback-Leibler divergence between the joint
        probabilities of the low-dimensional embedding and the high-dimensional
        data. t-SNE has a cost function that is not convex, i.e. with different
        initializations we can get different results.

        n_components : int, default=2
            Dimension of the embedded space.
        perplexity : float, default=30.0
            The perplexity is related to the number of nearest neighbors that is
            used in other manifold learning algorithms. Larger datasets usually
            require a larger perplexity. Consider selecting a value between 5
            and 50. Different values can result in significantly different results.
        n_iter : int, default=1000
            Maximum number of iterations for the optimization. Should be at least 250.
        learning_rate : float, default=200.0
            The learning rate for t-SNE is usually in the range [10.0, 1000.0].
             If the learning rate is too high, the data may look like a ‘ball’
             with any point approximately equidistant from its nearest neighbours.
             If the learning rate is too low, most points may look compressed in
             a dense cloud with few outliers. If the cost function gets stuck in
             a bad local minimum increasing the learning rate may help.
        early_exaggeration : float, default = 12.0
            Controls how tight natural clusters in the original space are in the
            embedded space and how much space will be between them. For larger
            values, the space between natural clusters will be larger in the
            embedded space. Again, the choice of this parameter is not very
            critical. If the cost function increases during initial optimization,
            the early exaggeration factor or the learning rate might be too high.
        n_iter_without_progress : int, default=300
            Maximum number of iterations without progress before we abort the
            optimization, used after 250 initial iterations with early exaggeration.
            Note that progress is only checked every 50 iterations so this value is
            rounded to the next multiple of 50.
        min_grad_norm : float, default=1e-7
            If the gradient norm is below this threshold, the optimization
            will be stopped.
        metric : str or callable, default=’euclidean’
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scipy.spatial.distance.pdist for its metric parameter, or
            a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is
            “precomputed”, X is assumed to be a distance matrix. Alternatively,
            if metric is a callable function, it is called on each pair of instances
            (rows) and the resulting value recorded. The callable should take
            two arrays from X as input and return a value indicating the distance
            between them. The default is “euclidean” which is interpreted as squared
            euclidean distance.
        init : {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default=’random’
            Initialization of embedding. Possible options are ‘random’, ‘pca’,
            and a numpy array of shape (n_samples, n_components). PCA initialization
            cannot be used with precomputed distances and is usually more globally
            stable than random initialization.
        verbose : int, default=0
            Verbosity level
        random_state : int, RandomState instance or None, default=None
            Determines the random number generator. Pass an int for reproducible
            results across multiple function calls. Note that different
            initializations might result in different local minima of the cost
            function.
        method : str, default=’barnes_hut’
            By default the gradient calculation algorithm uses Barnes-Hut
            approximation running in O(NlogN) time. method=’exact’ will run on
            the slower, but exact, algorithm in O(N^2) time. The exact algorithm
            should be used when nearest-neighbor errors need to be better than 3%.
            However, the exact method cannot scale to millions of examples.

        x_min : int
            minimal x value
        y_min : int
            minimal y value
        z_min : int
            minimal z value
        x_max : int
            maximal x value
        y_max : int
            maximal y value
        z_max : int
            maximal z value
        mass_start : int
            minimal mass
        mass_stop : int
            maximal mass
        """
        #comment adapted from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html?highlight=tsne#sklearn.manifold.TSNE
        self.principalDF, self.df_PCA, self.voxels = t_SNE(self.peak_data , n_components, perplexity,n_iter,early_exaggeration,n_iter_without_progress,min_grad_norm ,metric ,init ,verbose ,random_state ,method , x_min , x_max , y_min , y_max , z_min , z_max  , mass_start , mass_stop , learning_rate)


    def principal_components(self):
        """
        Display how much variation each principal component explains from data.
        """
        x = self.df_PCA['n']
        y = self.df_PCA['explained variance']
        y_cum = self.df_PCA['explained variance'].cumsum()
        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y,
                            mode='lines+markers',
                            name='explained variance'))
        fig.add_trace(go.Scatter(x=x, y=y_cum,
                            mode='lines+markers',
                            name='cummulated variance'))
        fig.show()



    def  PCA_2D(self,component_x = 1 , component_y = 2):
        """
        2D PCA plot
        component_x : int or str(int)
            principal component to plot as x-axis
        component_y : int or str(int)
            principal component to plot as y-axis
        """
        component_x = 'principal component ' + str(component_x)
        component_y = 'principal component ' + str(component_y)
        fig = px.scatter(x=self.principalDF[component_x] , y=self.principalDF[component_y] , color = self.principalDF['masses'], labels={"x": component_x, "y": component_y, "color": "mass"} )
        fig.show()



    def PCA_3D(self,component_x = 1 , component_y = 2, component_z = 3):
        """
        3D PCA plot
        component_x : int or str(int)
            principal component to plot as x-axis
        component_y : int or str(int)
            principal component to plot as y-axis
        component_z : int or str(int)
            principal component to plot as z-axis
        """
        component_x = 'principal component ' + str(component_x)
        component_y = 'principal component ' + str(component_y)
        component_z = 'principal component ' + str(component_z)
        fig = px.scatter_3d(self.principalDF , x=component_x , y=component_y , z=component_z , color='masses')
        fig.show()



    def KMeans(self, k = 2 , max_iter = 300 , x_min = 0 , x_max = 25 , y_min = 0 , y_max = 50 , z_min = 0 , z_max = 100 , mass_start = 1 , mass_stop = 250):
        """
        KMeans clustering on voxels.

        k : int , default=2
            The number of clusters to form as well as the number of centroids to
            generate.
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        x_min : int
            minimal x value
        y_min : int
            minimal y value
        z_min : int
            minimal z value
        x_max : int
            maximal x value
        y_max : int
            maximal y value
        z_max : int
            maximal z value
        mass_start : int
            minimal mass
        mass_stop : int
            maximal mass
        """
        # comment adapted from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans
        self.df_KMeans = k_mean_voxels(self.peak_data , k , max_iter  , x_min , x_max , y_min , y_max , z_min , z_max , mass_start , mass_stop)


    def KMeans_mini_batch(self,random_state = None,n_init = 10, batch_size = 100, k = 2 , max_iter = 300 , x_min = 0 , x_max = 25 , y_min = 0 , y_max = 50 , z_min = 0 , z_max = 100 , mass_start = 1 , mass_stop = 250):
        """
        Mini batch KMeans clustering on voxels

        k : int , default=2
            The number of clusters to form as well as the number of centroids to
            generate.
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        batch_size : int, default=100
            Size of the mini batches.
        random_state : int, RandomState instance or None, default=None
            Determines random number generation for centroid initialization and
            random reassignment. Use an int to make the randomness deterministic.
        x_min : int
            minimal x value
        y_min : int
            minimal y value
        z_min : int
            minimal z value
        x_max : int
            maximal x value
        y_max : int
            maximal y value
        z_max : int
            maximal z value
        mass_start : int
            minimal mass
        mass_stop : int
            maximal mass
        """
        #comment adapted from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
        self.df_KMeans , self.df_abundance_per_cluster = k_mean_mini_batch_voxels(self.peak_data , random_state ,n_init , batch_size, k , max_iter  , x_min , x_max , y_min , y_max , z_min , z_max , mass_start , mass_stop)


    def plot_cluster(self , clusters_to_keep = [0,1],mode = 'markers',size=3,colorscale = 'Rainbow',opacity = 0.7 ):
        #first filter dataframe to only keep relevant clusters
        self.filtered_KMeans = filter_df_KMeans(self.df_KMeans , clusters_to_keep )
        #3D plot voxels labelled by kmeans
        plot_Clustered(self.filtered_KMeans,mode ,size,colorscale ,opacity )


    def plot_cluster_composition(self,mass_min = 0, mass_max = 250,barmode='stack', cluster = 0 ):
        """
        """
        #stack label column
        df = self.df_abundance_per_cluster.stack(level = 'label')
        df = df.reset_index()
        df = df.rename(columns={"level_0": "mass" , 0: 'abundance','label' : "cluster"})
        df.head()
        fig = px.bar(df, color="cluster", y="abundance", x="mass", barmode="stack", color_discrete_map=True)
        fig.show()


    def plot_cluster_composition_multigraph(self,mass_min = 0, mass_max = 250,barmode='stack', cluster = 0 ):
        """
        Not yet working
        """
        #stack label column
        df = self.df_abundance_per_cluster.stack(level = 'label')
        df = df.reset_index()
        df = df.rename(columns={"level_0": "mass" , 0: 'abundance','label' : "cluster"})
        df.head()
        fig = px.scatter(df, x="mass", y="abundance", facet_col="cluster")
        fig.update_yaxes(nticks=20)
        fig.show()


    def plot_cluster_pie_chart(self,n_mass = 5 , cluster = 0):
        """
        Displaying n-most abundant masses for a given cluster
        n_mass : int , default=5
            top n masses per cluster
        cluster : int, default=0
            cluster to analyse
        """
        single_cluster_composition(self.df_abundance_per_cluster , n_mass , cluster )



    def silouhette_KMeans(self,random_state = 1,n_init = 10, batch_size = 100,  max_iter = 300 , x_min = 0 , x_max = 25 , y_min = 0 , y_max = 50 , z_min = 0 , z_max = 100 , mass_start = 1 , mass_stop = 250, range_clusters = [2,3,4], plot = False, minibatch = True):
        self.silouhette_score = silouhette(self.peak_data , random_state ,n_init , batch_size , max_iter , x_min , x_max , y_min , y_max , z_min , z_max , mass_start , mass_stop, range_clusters, plot, minibatch)




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
