import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib as plt
import plotly.express as px

class PCA_p():
    """
    PCA on peak_data
    """

    def __init__(self, array, x_max = 1 , y_max = 1, z_max = 1, principal_components=3, mass_start =1 , mass_stop=250):
        self.array = array
        self.dff = pd.DataFrame(columns = ["explained variance",'x','n'])

        self.mass_start = mass_start
        self.mass_stop = mass_stop
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.principal_components = principal_components

        #preform PCA at construction
        self.PCA_data(x_max , y_max, z_max, principal_components, mass_start , mass_stop)


        #def PCA_data(self,x_max , y_max, z_max, principal_components=3, mass_start =1 , mass_stop=250 ):
    def PCA_data(self, x_max , y_max , z_max , principal_components , mass_start , mass_stop ):

        #PCA on peak_data
        #re-copy
        self.mass_start = mass_start
        self.mass_stop = mass_stop
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.principal_components = principal_components


        #get a slice of the data
        voxels = self.array[:self.x_max , :self.y_max , : self.z_max , self.mass_start : self.mass_stop]
        dim = voxels.shape[3]

        #flatten (put all voxels in a line vector)
        voxels = voxels.flatten()

        #since we flattened, every 250 is a voxel (e.g. 0:249 is voxel #1 then 250 to 499 voxel#2,...
        #therefore reshape in square matrix of size [25 x 25 x 25 , 250 ]
        voxels = voxels.reshape(-1 ,dim)
        #we know that first channel (mass = 0 ) is just noise we removed it with [:,1:]


        #then transpose the data to have one element per row
        voxels = voxels.T

        # Standardise the features before PCA
        x = StandardScaler().fit_transform(voxels)
        #perform PCA on data
        pca = PCA(n_components=self.principal_components)
        principalComponents = pca.fit(x)
        PCA(n_components=self.principal_components)
        self.dff['explained variance'] = pca.explained_variance_ratio_
        self.dff['x'] = pca.singular_values_
        self.dff['n'] = [i for i in range(self.dff.shape[0])]

        #generate the features (here )
        features = [i for i in range(self.mass_start,self.mass_stop,1)]
        self.principalComponents = pca.fit_transform(x)
        self.principalDf = pd.DataFrame(data = self.principalComponents
                     , columns = ['principal component 1', 'principal component 2','principal component 3'])
        self.principalDf["masses"] = features


    def PCA_show_pc(self):
        """
        Function ploting explained variance
        """
        sns.lineplot(data = self.dff, x = 'n', y = 'explained variance')


    def PCA_2D(self,component_x = 1,component_y = 2):
        """
        component_x : int or str(of int)
            represents which principal component to plot
        """

        component_x = 'principal component ' + str(component_x)
        component_y = 'principal component ' + str(component_y)
        fig = px.scatter(x=self.principalDf[component_x], y=self.principalDf[component_y],color = self.principalDf['masses'])
        fig.show()


    def label_point(self, df , ax, pc_a , pc_b ,labels ):
        #print(df.shape[0])
        for i in range(df.shape[0]):
            ax.text(df.iloc[i,pc_a]+.02, df.iloc[i,pc_b], df.iloc[i,labels],fontsize=20)


    def PCA_3d(self):
        """
        Show PC1,2 and 3 in 3D interactive plot
        """

        fig = px.scatter_3d(self.principalDf, x='principal component 1', y='principal component 2', z='principal component 3',
                      color='masses')
        fig.show()
