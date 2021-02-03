import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib as plt
import plotly.express as px
#from sklearn.decomposition import KernelPCA





def PCA_pd(array, x_min,x_max , y_min,y_max,z_min, z_max, principal_components , mass_start , mass_stop ):
    """
    PCA on peak_data
    """

    df = pd.DataFrame(columns = ["explained variance",'x','n'])
    #transform data, self.voxel is created in the function
    voxels = data_transform(array , x_min,x_max , y_min,y_max,z_min, z_max, principal_components , mass_start , mass_stop)
    #print(self.voxels.shape)

    per_mass = True
    # Standardise the features before PCA
    x = StandardScaler().fit_transform(voxels[1:,:]) # do not scale first row as it contains masses
    if per_mass :
        x = x.T #necessary to have each element per row
    #print(8 , "\n" , x.shape )
    #perform PCA on data
    pca = PCA(n_components=principal_components)
    principalComponents = pca.fit(x)
    PCA(n_components=principal_components)
    df['explained variance'] = pca.explained_variance_ratio_
    df['x'] = pca.singular_values_
    df['n'] = [i for i in range(1,df.shape[0]+1,1)]
    #generate labels for principal components
    lbl = ["principal component " + str(i) for i in range(principal_components)]
    #print("8.5","\n", lbl)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = lbl)

    #print(9 , "\n" ,self.principalDf.head())
    #print(10 , "\n" ,self.principalDf.shape)
    #not needed anymore -> generate the features
    #features = [i for i in range(self.mass_start,self.mass_stop,1)]
    #self.principalDf["masses"] = features
    if per_mass:
        principalDf["masses"] = voxels[0,:]
    else:
        principalDf["masses"] = [i for i in range(principalDf.shape[0])]
    #print(11 , "\n" ,self.principalDf.head())
    print("done")
    return (principalDf , df)


def data_transform(array,x_min,x_max , y_min,y_max,z_min, z_max , principal_components , mass_start , mass_stop):
    """
    transform 4D matrix into 2D matrix with:
        each row representing a voxel
        each columns representing an isotope/mass
    the returned data is NOT standardised
    """
    #get a slice of the data
    #we know that first channel (mass = 0 ) is just noise so remove it by adjusting mass_start = 1
    voxels = array[x_min:x_max , y_min:y_max , z_min: z_max , mass_start : mass_stop]
    dim = voxels.shape[3] #step needs to be done here
    #print(0,self.voxels.shape[3])
    #print(1,self.voxels.shape)

    #flatten (put all voxels in a line vector)
    voxels = voxels.flatten()
    #print(2,self.voxels.shape)

    #since we flattened, every 250 is a voxel (e.g. 0:249 is voxel #1 then 250 to 499 voxel#2,...
    #therefore reshape in square matrix of size [25 x 25 x 25 , 250 ]
    voxels = voxels.reshape(-1 ,dim)

    #print(3,self.voxels.shape)
    #append masses to top of array
    m = np.array(([i for i in range(mass_start , mass_stop , 1 )],)) #this is a one row array
    #print(m.shape,"\n",m,"\n")
    #concatenate it to top of voxels array (on top of it)
    #concatenate masses array (1,2,3,4..) as first column and
    voxels = np.concatenate((m,voxels),axis =0)
    return voxels



def label_point(df , ax, pc_a , pc_b ,labels ):
    #print(df.shape[0])
    for i in range(df.shape[0]):
        ax.text(df.iloc[i,pc_a]+.02, df.iloc[i,pc_b], df.iloc[i,labels],fontsize=20)


def PCA_3D(principalDf):
    """
    Show PC1,2 and 3 in 3D interactive plot
    """
    fig = px.scatter_3d(principalDf, x='principal component 1', y='principal component 2', z='principal component 3',
                  color='masses')
    fig.show()
