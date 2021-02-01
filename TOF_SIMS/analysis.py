import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib as plt


def PCA_data(array , mass_start = 1, mass_stop = 250, x_max =30, y_max=30, z_max =30):
    #get a slice of the data

    voxels = array[:x_max,:y_max,:z_max,mass_start:mass_stop]
    dim = voxels.shape[3]
    #print(voxels.shape)

    #flatten (put all voxels in a line vector)
    voxels = voxels.flatten()
    #print("after flatten",voxels.shape)

    #since we flattened, every 250 is a voxel (e.g. 0:249 is voxel #1 then 250 to 499 voxel#2,...
    #therefore reshape in square matrix of size [25 x 25 x 25 , 250 ]
    voxels = voxels.reshape(-1 ,dim)
    #we know that first channel (mass = 0 ) is just noise we removed it with [:,1:]


    #then transpose the data to have one element per row
    voxels = voxels.T
    print(voxels.shape)
    # Standardise the features before PCA
    x = StandardScaler().fit_transform(voxels)
    #perform PCA on data
    pca = PCA(n_components=10)
    principalComponents = pca.fit(x)
    PCA(n_components=10)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    dff = pd.DataFrame({ 'explained variance':pca.explained_variance_ratio_, 'x':pca.singular_values_})
    dff['n'] = [i for i in range(dff.shape[0])]
    print(dff.head())
    sns.lineplot(data = dff, x = 'n', y = 'explained variance')

    #generate the features (here )
    features = [i for i in range(mass_start,mass_stop,1)]
    print(len(features))
    print(features[:5])

    pca = PCA(n_components=3)

    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2','principal component 3'])



    principalDf["masses"] = features

    import matplotlib.pyplot as plt


    ax = sns.lmplot('principal component 1', # Horizontal axis
               'principal component 2', # Vertical axis
               data=principalDf, # Data source
               fit_reg=False, # Don't fix a regression line
               height = 10,
               aspect =2 ) # size and dimension

    plt.title('Principal components')
    # Set x-axis label
    plt.xlabel('principal component 1')
    # Set y-axis label
    plt.ylabel('principal component 2')

    def label_point(df , ax):
        print(df.shape[0])
        for i in range(df.shape[0]):
            ax.text(df.iloc[i,0]+.02, df.iloc[i,1], df.iloc[i,3],fontsize=20)

    label_point(principalDf , plt.gca())

    ax = sns.lmplot('principal component 2', # Horizontal axis
               'principal component 3', # Vertical axis
               data=principalDf, # Data source
               fit_reg=False, # Don't fix a regression line
               height = 10,
               aspect =2 ) # size and dimension

    plt.title('Principal components')
    # Set x-axis label
    plt.xlabel('principal component 2')
    # Set y-axis label
    plt.ylabel('principal component 3')

    def label_point(df , ax):
        print(df.shape[0])
        for i in range(df.shape[0]):

            ax.text(df.iloc[i,1]+.02, df.iloc[i,2], df.iloc[i,3],fontsize=20)

    label_point(principalDf , plt.gca())

    import plotly.express as px

    fig = px.scatter_3d(principalDf, x='principal component 1', y='principal component 2', z='principal component 3',
                  color='masses')
    fig.show()
