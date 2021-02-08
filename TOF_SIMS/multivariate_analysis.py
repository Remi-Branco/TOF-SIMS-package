import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib as plt
import plotly.express as px
from sklearn.decomposition import KernelPCA, PCA, IncrementalPCA
from sklearn.manifold import TSNE



def PCA_pd(array, x_min,x_max , y_min,y_max,z_min, z_max, principal_components , mass_start , mass_stop,with_std, with_mean ):
    """
    PCA on peak_data
    """

    df = pd.DataFrame(columns = ["explained variance",'x','n'])
    voxels, initial_shape = data_transform(array , x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop)
    #print(self.voxels.shape)

    # Standardise the features before PCA
    scaler = StandardScaler(with_std = with_std,with_mean = with_mean)
    x = scaler.fit_transform(voxels[1:,:])

    x = x.T #necessary to have element per row and voxels per column
    #print(8 , "\n" , x.shape )
    #perform PCA on data
    pca = PCA(n_components=principal_components)
    principalComponents = pca.fit(x)
    #PCA(n_components=principal_components)
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

    principalDf["masses"] = voxels[0,:]

    # keep but not usefull yet   principalDf["masses"] = [i for i in range(principalDf.shape[0])]

    print("done")
    return (principalDf , df, voxels)


def data_transform(array,x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop):
    """
    transform 4D matrix into 2D matrix with:
        each row representing a voxel
        each columns representing an isotope/mass
    the returned data is NOT standardised
    """
    #get a slice of the data
    #we know that first channel (mass = 0 ) is just noise so remove it by adjusting mass_start = 1
    voxels = array[z_min:z_max , y_min:y_max , x_min: x_max , mass_start : mass_stop]

    dim = voxels.shape[3] #step needs to be done here
    initial_shape = voxels.shape #for reshaping labels later on
    #transform 4D dataset into long list of masses with one voxel per row and one mass per column
    voxels = voxels.reshape(-1,dim)
    print()
    #append masses to top of array
    m = np.array(([i for i in range(mass_start , mass_stop , 1 )],)) #this is a one row array
    #print(m.shape,"\n",m,"\n")
    #concatenate it to top of voxels array (on top of it)
    #concatenate masses array (1,2,3,4..) as first column and
    voxels = np.concatenate((m,voxels),axis =0)
    return voxels, initial_shape



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

def kPCA(array , kernel , gamma , x_min , x_max , y_min , y_max , z_min , z_max , principal_components , mass_start , mass_stop ):
    """
    kernel PCA
    """
    df = pd.DataFrame(columns = ["explained variance",'x','n'])
    #transform data, self.voxel is created in the function
    voxels, initial_shape = data_transform(array , x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop)

    scaler = StandardScaler(with_std = False)
    x = scaler.fit_transform(voxels[1:,:])

    x = x.T #important to have one element per row
    kPCA = KernelPCA(n_components=principal_components, kernel=kernel, gamma = gamma)
    X_transformed = kPCA.fit_transform(x)
    '''
    df['explained variance'] = kPCA.explained_variance_ratio_
    df['x'] = kPCA.singular_values_
    df['n'] = [i for i in range(1,df.shape[0]+1,1)]
    '''

    #print(1,X_transformed.shape)
    lbl = ["principal component " + str(i) for i in range(principal_components)]
    principalDf = pd.DataFrame(data = X_transformed, columns = lbl)
    principalDf["masses"] = voxels[0,:]
    return (principalDf, df, voxels)



def incPCA(array ,n_batches, x_min , x_max , y_min , y_max , z_min , z_max , principal_components , mass_start , mass_stop ,with_std, with_mean):
    """
    Incremental PCA
    """
    df = pd.DataFrame(columns = ["explained variance",'x','n'])
    #transform data, self.voxel is created in the function
    voxels, initial_shape = data_transform(array , x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop)

    # Scale but NOT standardise features before PCA
    scaler = StandardScaler(with_std = with_std,with_mean = with_mean)
    x = scaler.fit_transform(voxels[1:,:])


    x = x.T #important to have one element per row

    incPCA = IncrementalPCA(n_components=principal_components)

    for X_batch in np.array_split(x , n_batches):
        incPCA.partial_fit(X_batch)
    X_transformed = incPCA.transform(x)

    df['explained variance'] = incPCA.explained_variance_ratio_
    df['x'] = incPCA.singular_values_
    df['n'] = [i for i in range(1,df.shape[0]+1,1)]

    #print(1,X_transformed.shape)
    lbl = ["principal component " + str(i) for i in range(principal_components)]
    principalDf = pd.DataFrame(data = X_transformed, columns = lbl)
    principalDf["masses"] = voxels[0,:]
    return (principalDf, df, voxels)

def t_SNE(array ,n_components, perplexity,n_iter, x_min , x_max , y_min , y_max , z_min , z_max  , mass_start , mass_stop , learning_rate):
    """
    T-SNE
    """
    df = pd.DataFrame(columns = ["explained variance",'x','n'])
    #transform data, self.voxel is created in the function
    voxels, initial_shape = data_transform(array , x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop)

    # Standardise the features before PCA
    x = StandardScaler().fit_transform(voxels[1:,:]) # do not scale first row as it contains masses

    x = x.T #important to have one element per row

    #time_start = time.time()
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=perplexity, n_iter=n_iter,learning_rate = learning_rate)

    tsne_results = tsne.fit_transform(x)

    principalDf = pd.DataFrame({"principal component 1": tsne_results[:,0],
                          "principal component 2": tsne_results[:,1]})

    #lbl = ["principal component " + str(i) for i in range(principal_components)]
    #principalDf = pd.DataFrame(data = X_transformed, columns = lbl)
    principalDf["masses"] = voxels[0,:]
    #principalDf["masses"] = features
    return (principalDf, df, voxels)
