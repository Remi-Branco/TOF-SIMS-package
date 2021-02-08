from sklearn.cluster import KMeans,MiniBatchKMeans

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import seaborn as sns
import numpy as np


#remember the . before thread_classes !
#not working curently
from .multivariate_analysis import data_transform



def k_mean_voxels(array , k ,max_iter , x_min , x_max , y_min , y_max , z_min , z_max , mass_start , mass_stop):
    """
    Run KMeans clustering to cluster voxels in groups with similar isotope abundance

    """
    voxels, initial_shape = data_transform(array,x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop)
    #copy vovels without first row which contains masses
    X = voxels[1:,:]
    #here the dataset will be segmented for each voxel, not masses
    print(X.shape)
    #scale and standardise
    scaler = StandardScaler(with_mean = True, with_std = True) #must be standardised, not just scaled
    X_transformed = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters = k, max_iter = max_iter)

    #y_pred contains labels for each voxel
    y_pred = kmeans.fit_predict(X_transformed)
    #reshape labels into 4D dataset with 4th dim as label
    peak_data_lbld = y_pred.reshape(initial_shape[0],initial_shape[1],initial_shape[2],1)
    df_labelled = transform_data(peak_data_lbld)
    return df_labelled



def k_mean_mini_batch_voxels(array , random_state ,n_init , batch_size , k ,max_iter , x_min , x_max , y_min , y_max , z_min , z_max , mass_start , mass_stop):
    """
    Run KMeans clustering to cluster voxels in groups with similar isotope abundance

    """
    voxels, initial_shape = data_transform(array,x_min,x_max , y_min,y_max,z_min, z_max , mass_start , mass_stop)
    #copy vovels without first row which contains masses
    X = voxels[1:,:]
    #here the dataset will be segmented for each voxel, not masses
    #print(X.shape)
    #scale and standardise
    scaler = StandardScaler(with_mean = True, with_std = True) #must be standardised, not just scaled
    X_transformed = scaler.fit_transform(X)
    kmeans = MiniBatchKMeans(n_clusters = k, max_iter = max_iter, batch_size = batch_size,random_state=random_state,n_init = n_init )

    #y_pred contains labels for each voxel
    y_pred = kmeans.fit_predict(X_transformed)
    #reshape labels into 4D dataset with 4th dim as label
    peak_data_lbld = y_pred.reshape(initial_shape[0],initial_shape[1],initial_shape[2],1)
    df_labelled = transform_data(peak_data_lbld)
    df_abundance_per_cluster = abundance_per_cluster(X,y_pred, mass_start, mass_stop )
    return df_labelled, df_abundance_per_cluster


def filter_df_KMeans(df,clusters_to_keep):
    #filter using a query, this query allows multiple groups to be filtered dynamically
    query =  ' | '.join(["label=={}".format(k) for k in clusters_to_keep])
    #print(query)
    print("Cluster(s)", ", ".join(["{}".format(k) for k in clusters_to_keep]), "kept")
    #filter df1 based on query
    df = df.query(query)
    return df


def plot_Clustered(df, mode, size,colorscale,opacity):
    fig = go.Figure(data = [go.Scatter3d(x = df['x'],                                                     y = df['y'],
                                                 z= df['z'],
                                                 mode = mode,
                                                 marker = dict(
                                                     size = size ,
                                                     color = df['label'],
                                                     colorscale = colorscale,
                                                     opacity = opacity))])

    fig.update_layout(margin =dict(l=0, r=0, b=0, t=0))
    fig.show()


def abundance_per_cluster(X, y_pred,mass_start, mass_stop):

    column = [str(m) for m in range(mass_start, mass_stop,1)]
    #print(column)
    df2 = pd.DataFrame(data = X,columns = column )
    df2.head()
    #append the labels
    df2['label'] = y_pred
    #df2
    #compute average abundance of each isotope per group
    e = df2.groupby('label').mean()
    #e
    #transpose the dataframe to have masses as columns
    f = e.T

    return f





def transform_data(array):
    x = []
    y = []
    z = []
    l = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            #print("J",j)
            for k in range(array.shape[2]):
                #print(i,j,k,four_D_array[i,j,k,mass])
                x.append(k)
                y.append(j)
                z.append(i)
                l.append(array[i,j,k,0] )
    df = pd.DataFrame({'x': x, 'y': y,'z': z,'label':l})
    print("There are ")
    print(df.shape[0]," voxels")
    print(df.groupby('label').count())
    return df
