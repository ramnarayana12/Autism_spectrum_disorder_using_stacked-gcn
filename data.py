import os
import csv
import numpy as np
import scipy.io as sio

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from nilearn.connectome import ConnectivityMeasure

from scipy.spatial import distance


# Reading and computing the input data

# Selected pipeline
pipeline = 'cpac'

# Input data variables
root_folder = 'c:\\Users\\HP\\Downloads\\abide\\ABIDE'
data_folder = os.path.join(root_folder, 'Outputs','cpac','filt_noglobal','rois_ho')
phenotype = os.path.join(root_folder, 'Phenotypic_V1_0b_preprocessed1.csv')


def fetch_filenames(subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho':'_rois_ho.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)  # os.path.join(data_folder, subject_IDs[i]))
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames

    
"""
# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
    
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        #ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        # Search for the file directly in the rois_ho folder
        #ro_file = [f for f in os.listdir(os.path.join(data_folder, 'rois_ho')) if f.endswith('_rois_' + atlas_name + '.1D') and subject_list[i] in f]
        ro_file = [f for f in os.listdir(os.path.join(data_folder, 'rois_ho')) if f.endswith('_rois_' + atlas_name + '.1D') and subject_list[0] in f]

        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries

"""

def get_timeseries(subject_list, atlas_name):
    """
    Fetches timeseries data for all subjects in the given subject list.
    
    Parameters:
        subject_list (list): List of subject IDs.
        atlas_name (str): The atlas name used (e.g., 'ho').
    
    Returns:
        list: A list of timeseries arrays, each of shape (timepoints x regions).
    """
    timeseries = []
    for subject in subject_list:
        # Match files based on the subject ID and atlas
        ro_file = [
            f for f in os.listdir(data_folder)
            if f.endswith(f'_rois_{atlas_name}.1D') and subject in f
        ]

        if not ro_file:
            print(f"Warning: No timeseries file found for subject {subject}.")
            continue
        
        try:
            # Load the timeseries data
            file_path = os.path.join(data_folder, ro_file[0])
            print(f"Reading timeseries file: {file_path}")
            timeseries.append(np.loadtxt(file_path, skiprows=0))
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    return timeseries


"""
# Compute connectivity matrices
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path="/home/ravi/ABIDE/"):
    
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity
"""
import os
import scipy.io as sio

import os

def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path="c:\\Users\\HP\\Downloads\\abide\\ABIDE\\Outputs\\connectivity_matrices"):
    """
    Computes connectivity matrices for a single subject.
    
    Parameters:
        timeseries (numpy.ndarray): Timeseries data of shape (timepoints x regions).
        subject (str): Subject ID.
        atlas_name (str): Name of the parcellation atlas used.
        kind (str): Type of connectivity (e.g., 'correlation').
        save (bool): Whether to save the output matrix.
        save_path (str): Path to save the matrices.
    
    Returns:
        numpy.ndarray: Connectivity matrix of shape (regions x regions).
    """
    print(f"Estimating {kind} matrix for subject {subject}")

    # Compute connectivity
    conn_measure = connectome.ConnectivityMeasure(kind=kind)
    connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        # Create subject folder if it doesn't exist
        subject_dir = os.path.join(save_path, str(subject))
        os.makedirs(subject_dir, exist_ok=True)

        # Save the matrix
        subject_file = os.path.join(subject_dir, f"{subject}_{atlas_name}_{kind.replace(' ', '_')}.mat")
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


# Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'c:\\Users\\HP\\Downloads\\abide\\ABIDE\\Outputs\\cpac\\subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# Dimensionality reduction step for the feature vector using a ridge classifier
def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices

def compute_connectivity_matrices():
    """
    This function computes and saves connectivity matrices for all subjects.
    """
    # Get the list of all subjects
    subject_list = get_ids()  # Fetch all subjects without specifying a limit

    # Specify atlas name and connectivity type
    atlas_name = 'ho'  # or 'cc200', depending on your dataset
    kind = 'correlation'  # Choose 'correlation', 'partial correlation', or 'tangent'

    # Specify the save path for the .mat files
    save_path = 'c:\\Users\\HP\\Downloads\\abide\\ABIDE\\Outputs\\connectivity_matrices'

    # Iterate over all subjects and compute connectivity
    for subject in subject_list:
        timeseries = get_timeseries([subject], atlas_name)  # Fetch the timeseries for the subject
        if not timeseries or len(timeseries) == 0:
            print(f"Skipping subject {subject} due to missing timeseries data.")
            continue
        connectivity = subject_connectivity(timeseries[0], subject, atlas_name, kind, save=True, save_path=save_path)

    print("Connectivity matrices saved successfully for all subjects.")

compute_connectivity_matrices()


"""
def compute_connectivity(subject_list, atlas_name, kind, save=True, save_path=data_folder):
    connectivity_matrices = {}
    conn_measure = ConnectivityMeasure(kind=kind)
    atlas_folder = os.path.join(data_folder, f'rois_{atlas_name}')

    for subject in subject_list:
        try:
            time_series_file = os.path.join(atlas_folder, f"{subject}_rois_{atlas_name}.1D")
            time_series = np.loadtxt(time_series_file)
            connectivity_matrices[subject] = conn_measure.fit_transform([time_series])[0]
        except FileNotFoundError:
            print(f"Missing time series file for subject {subject}. Skipping...")
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
    
    return connectivity_matrices
"""

import os

def get_networks(subject_list, kind, atlas_name="ho", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks

    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """
    all_networks = []
    data_folder = 'c:\\Users\\HP\\Downloads\\abide\\ABIDE\\Outputs\\connectivity_matrices'  # Update the path to match the correct directory
    
    for subject in subject_list:
        file_path = os.path.join(data_folder, str(subject), f"{subject}_{atlas_name}_{kind}.mat")
        
        # Check if the file exists before loading
        if not os.path.exists(file_path):
            print(f"Warning: Connectivity matrix file for subject {subject} not found at {file_path}. Skipping subject.")
            continue  # Skip this subject or handle it in another way if needed

        # Load the connectivity matrix if file exists
        matrix = sio.loadmat(file_path).get(variable)
        if matrix is not None:
            all_networks.append(matrix)
        else:
            print(f"Warning: Variable '{variable}' not found in {file_path}. Skipping subject.")

    # If no valid matrices were found, return an empty array or handle as needed
    if not all_networks:
        print("No valid connectivity matrices were found.")
        return np.array([])

    # Convert to vector form and return the stacked matrices
    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix

#computes pairwise distance between subjects using certain phenotypic information
# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]]) 
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, pd_dict):
    pd_affinity = create_affinity_graph_from_scores(['SEX', 'SITE_ID'], pd_dict) 
    distv = distance.pdist(features, metric='correlation') 
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    print(f"Phenotypic affinity shape: {pd_affinity.shape}")  # Should be (921, 921)
    print(f"Feature similarity shape: {feature_sim.shape}")
    adj = pd_affinity * feature_sim  

  # Should be (921, 921)
    return adj





"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from nilearn.connectome import ConnectivityMeasure

def visualize_connectivity_matrix(matrix, subject, atlas_name, kind, save_path):
    
    Visualizes and saves the connectivity matrix as a heatmap.
    
    Parameters:
        matrix (numpy.ndarray): The connectivity matrix (regions x regions).
        subject (str): Subject ID.
        atlas_name (str): Atlas name used for generating the matrix.
        kind (str): Type of connectivity measure (e.g., 'correlation').
        save_path (str): Directory to save the visualizations.
    
    # Create save directory if not exists
    os.makedirs(save_path, exist_ok=True)

    # Plot the connectivity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Connectivity Strength')
    plt.title(f"Connectivity Matrix: {subject} ({kind}, {atlas_name})")
    plt.xlabel('Region Index')
    plt.ylabel('Region Index')

    # Save the heatmap as a PNG file
    save_file = os.path.join(save_path, f"{subject}_{atlas_name}_{kind}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"Connectivity matrix visualization saved for subject {subject}: {save_file}")

def compute_and_visualize_connectivity_matrices():
    
    Computes, visualizes, and saves connectivity matrices for all subjects.
    
    # Get the list of all subjects
    subject_list = get_ids()  # Fetch all subjects without specifying a limit

    # Specify atlas name and connectivity type
    atlas_name = 'ho'  # or 'cc200', depending on your dataset
    kind = 'correlation'  # Choose 'correlation', 'partial correlation', or 'tangent'

    # Specify the save paths for the .mat files and visualizations
    matrix_save_path = '/home/ravi/ABIDE/Outputs/connectivity_matrices/'
    viz_save_path = '/home/ravi/ABIDE/connectivity_matrices_visualizations/'

    # Iterate over all subjects and process
    for subject in subject_list:
        timeseries = get_timeseries([subject], atlas_name)  # Fetch the timeseries for the subject
        if not timeseries or len(timeseries) == 0:
            print(f"Skipping subject {subject} due to missing timeseries data.")
            continue
        
        # Compute connectivity matrix
        connectivity = subject_connectivity(timeseries[0], subject, atlas_name, kind, save=True, save_path=matrix_save_path)
        
        # Visualize and save the matrix
        visualize_connectivity_matrix(connectivity, subject, atlas_name, kind, viz_save_path)

    print("All connectivity matrices computed, visualized, and saved successfully.")

# Execute the function
compute_and_visualize_connectivity_matrices()
"""

