########################################################################

# Import standard modules
import os
import numpy as np
import pandas as pd
import scipy as sp
import random
import pickle
import os
import time
from scipy import sparse
from random import sample
from tqdm import tqdm

# Import NN framework
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.nn.parallel
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import BatchSampler, SequentialSampler
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchvision import models

########################################################################
# Graph data utility functions

# Feed in the raw summarised/full data to map contracts and clauses to an index which can be accessed to find out what they are
# This is the one that is mainly used
def create_data(data, export, name1, name2, name3, verbose):
    """
    Inputs: data: a string object, location of the .pickle file
            export: a boolean value, exports the transformed data set to a .pickle file
            name1: str value, name of the exported graph structure data, must have .pickle 
            name2: str value, name of the data dictionary to identify contracts
            name3: str value, name of the data dictionary to identfiy clauses
            transpose: a boolean value, transpose data or not
            verbose: a boolean value, shows what the DataFrame object looks like
    Outputs: df: a pd.DataFrame object, the data set
             contracts: a pd.DataFrame object, the mapping of contracts to a numerical ID
             clauses: a pd.DataFrame object, the mapping of clauses to a numerical ID
    """
    start = time.perf_counter()
    
    if '.pickle' in data:
        # read pickle file 
        file = open(f"{data}", "rb")
        raw_data = pickle.load(file)
    else:
        raise ValueError("data is not in a .pickle format")
    
    ### DATA TRANSFORMATION ###
    
    if verbose == True:
        print("Unpivoting data...")
    else:
        pass
    
        # read dictionary and melt it 
    df_to_melt = pd.DataFrame.from_dict(raw_data, orient = 'columns').T
    melt_df = pd.melt(df_to_melt)
    
        # drop rows where it has 'None' in it
    melt_df.dropna(axis = 0, how = 'any', inplace = True)
    
        # creating numerical id's for contracts and clauses
        
    if verbose == True:
        print("Storing indices...")
    else:
        pass
    
    contracts = pd.DataFrame(melt_df.drop_duplicates(subset = 'variable').iloc[:,0].values)
    contracts.columns = ['variable']
    contracts['idx'] = contracts.index
    
    
    clauses = pd.DataFrame(melt_df.drop_duplicates(subset = 'value').iloc[:,1].values)
    clauses.columns = ['value']
    clauses['idx'] = clauses.index
    
        # Create a df which has a contracts and clauses with no gaps - i.e each row is a clause in a contract
    df = melt_df.merge(contracts, on = 'variable', how = 'left')
    df = df.merge(clauses, on = 'value', how = 'left')
    
        # create our rating level
    df['rating'] = 1
    
        # rename cols to something nicer
    df.columns = ['contract_id','clause','contract_idx','clause_idx','rating']
    
    # drop the old identifiers
    df.drop(labels = ['contract_id', 'clause'], axis = 1).head()
    
        # Create a key in contracts and clauses dfs
    
    contracts['key'] = 1
    clauses['key'] = 1
    
    if verbose == True:
        print("Creating graph data...")
        print("")
    else:
        pass

        # Now we need to form a table where each row is every possible clause
        # therefore rating is binary either 1 it exists, or 0 it doesn't exist
    df2 = pd.merge(left = contracts, right = clauses,  how = 'left', on = 'key')
    df2 = df2.drop(labels = ['key', 'variable'], axis = 1)
    df2 = df2[['idx_x','idx_y','value']]
    df2.columns = ['contract_idx', 'clause_idx', 'clause']
    df2['rating'] = 0
    
    out = pd.merge(left = df2, right = df, on = ['contract_idx', 'clause_idx'], how = 'left')
    out = out.drop(labels = ['contract_id', 'clause_y', 'clause_x','rating_x'], axis = 1)
    out.columns = ['contract_idx', 'clause_idx', 'rating']
    out = out.fillna(value = 0)
    
    # convert rating column to int
    out.rating = out.rating.astype(np.int32)
    
    # Drop the keys in contract and clauses dfs
    
    contracts.drop(labels = 'key', axis = 1, inplace = True)
    clauses.drop(labels = 'key', axis = 1, inplace = True)
    
    # show data sets
    if verbose == True:
        print("Contract indices:")
        print(contracts.head())
        print("")
        print("Clause indices:")
        print(clauses.head())
        print("")
        print("Data for graph model:")
        print(out.head())        
    else:
        pass
    
    # export as pickle
    if export == True:
        print( f"{name1}", "has been exported to:", "D:\Python\Thesis\gcmc_ready_data\\")
        out.to_pickle(("D:\Python\Thesis\gcmc_ready_data\\" + f"{name1}"))
        
        print( f"{name2}", "has been exported to:", "D:\Python\Thesis\gcmc_ready_data\\")
        contracts.to_pickle(("D:\Python\Thesis\gcmc_ready_data\\" + f"{name2}"))
        
        print( f"{name3}", "has been exported to:", "D:\Python\Thesis\gcmc_ready_data\\")
        clauses.to_pickle(("D:\Python\Thesis\gcmc_ready_data\\" + f"{name3}"))
        
    print("Time elapsed in mins: ", (time.perf_counter() - start)/60)
            
    return out, contracts, clauses  

def map_data(data):
    """
    Map data to proper indices in case they are not in a continuous [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    seq = map(lambda x: id_dict[x], data)
    data = np.array(list(seq))
    n = len(uniq)

    return data , id_dict, n

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sparse.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm

def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sparse.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sparse.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sparse.hstack([u_features, zero_csr_u], format='csr')
    v_features = sparse.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features

def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sparse.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sparse.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm

def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

########################################################################

# This creates our training, validation and testing splits 
def create_trainvaltest_split(data, train_p, val_p, test_p, export, location, fracc):
    """
    Inputs: data: a pd.DataFrame object, our data with columns ['contract_idx','clause_idx','rating']
            train_p: a float object, % of data to use for training
            val_p: a float object, % of data to use for validation
            test_p: a float object, % of data to use for testing
            export: a boolean object, decides to export to pd.DataFrames or not
            location: a str object, place to export the 3 files, REQUIRES '\\' at the end of the str, see notes
            fracc: an int value, when exporting features, this is a label for the filename indicating features for what sampled data set
    
    Outputs: rating_train: a pd.DataFrame object containing train_p% of the data
             rating_val: a pd.DataFrame object containing val_p% of the data
             rating_test: a pd.DataFrame object containing test_p% of the data
    
    Notes: train_p + val_p + test_p = 1
           location = "D:\Python\Thesis\gcmc_ready_data\\"
           ignore the df indices in the output and refer to the values instead
   
    """ 
    
    if train_p + val_p + test_p != 1:
        raise ValueError("Training, Validation and Testing proportions must sum to 1.")
    else:
        pass
    
    # needed so we can take a % of the data
    total_length = len(data)
    
    # shuffle the data, no replacement since we just want a shuffle and reset index
    shuffled_data = data.sample(frac = 1, random_state = 1234, replace = False)
    shuffled_data.reset_index(inplace = True)
    
    #85% training
    len_train = int(total_length*0.85)
    
    # 5% validation, here we do 0.9 since 0.9-0.85 = 0.05
    len_val   = int(total_length*0.9)
    
    # create train,val,test splits and drop the additional column created
    rating_train = shuffled_data[:len_train].drop(labels = 'index', axis = 1) # 85% training
    rating_val   = shuffled_data[len_train:len_val].drop(labels = 'index', axis = 1) # 5% validation
    rating_test  = shuffled_data[len_val:].drop(labels = 'index', axis = 1) # 10% testing
    
    
    if export == True:
        rating_train.to_pickle(f"{location}" + f'{fracc}' + "train" + f"{train_p*100}" + ".pickle")
        rating_val.to_pickle(f"{location}" + f'{fracc}' + "val" + f"{val_p*100}" + ".pickle")
        rating_test.to_pickle(f"{location}" + f'{fracc}' + "test" + f"{test_p*100}" + ".pickle")
        print("Training, validation and testing splits have been exported to:",location)
    else:
        print("Data not exported")
        
    return rating_train, rating_val, rating_test

# This obtains our features
def get_features(location, export, exp_loc, frac):
    """
    Inputs: location: a str value, location containing the pickled data file
            export: a boolean value, decides to export features or not
            exp_loc: a str value, location to export features, must have '\\' at the end of str
            frac: an int value, when exporting features, this is a label for the filename indicating features for what sampled data set
            
    Outputs: num_users: an int object, number of users (aka contracts)
             num_items: an int object, number of items (aka clauses)
             num_side_features: an int object, number of side features (additional features outside of data)
             num_features: an int object, number of features (features within the data)
             u_features: a sparse matrix object, of size num_users x num_users, i.e user features 
             v_features: a sparse matrix object, of size num_items x num_items, i.e item features
             u_features_side: a sparse matrix object, of size num_users x num_users, i.e user features 
             v_features_side: a sparse matrix object, of size num_items x num_items, i.e item features
             
    Notes:
    All features saved are in .npy format and can be read using numpy
    Builds and returns features
    example location: "D:\\Python\\Files\\Graph\\new_data.pickle"
    example exp_loc: "D:\\Python\\Files\\Graph\\"
    """
    start = time.perf_counter()
    
    # Set up data types to convert our columns to
    dtypes = {
            'contract_idx': np.int32, 
            'clause_idx': np.int32,
            'rating': np.float32
            }
    
    if '.pickle' in location:
    
    # Read the raw data raw data has ['contract_idx', 'clause_idx, 'rating']
        raw_data = pd.read_pickle(location)   
    else:
        raise ValueError("fname variable is not in a .pickle format, convert data to .pickle/find correct location of data")
        
    # Show number of classes or level ratings
    
    num_classes = raw_data['rating'].nunique()
    
    # Set up some variables containing dimensions of the data
    
        # unique contracts
    num_users = raw_data.drop_duplicates(subset = 'contract_idx').iloc[:,0].shape[0]
    
        # unique clauses
    num_items = raw_data.drop_duplicates(subset = 'clause_idx').iloc[:,0].shape[0]
    
    ########
    
    # Features go here - this section is work-in-progress and is totally defined by
        # what features we want to use it can be anything, from explicit to implicit data
        # or even sentence embeddings.
    
        # Examples are tf-idf, topics, demographics inferred on the contract
    
    # For now use identity matrix as a feature to test
    
    u_features = sparse.csr_matrix(np.eye(N = num_users, dtype = np.float32))
    v_features = sparse.csr_matrix(np.eye(N = num_items, dtype = np.float32))
    
    ########
    
    print("Normalizing feature vectors...")
    # In this case because we have no side features we use u_features and v_features and normalise them
    # replace for real side features if necessary
    u_features_side = normalize_features(u_features) # vector of size (num_users+num_items)x1
    v_features_side = normalize_features(v_features) # vector of size (num_users+num_items)x1
    print("Normalizing complete!")
    
    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)
    
    # Uncomment to get array files, it is in sparse format
    #u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    #v_features_side = np.array(v_features_side.todense(), dtype=np.float32)
    
    num_side_features = u_features_side.shape[1]
    
    # node id's for node input features
    id_csr_u = sparse.identity(num_users, format='csr')
    id_csr_v = sparse.identity(num_items, format='csr')
    
    
    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)
    
    # Uncomment to get array files, it is in sparse format
    #u_features = u_features.toarray()
    #v_features = v_features.toarray()
    
    num_features = u_features.shape[1]
    print(u_features.shape)
    
    # create a helpful dictionary to be exported so we don't need to remember the numbers to input into graph model
    helpful_dict = {"Num_users": num_users,
                    "Num_items": num_items,
                    "Num_classes": num_classes,
                    "Num_features": num_features,
                    "Num_side_features": num_side_features
                   }
    
    if export == True and type(exp_loc) == str:
        sparse.save_npz(exp_loc + "u_features" + f'{frac}', u_features)
        sparse.save_npz(exp_loc + "v_features" + f'{frac}', v_features)
        sparse.save_npz(exp_loc + "u_features_side" + f'{frac}', u_features_side)
        sparse.save_npz(exp_loc + "v_features_side" + f'{frac}', v_features_side)
        
        out = open(exp_loc + "gcmc_dims" + f'{frac}' + ".pickle", "wb")
        pickle.dump(helpful_dict, out)
        out.close()
        
        print("Features and dim-dict have been exported to:", exp_loc)
        print("Features exported are .npz use scipy to load")
    else:
        print("Features not exported, export disabled or location is not str-type")
        
    
    print("Time elapsed in mins: ", (time.perf_counter() - start)/60)
    
    # return first line is numbers, second line is matrices

    return num_users, num_items, num_side_features, num_features, \
           u_features, v_features, u_features_side, v_features_side

# This function converts our data to tensor form to be fed into the NN, beware that files outputted can be large and need compression
def tensor_graph(r_train, r_val, r_test, exp_loc, frac):
    """
    Inputs: r_train: a pd.DataFrame object, the training set exported from create_trainvaltest_split()
            r_val: a pd.DataFrame object, the training set exported from create_trainvaltest_split()
            r_test: a pd.DataFrame object, the training set exported from create_trainvaltest_split()
            exp_loc:a str object, directory (must have \\ at the end of the str) of where to export the split tensors 
            frac: an int value, when exporting features, this is a label for the filename indicating features for what sampled data set
            
    Outputs: training, validation, testing splits in tensor format as a file, otherwise nothing
    
    Notes: function takes in training, validation and testing rating matrices 
           and a parameter defining the number of classes/ratings to create a tensor of r x u x v
           where r = ratings, u = users, v = items
           example of exp_loc: exp_loc = 'D:\Python\Thesis\gcmc_ready_data\\'
           function get_features is used to get num_users and num_items
    """
    
    if os.path.isdir(f'{exp_loc}') == False:
        raise ValueError("Directory does not exist or is not a directory")
    else:
        pass
    
    start = time.perf_counter()
    
    # define the number of classes/ratings, use val set since it is smaller
    # adjust to train set for assurance of class existence 
    num_classes = len(r_val.rating.value_counts())
    
    # get the num_users and num_items
    # Option 1: use get_features()
    print("Getting user and item counts...")
    #num_users, num_items,_,_, \
    #_,_,_,_ = get_features(location = "D:\Python\Thesis\gcmc_ready_data\\data.pickle", export = False, exp_loc = None)
    # Option 2: get it from gcmc_dims from the working directory i.e exp_loc
    pickle_in = open(exp_loc + "gcmc_dims" + f'{frac}' + ".pickle", "rb")
    dims = pickle.load(pickle_in)
    num_users = dims['Num_users']
    num_items = dims['Num_items']

    print("Converting to tensor object...")
    idx = ['train','val','test'] # list of names put into output files
    for i, ratings in enumerate(tqdm([r_train, r_val, r_test], position = 0, desc = 'Creating tensor for each split')):
        rating_mtx = torch.zeros(num_classes, num_users, num_items)
        
        for index, row in tqdm(ratings.iterrows(), position = 0, desc = 'Filling in'):
            u = row[0]-1
            v = row[1]-1
            r = row[2]-1
        
            rating_mtx[r, u, v] = 1
            
        torch.save(rating_mtx, f'{exp_loc}' + f'{frac}' + 'ratings_tensor_%s.pkl' % (idx[i]))
        
    print("Tensors exported to:", exp_loc)
    print("Time elapsed in mins: ", (time.perf_counter() - start)/60)
    
    return 