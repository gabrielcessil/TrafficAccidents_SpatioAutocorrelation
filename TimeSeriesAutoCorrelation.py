import numpy as np
import math
import matplotlib.pyplot as plt
from adjustText import adjust_text
import TimeSeriesTransformation as tst
import pandas as pd
from statsmodels.tsa.seasonal import STL

def delta(A, i): return A[i+1] - A[i]

def deltaA_deltaB(A,B,i): return delta(A, i)*delta(B, i)

def sum_AB(A,B):   
    amount = 0
    n = len(A)-1
    for i in range(n):
        amount += deltaA_deltaB(A,B,i)
    return amount

def sum_Asquare(A):   
    amount = 0

    n = len(A)-1
    for i in range(n):
        amount += delta(A,i)**2
    
    return amount
    
def timing_correlation(A,B): return sum_AB(A,B) / (  math.sqrt(sum_Asquare(A)) * math.sqrt(sum_Asquare(B))  )
   
def mapping(a,alpha=2): return 2/(1+math.exp(alpha*a))

#def distance(A,B): return math.dist(A,B) #np.linalg.norm(np.array(A)-np.array(B)).tolist()
    
# Distance normalized by the deviation: sum of z-score of each timestep
def average_zScore(A,B,std): 
    return np.sum([(a-b)/s for a,b,s in zip(A,B,std)])/len(A)

# X is a list of time series
def average_perTimeStep(X):
    # Compute the average for each index
    averages = []
    list_length = len(X[0]) # Number of time steps
    for k in range(list_length):
        elements = [timeSeries[k] for timeSeries in X]
        averages.append(np.mean(elements))
    return averages
        
def std_perTimeStep(X):
    std = [] # The std for each time step k
    list_length = len(X[0]) # Number of time steps
    for k in range(list_length): # Iterate each time step
        # Get the elements of all time series at time step k
        elements = [timeSeries[k] for timeSeries in X] 
        std.append(np.std(elements))
    return std
    

def timingScore(W,X,alpha):
    X_ = X.mean(axis=1) 
    X_std = X.std(axis=1)
    
    Z_uf = {}
    for state_name, series_x in X.items():
        Z_mag_uf = average_zScore( series_x.to_list(), X_.to_list(), X_std.to_list())
    
        Corr_uf = timing_correlation( series_x.to_list(), X_.to_list())
    
        Z_uf[state_name] = mapping(Corr_uf,alpha)*Z_mag_uf
    
    return Z_uf

def get_LocalMoranIndex(W,scores):
    I_list = []
    N = len(scores)
    mean_Z = np.mean(scores)
    std_Z = np.std(scores)
    
    def ZScore(z): return (z-mean_Z)/std_Z
        
    for i in range(N):
        ILi = 0
        for j in range(N):
            ILi += W[i][j]*ZScore(scores[j])
            
        ILi = ILi*ZScore(scores[i])
        
        I_list.append(ILi)
    
    return I_list
    
def get_GlobalMoranIndex(W,Z):
    N = len(Z)
    
    sum_W = 0
    sum_WZZ = 0
    sum_Z = 0
    
    for i in range(N):
        for j in range(N):
            sum_W += W[i][j]
            sum_WZZ += W[i][j]*Z[i]*Z[j]
        sum_Z += Z[i]**2
    
    I = (N/sum_W)*(sum_WZZ/sum_Z)

    return I

# Função para criar a matriz de pesos espaciais
def create_weight_matrix(neighbors, included_states=None):
    if included_states is None or not included_states:
        included_states = list(neighbors.keys())
    
    adjacency_matrix = []
    
    for state in included_states:
        row = [1 if neighbor in neighbors[state] and neighbor in included_states else 0 for neighbor in included_states]
        adjacency_matrix.append(row)
    
    return adjacency_matrix, included_states

def standardize_adjacency_matrix(adjacency_matrix):
    adjacency_matrix = np.array(adjacency_matrix, dtype=float)  # Converter para numpy array
    row_sums = adjacency_matrix.sum(axis=1, keepdims=True)  # Soma dos elementos de cada linha
    # Evitar divisão por zero: se a soma de uma linha for zero, mantemos a linha inalterada
    standardized_matrix = np.divide(adjacency_matrix, row_sums, where=row_sums!=0)
    return standardized_matrix

    
def getNeighborhoodAvg(Z_uf_dict,neighbors):
    
    neighborhood_avg = {}
    for uf, neighborhood in neighbors.items():
        uf_sum = 0
        for neig_uf in neighborhood:
            uf_sum += Z_uf_dict[neig_uf] if neig_uf in Z_uf_dict else 0
            
        neighborhood_avg[uf] = uf_sum / len(neighborhood)
        
    return neighborhood_avg

    
def MoranIndex(valueTimeSeries_Dataset,neighbors,alpha=2):
        # Create the standardize neighborhood matrix
        uf_list = list(neighbors.keys())
        W, included_states = create_weight_matrix(neighbors,uf_list)
        W = standardize_adjacency_matrix(W)
            
        # Calculate the score of each state
        states_Z = timingScore(W,valueTimeSeries_Dataset, alpha)
        # Reorganize scores to match the neighbors sequence
        sorted_states_Z = {}
        for state in included_states:
            sorted_states_Z[state] = states_Z[state] if state in states_Z else 0
    
        
        # Calculate the Moran's Index
        gm_index = get_GlobalMoranIndex(W,list(sorted_states_Z.values()))
        lm_index = get_LocalMoranIndex(W,list(sorted_states_Z.values()))
        
        
        I_uf_dict = dict(zip(included_states, lm_index))
        
        return gm_index, I_uf_dict, sorted_states_Z

import seaborn as sns

def get_quartis(Z_uf_dict, neighbors):
    
    z_values = list(Z_uf_dict.values()) 
    z_values_norm = (np.array(z_values) - np.mean(z_values)) / np.std(z_values)
    neighborhood_avg = getNeighborhoodAvg(Z_uf_dict, neighbors)
    y_values = list(neighborhood_avg.values())
    
    quartil_uf = {}
    for uf,x_value,y_value in zip (list(Z_uf_dict.keys()),z_values_norm,y_values):
        if x_value > 0 and y_value > 0:
            quartil_uf[uf] = 'AA'
        elif x_value > 0 and y_value <0:
            quartil_uf[uf] = 'AB'
        elif x_value < 0 and y_value >0:
            quartil_uf[uf] = 'BA'
        else:
            quartil_uf[uf] = 'BB'
    
    return quartil_uf

def normalize_by_dict(df, state_dict, per_ = 100000):
    """
    Normalizes each column of the DataFrame by the corresponding value from the dictionary.

    Parameters:
    - df (pd.DataFrame): DataFrame with state names as columns.
    - state_dict (dict): Dictionary with state names as keys and the values to divide by.

    Returns:
    - pd.DataFrame: Normalized DataFrame.
    """
    
    df_normalized = df.copy()
    for state in df.columns:
        if state in state_dict:
            df_normalized[state] = df[state]*per_ / state_dict[state]
        else:
            print(f"Warning: {state} not found in state_dict. Column will remain unchanged.")
    return df_normalized
 
