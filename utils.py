import numpy as np
import pandas as pd

def coherence(distance_matrix, lag, tolerance, classification):
    ul = lag + tolerance
    ll = lag - tolerance
    mask = np.logical_or(np.less(distance_matrix, ll), np.less(ul, distance_matrix))
    (row, col) = np.where(mask==False)
    correct = 0
    total_inlag = row.shape[0]
    for i in range(total_inlag):
            if classification[row[i]] == classification[col[i]]:
                correct += 1
    unique, abscount = np.unique(classification, return_counts=True)
    probs = abscount/len(classification)
    denom = (probs**2).sum()
    return (correct/total_inlag)/denom

# Function for calculating the BIC from a kmeans clustering model

def BIC_kmeans(kmeans, data):
    
    k = kmeans.cluster_centers_.shape[0]
    r, m = data.shape
    var_mle = 1/(m*r)*kmeans.inertia_
    predict = kmeans.predict(data)
    Rn = np.unique(predict, return_counts = True)[1]
    logRn = np.log(Rn)
    
    logl = np.sum(Rn*logRn) - r*m/2*np.log(2*np.pi*var_mle) - m/2*(r-k)  - r*np.log(r)
    params = k + m*k
    BIC = -logl + params*np.log(r)/2
    
    return BIC

## Function to calculate NCE 
## (in fuzzy-c-means, the particion entropy coefficient is calculated using log base 2, while in Lark paper it is calculated using base e)

def calc_NCE(fcm_model, k):
    nce = (-np.sum(fcm_model.u * np.log(fcm_model.u)) / fcm_model.n_samples)/np.log(k)
    return nce

# A function is created to perform smoothing with different values for the smoothing range (R)

def smoothing(prev_memb, distance_matrix, r_vario, r_smoothing, coords):

    distance_mat = np.where(distance_matrix > r_smoothing, np.nan, distance_matrix) # replace distances larger than R by nan
    distance_df = pd.DataFrame(distance_mat, index=coords.index, columns=coords.index)
    w_nonstd =  np.exp(-(distance_df/r_vario)**2)    #calculate 1-f(hij)
    col_sum = w_nonstd.sum()
    weights = np.array(w_nonstd/col_sum) #column index corresponds with i, row number with j
    weights = np.nan_to_num(weights, nan=0, copy=True)
    memb_upd = np.matmul(weights.T, prev_memb) # Matrix mult. of weights with the membership values of FKM gives the updated memberships
    hardmemb_upd = np.argmax(memb_upd, axis=1) # Hard classification is obtained by classifying each point to the class of largest membership
    return (memb_upd, hardmemb_upd)