# Thesis_EMIclassification

The code that was used as part of the data analysis for this thesis is organized in 5 notebooks:

* 1.expl_interp.ipynb

Contains the code for interpolation of the raw data, making use of the gdal library. The raw data is not stored in Github because of its large size.\
Besides, the notebook also contains data exploration in the form of histograms, the correlation matrix and a principal components analysis.

* 2.K_means.ipynb

Contains the code for classifications based on the K-means algorithm. In essence it contains the results of a classification only considering the QP variables 
and another classification also considering the coordinates of each point on the grid. It also has some code exploring how to understand the observed trend in the BIC that does not
reach an optimum for any of the approaches.

* 3.Gaussian_mixture.ipynb

Contains the code for the classification by means of the estimation of a gaussian mixture model. This is done through the EM-algorithm that is available through scikit-learn. 
It also contains an illustration using a toy dataset to better understand the assumptions that are made in this analysis.

* 4.SOM.ipynb

Contains the code for the construction of a Self-Organizing Map using the miniSOM library. To deduce a sensible classification from the SOM, Kmeans is performed on the
SOM grid.

* 5.FKM_smoothing.ipynb

Contains the code for a fuzzy k-means classification followed by spatial smoothing taking into account the spatial autocorrelation of the variables.

* 6.Extra.ipynb

Contains the exploration of some ideas related to what was encountered during the analyses

* utils.py

Contains self-written functions that are called in the analyses.
