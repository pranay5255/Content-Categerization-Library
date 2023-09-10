# Content-Categerization-Library
The Pipeline uses different libraries and techniques manually implemented in python to handle multi-class classification problems which promises higher accuracy than the native TF-IDF feature extraction pipeline. The pipeline can be used to get baseline scores for text categorisation.
## Algorithms used for Pipeline:
### Filtering -- Zipf Law and Chi-square filters
### Feature Extraction -- TF-IDF, Bag Of Words, GLOVE vectors , Novel document vector aggregation using GloVe vectors
### Modelling -- 
 #### Tree based algorithms --> Light GBM, XGboost, Random Forest
 #### Linear Algorithms --> Logistic, SGD classifier

The pipeline Outputs a classification score report of all individual classes with corresponding weighted Precision and Recall. 
