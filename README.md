# LGBM Classifier
  This is a classifier for SpMMul based on LightGBM
  By loading 'model.txt', you will get access to a well-trained model
  The input of the model is a vector with 7 entried, namely 'nnz', 'mat_size', 'dev_row','k', 'CUDA_cores', 'bandwidth' and 'L2_cache'.
  The output of the model is a tag (from 0 to 7), corresponding to the 8 algorithms.
  In the folder 'data', we provide some sample data for the project, by runing lgbm_main.py directly, you will get to know more about the project.  
