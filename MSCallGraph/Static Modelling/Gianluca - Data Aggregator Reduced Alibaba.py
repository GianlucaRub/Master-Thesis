#!/usr/bin/env python
# coding: utf-8

# In this notebook I am joining the previously processed reduced dataset parts. It will be used for creating the trace test set.

# In[1]:


import pandas as pd
import numpy as np
import zipfile
import joblib
from tqdm import tqdm
import contextlib


# In[3]:


to_skip = tuple([2,57,60,64,66,75,98,101,125,129,130,144]) # corrupted files
df_list = []

# define a function to extract and process a single file
def extract_and_process_file(i):  
    if to_skip.count(i) > 0:
        return None
        
    # specify the name of the zip file and the directory to extract the contents
    zip_file_name = f'MSCallGraph_{i}_reduced.zip'
    extract_dir = ""

    # create a ZipFile object and extract the contents to the specified directory
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # read the CSV file and process it
    df = pd.read_csv(f'MSCallGraph_{i}_reduced.csv').drop(['Unnamed: 0'],axis = 1).drop_duplicates()
    return df


# In[4]:


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# In[5]:


# use joblib.Parallel to extract and process the files in parallel
n_jobs = -1 # use all available CPUs
print("cpu ",n_jobs)
with tqdm_joblib(tqdm(desc="aggregating reducedd", total=145)) as progress_bar:
    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(extract_and_process_file)(i) for i in range(0, 145))

# filter out the None values and append the processed dataframes to df_list
for df in results:
    if df is not None:
        df_list.append(df)


# In[6]:


df = pd.concat(df_list).drop_duplicates()
df


# In[7]:


df.to_csv('MSCallGraph_joined_reduced.csv',index=False)


# In[8]:


df = pd.read_csv('MSCallGraph_joined_reduced.csv')
df


# In[10]:


print(df)


# In[9]:


import zipfile
filename = 'MSCallGraph_joined_reduced.csv'
zipfilename = 'MSCallGraph_joined_reduced.zip'

# Create a ZipFile object and add the file to it
with zipfile.ZipFile(zipfilename, 'w', compresslevel=9, compression=zipfile.ZIP_LZMA) as zip:
    zip.write(filename)

