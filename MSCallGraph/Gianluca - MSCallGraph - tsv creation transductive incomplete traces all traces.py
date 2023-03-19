#!/usr/bin/env python
# coding: utf-8

# I am creating a single trace in a transductive setting, where the train part of the trace dataset contains all the entities and relations

# In[1]:


import pandas as pd
import numpy as np
import os
import joblib
import zipfile
from tqdm import tqdm
import contextlib
import asposecells
import jpype
from asposecells.api import Workbook, FileFormatType
jpype.startJVM()

seed = 1234


# In[2]:


# # specify the name of the zip file and the directory to extract the contents
# zip_file_name = f'MSCallGraph_joined_reduced.zip'
# extract_dir = ""

# # create a ZipFile object and extract the contents to the specified directory
# with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)


# In[3]:


complete_df = pd.read_csv('MSCallGraph_joined_reduced.csv').drop(['Unnamed: 0'],axis=1)
complete_df


# In[4]:


# triplets_df = complete_df.drop(['traceid'],axis=1)
# triplets_df


# In[5]:


# df = triplets_df
df = complete_df.drop(['traceid'],axis=1)
df = df.drop_duplicates()
df


# In[6]:


entities = set(df['dm']) | set(df['um'])
len(entities)


# In[7]:


baseline = []
for entity in entities:
    baseline.append(df[df['dm'] == entity].head(1))
    baseline.append(df[df['um'] == entity].head(1))


# In[8]:


baseline_df = pd.concat(baseline)
baseline_df = baseline_df.drop_duplicates()
baseline_df


# In[9]:


n_train_items = int(len(df)*0.64)
n_items_to_add = n_train_items - len(baseline_df)
print("the number of items to add to the train set is: ",n_items_to_add)


# In[10]:


not_baseline_df = df.merge(baseline_df, how='outer', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge',axis= 1)
not_baseline_df


# In[11]:


not_baseline_train = not_baseline_df.sample(n_items_to_add,random_state = seed)
train_df = pd.concat([baseline_df, not_baseline_train])
train_df


# In[12]:


remaining_df = not_baseline_df.drop(not_baseline_train.index) 


# In[13]:


validation_df = remaining_df.sample(frac = 0.44, random_state = seed)
validation_df


# In[14]:


test_df = remaining_df.drop(validation_df.index)  # drop the sampled rows to get the second DataFram
test_df


# In[15]:


test_traces_df = complete_df.iloc[test_df.index].drop(columns=['um','rpctype','dm'],axis=1)
test_traces_df


# In[16]:


test_traces = list(set(test_traces_df['traceid']))
len(test_traces)


# In[17]:


# complete_test_traces_df = complete_df[complete_df['traceid'].isin(test_traces)].drop(columns=['um','rpctype','dm'],axis=1)
# complete_test_traces_df


# In[18]:


def create_train_test_trace(complete_df, trace):
    trace_df = complete_df[complete_df['traceid']==trace].drop_duplicates()
    trace_triplets_df = trace_df.drop(columns=['traceid'],axis=1).drop_duplicates()

    trace_entities = set(trace_triplets_df['dm']) | set(trace_triplets_df['um'])
    trace_relations = set(trace_triplets_df['rpctype'])
    trace_baseline = []
    for entity in trace_entities:
        trace_baseline.append(trace_triplets_df[trace_triplets_df['dm'] == entity].head(1))
        trace_baseline.append(trace_triplets_df[trace_triplets_df['um'] == entity].head(1))

    for relation in trace_relations:
        trace_baseline.append(trace_triplets_df[trace_triplets_df['rpctype'] == relation].head(1))


    trace_train_df = pd.concat(trace_baseline)
    trace_train_df = trace_train_df.drop_duplicates()

    trace_test_df = trace_triplets_df.drop(trace_train_df.index) 

    return trace_train_df, trace_test_df


# In[19]:


# trace_train_df_list = []
# trace_test_df_list = [] 
# trace_id_list = []
# for trace in test_traces:
#     train_df, test_df = create_train_test_trace(complete_df, trace)
#     if len(test_df) > 0:
#         trace_train_df_list.append(train_df)
#         trace_test_df_list.append(test_df)
#         trace_id_list.append(trace)


# In[20]:


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


# In[21]:


trace_train_df_list = []
trace_test_df_list = [] 
trace_id_list = []

def process_trace(trace):
    train_df, test_df = create_train_test_trace(complete_df, trace)
    if len(test_df) > 0:
        return (train_df, test_df, trace)
    else:
        return None

with tqdm_joblib(tqdm(desc="processing traces", total=len(test_traces))) as progress_bar:
    results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(process_trace)(trace) for trace in test_traces
)

for result in results:
    if result is not None:
        train_df, test_df, trace = result
        trace_train_df_list.append(train_df)
        trace_test_df_list.append(test_df)
        trace_id_list.append(trace)


# In[22]:


def create_triplets(df):
    triplets = []
    for i in range(len(df)):
        head = df.iloc[i]['dm']
        tail = df.iloc[i]['um']
        rel = df.iloc[i]['rpctype']
        triplets.append([head,rel,tail])
    return triplets


# In[23]:


def create_tsv(triplets, file_name):


    # Create Workbook object.
    workbook = Workbook(FileFormatType.TSV)

    # Access the first worksheet of the workbook.
    worksheet = workbook.getWorksheets().get(0)

    # Get the desired cell(s) of the worksheet and input the value into the cell(s).


    i = 1
    for elem in triplets: 
        worksheet.getCells().get("A"+str(i)).putValue(elem[0])
        worksheet.getCells().get("B"+str(i)).putValue(elem[1])
        worksheet.getCells().get("C"+str(i)).putValue(elem[2])
        i+=1


    # Save the workbook as TSV file.
    workbook.save(file_name)


    file = open(file_name,'r')  
    lines = file.readlines()  
    file.close()
    file = open(file_name,'w')  
    lines = lines[:-1]
    file.writelines(lines)
    file.close()


# In[24]:


# Create a directory
directory = "MSCallGraph_traces"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"{directory} created successfully")
else:
    print(f"{directory} already exists")

    # Create a directory
directory = "MSCallGraph_traces/train"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"{directory} created successfully")
else:
    print(f"{directory} already exists")
    # Create a directory
directory = "MSCallGraph_traces/test"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"{directory} created successfully")
else:
    print(f"{directory} already exists")


# In[26]:


with tqdm(desc="writing train tsv", total=len(trace_id_list)):
    for elem in zip(trace_train_df_list,trace_id_list):
        create_tsv(create_triplets(elem[0]),f"MSCallGraph_traces/train/{elem[1]}_transductive_train.tsv")
        tqdm.update(1)


# In[27]:


with tqdm(desc="writing test tsv", total=len(trace_id_list)):  
    for elem in zip(trace_test_df_list,trace_id_list):
        create_tsv(create_triplets(elem[0]),f"MSCallGraph_traces/test/{elem[1]}_transductive_test.tsv")
        tqdm.update(1)


# In[28]:


jpype.shutdownJVM()


# In[29]:


def zip_folder(folder_path, output_path):
    """
    Compresses a folder into a zip file.

    :param folder_path: The path to the folder to be compressed.
    :param output_path: The path and filename of the resulting zip file.
    """

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Iterate over all the files in the folder and add them to the zip file.
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zip_file.write(os.path.join(root, file))

    print(f"{output_path} created successfully!")


# Example usage
folder_path = "MSCallGraph_traces"
output_path = folder_path+".zip"

print("Zipping")
zip_folder(folder_path, output_path)
print("Done")


# In[ ]:




