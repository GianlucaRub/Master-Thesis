#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pykeen.pipeline import pipeline
from pykeen.datasets import Nations, get_dataset
import torch
from pykeen.evaluation import evaluate, RankBasedEvaluator
from pykeen.metrics.ranking import HitsAtK
import pandas as pd


import logging
from pathlib import Path

import click
import more_click
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.losses import NSSALoss,CrossEntropyLoss
from pykeen.models.inductive import InductiveNodePiece, InductiveNodePieceGNN
from pykeen.trackers import ConsoleResultTracker, WANDBResultTracker, FileResultTracker
from pykeen.training import SLCWATrainingLoop
from pykeen.typing import TESTING, TRAINING, VALIDATION
from pykeen.utils import resolve_device, set_random_seed
from torch.optim import Adam


from pykeen.metrics.ranking import HitsAtK

from pathlib import Path

from pykeen.datasets.inductive.base import DisjointInductivePathDataset
from pykeen.datasets.base import PathDataset, Dataset
from typing_extensions import Literal
import os
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import InductiveNodePiece, TransE, RGCN, QuatE, RotatE
from pykeen.typing import TESTING, TRAINING, VALIDATION

import time

import platform

import sys

import cpuinfo

import psutil

import subprocess

import zipfile

seed = 1234


# In[3]:


def show_metrics(dictionary,model_name,csv_name):
    for key in dictionary.keys():
        print(key)
        df = pd.DataFrame(dictionary[key])
        df.to_csv(f"{model_name}/{model_name}_{csv_name}_{key}.csv")
        print(df)


# In[4]:


DATA_TYPE = "_transductive.tsv"
TRAIN_PATH = "MSCallGraph_train" + DATA_TYPE
TEST_PATH = "MSCallGraph_test" + DATA_TYPE
VALIDATE_PATH = "MSCallGraph_validation" + DATA_TYPE


# In[7]:


dataset = PathDataset(training_path = TRAIN_PATH,
                     testing_path = TEST_PATH,
                     validation_path = VALIDATE_PATH,
                      eager = True
                     )


# In[8]:


model_name = 'rotatE_transductive'


# In[9]:


tracker = ConsoleResultTracker()


# In[10]:


loss = NSSALoss() #used by RotatE and NodePiece
embedding_dim = 200


# In[11]:


model = RotatE(
        triples_factory=dataset.training,
        random_seed = seed,
        loss = loss,
        embedding_dim = embedding_dim
    ).to(resolve_device())
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Space occupied: {model.num_parameter_bytes} bytes")


# In[12]:


directory = model_name

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f'Directory {directory} created successfully!')
else:
    print(f'Directory {directory} already exists.')


# In[13]:


learning_rate = 5e-3 #1e-3 and 1e-2 stops after 20 iterations and actually does not reduce the error too much
optimizer = Adam(params=model.parameters(), lr=learning_rate)
num_epochs = 2000
patience = 20


# In[14]:


metrics = ['meanreciprocalrank', HitsAtK(1),
                 HitsAtK(3), HitsAtK(5), HitsAtK(10)]

train_evaluator = RankBasedEvaluator(
        metrics=metrics,
        add_defaults=False,
    )
valid_evaluator = RankBasedEvaluator(
        metrics=metrics,
        add_defaults=False,
    )
test_evaluator = RankBasedEvaluator(
        metrics = metrics,
        add_defaults=False
    )


# In[15]:


from pykeen.stoppers import EarlyStopper

stopper = EarlyStopper(
    model = model,
    metric='meanreciprocalrank',
    patience=patience,
    frequency=1,
    evaluator = valid_evaluator,
    training_triples_factory = dataset.training,
    evaluation_triples_factory = dataset.validation,
    result_tracker = tracker

)



# In[16]:


# default training regime is negative sampling (SLCWA)
# you can also use the 1-N regime with the LCWATrainingLoop
# the LCWA loop does not need negative sampling kwargs, but accepts label_smoothing in the .train() method
training_loop = SLCWATrainingLoop(
        triples_factory=dataset.training,
        model=model,
        result_tracker=tracker,
        optimizer=optimizer
)


# In[17]:


training_start = time.time()
train_epoch =  training_loop.train(
        triples_factory=dataset.training,
        num_epochs=num_epochs,
#         callbacks="evaluation",
#         callback_kwargs=dict(
#             evaluator=valid_evaluator,
#             evaluation_triples=dataset.validation.mapped_triples,
#             prefix="validation",
#             frequency=1,
#             additional_filter_triples=dataset.training.mapped_triples,
#         ),
        stopper = stopper
        
    )
training_duration = time.time() - training_start


# In[18]:


print("Train error per epoch:")
df = pd.DataFrame(train_epoch)
print(df)
df.to_csv(f"{model_name}/{model_name}_train_error_per_epoch.csv")


# In[19]:


training_evaluation_start = time.time()
# train
print("Train error")
show_metrics(train_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.training.mapped_triples,
        additional_filter_triples=[
        dataset.training.mapped_triples,
    ]
    ).to_dict(),model_name,'train_metrics')
training_evaluation_duration = time.time() - training_evaluation_start


# In[20]:


validation_evaluation_start = time.time()
# validation
print("Validation error")
show_metrics(valid_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.validation.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.training.mapped_triples
        ],
    ).to_dict(),model_name,'validation_metrics')
validation_evaluation_duration = time.time() - validation_evaluation_start


# In[21]:


testing_evaluation_start = time.time()
# result on the test set
print("Test error")
show_metrics(test_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    ).to_dict(),model_name,'test_metrics')
testing_evaluation_duration = time.time() - testing_evaluation_start


# In[22]:


infodict = {}
infodict['device'] = model.device
infodict['parameters bytes'] = model.num_parameter_bytes
infodict['number parameters'] = model.num_parameters
infodict['training duration'] = training_duration
infodict['training evaluation duration'] = training_evaluation_duration
infodict['validation evaluation duration'] = validation_evaluation_duration
infodict['testing evaluation duration'] = testing_evaluation_duration
infodict["Operating system name"] = platform.system()
infodict["Operating system version"] = platform.release()
infodict["Processor architecture"] = platform.machine()
infodict["Python version"] = sys.version
infodict["Processor model name"] = cpuinfo.get_cpu_info()['brand_raw']
infodict['Number cpu cores'] = os.cpu_count()
infodict["Total physical memory"] = psutil.virtual_memory().total


# In[23]:


output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv'])
output = output.decode('utf-8')  # convert byte string to regular string

# split output into rows and remove header row
rows = output.strip().split('\n')[1:]

# extract GPU names from each row
gpu_names = []
for row in rows:
    name = row.strip()
    gpu_names.append(name)

infodict['GPU'] = gpu_names[0]


# In[24]:


infodict['loss'] = NSSALoss
infodict['embedding_dim'] = embedding_dim
infodict['learning_rate'] = learning_rate
infodict['optimizer'] = Adam
infodict['num_epochs'] = num_epochs
infodict['patience'] = patience


# In[25]:


info_df = pd.DataFrame(columns=['name','value'], data = infodict.items())
info_df.to_csv(f"{model_name}/{model_name}_information.csv")
print(info_df)


# In[26]:


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file))

folder_path = model_name
output_path = f'{model_name}.zip'

zip_folder(folder_path, output_path)


# In[ ]:




