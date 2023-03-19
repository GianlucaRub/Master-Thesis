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
from typing_extensions import Literal
import os
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import InductiveNodePiece
from pykeen.typing import TESTING, TRAINING, VALIDATION

import time

import platform

import sys

import cpuinfo

import psutil

import subprocess

import zipfile

seed = 1234


# In[2]:


class InductiveLPDataset(DisjointInductivePathDataset):
    """An inductive link prediction dataset for the ILPC 2022 Challenge."""

    
    
    
    def __init__(self , **kwargs):
        """Initialize the inductive link prediction dataset.

        :param size: "small" or "large"
        :param kwargs: keyword arguments to forward to the base dataset class, cf. DisjointInductivePathDataset
        """
        DATA_TYPE = "_fully_inductive.tsv"
        TRAIN_PATH = "MSCallGraph_train" + DATA_TYPE
        TEST_PATH = "MSCallGraph_test" + DATA_TYPE
        VALIDATE_PATH = "MSCallGraph_validation" + DATA_TYPE
        INFERENCE_PATH = "MSCallGraph_inference" + DATA_TYPE


        super().__init__(
            transductive_training_path=os.getcwd()+"/"+TRAIN_PATH,
            inductive_inference_path=os.getcwd()+"/"+INFERENCE_PATH,
            inductive_validation_path=os.getcwd()+"/"+VALIDATE_PATH,
            inductive_testing_path=os.getcwd()+"/"+TEST_PATH,
            create_inverse_triples=True,
            eager=True,
            **kwargs
        )


# In[3]:


def show_metrics(dictionary,model_name,csv_name):
    for key in dictionary.keys():
        print(key)
        df = pd.DataFrame(dictionary[key])
        df.to_csv(f"{model_name}/{model_name}_{csv_name}_{key}.csv")
        print(df)


# In[4]:


dataset = InductiveLPDataset()


# In[5]:


model_name = 'nodepiece_inductive'


# In[6]:


tracker = ConsoleResultTracker()


# In[7]:


loss = NSSALoss() #used by RotatE and NodePiece
num_tokens = 20
embedding_dim = 200


# In[8]:


model = InductiveNodePiece(
        triples_factory=dataset.transductive_training,
        inference_factory=dataset.inductive_inference,
        random_seed = seed,
        loss = loss,
        num_tokens = num_tokens,
        embedding_dim = embedding_dim
    ).to(resolve_device())
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Space occupied: {model.num_parameter_bytes} bytes")


# In[9]:


directory = model_name

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f'Directory {directory} created successfully!')
else:
    print(f'Directory {directory} already exists.')


# In[10]:


learning_rate = 1e-3
optimizer = Adam(params=model.parameters(), lr=learning_rate)
num_epochs = 200
patience = 20


# In[11]:


metrics = ['meanreciprocalrank', HitsAtK(1),
                 HitsAtK(3), HitsAtK(5), HitsAtK(10)]

train_evaluator = RankBasedEvaluator(
        mode=TRAINING,
        metrics=metrics,
        add_defaults=False,
    )
valid_evaluator = RankBasedEvaluator(
        mode=VALIDATION,
        metrics=metrics,
        add_defaults=False,
    )
test_evaluator = RankBasedEvaluator(
        mode=TESTING,
        metrics = metrics,
        add_defaults=False
    )


# In[12]:


from pykeen.stoppers import EarlyStopper

stopper = EarlyStopper(
    model = model,
    metric='meanreciprocalrank',
    patience=patience,
    frequency=1,
    evaluator = valid_evaluator,
    training_triples_factory = dataset.inductive_inference,
    evaluation_triples_factory = dataset.inductive_validation,
    result_tracker = tracker

)



# In[13]:


# default training regime is negative sampling (SLCWA)
# you can also use the 1-N regime with the LCWATrainingLoop
# the LCWA loop does not need negative sampling kwargs, but accepts label_smoothing in the .train() method
training_loop = SLCWATrainingLoop(
        triples_factory=dataset.transductive_training,
        model=model,
        mode=TRAINING,  # must be specified for the inductive setup
        result_tracker=tracker,
        optimizer=optimizer
)


# In[14]:


training_start = time.time()
train_epoch =  training_loop.train(
        triples_factory=dataset.transductive_training,
        num_epochs=num_epochs,
#         callbacks="evaluation",
#         callback_kwargs=dict(
#             evaluator=valid_evaluator,
#             evaluation_triples=dataset.inductive_validation.mapped_triples,
#             prefix="validation",
#             frequency=1,
#             additional_filter_triples=dataset.inductive_inference.mapped_triples,
#         ),
        stopper = stopper
        
    )
training_duration = time.time() - training_start


# In[ ]:


torch.save(model,f"{model_name}/model.pth")
model = torch.load(f"{model_name}/model.pth")


# In[15]:


print("Train error per epoch:")
df = pd.DataFrame(train_epoch)
print(df)
df.to_csv(f"{model_name}/{model_name}_train_error_per_epoch.csv")


# In[16]:


training_evaluation_start = time.time()
# train
print("Train error")
show_metrics(train_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.transductive_training.mapped_triples,
        additional_filter_triples=[
        dataset.transductive_training.mapped_triples,
    ]
    ).to_dict(),model_name,'train_metrics')
training_evaluation_duration = time.time() - training_evaluation_start


# In[17]:


validation_evaluation_start = time.time()
# validation
print("Validation error")
show_metrics(valid_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_validation.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.inductive_inference.mapped_triples
        ],
    ).to_dict(),model_name,'validation_metrics')
validation_evaluation_duration = time.time() - validation_evaluation_start


# In[18]:


testing_evaluation_start = time.time()
# result on the test set
print("Test error")
show_metrics(test_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_testing.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.inductive_inference.mapped_triples,
            dataset.inductive_validation.mapped_triples,
        ],
    ).to_dict(),model_name,'test_metrics')
testing_evaluation_duration = time.time() - testing_evaluation_start


# In[19]:


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


# In[21]:


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


# In[22]:


infodict['loss'] = NSSALoss
infodict['num_tokens'] = num_tokens
infodict['embedding_dim'] = embedding_dim
infodict['learning_rate'] = learning_rate
infodict['optimizer'] = Adam
infodict['num_epochs'] = num_epochs
infodict['patience'] = patience


# In[23]:


info_df = pd.DataFrame(columns=['name','value'], data = infodict.items())
info_df.to_csv(f"{model_name}/{model_name}_information.csv")
print(info_df)


# In[24]:


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file))

folder_path = model_name
output_path = f'{model_name}.zip'

zip_folder(folder_path, output_path)


# In[ ]:




