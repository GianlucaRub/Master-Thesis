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
from pykeen.losses import NSSALoss
from pykeen.models.inductive import InductiveNodePiece, InductiveNodePieceGNN
from pykeen.trackers import ConsoleResultTracker, WANDBResultTracker
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


def show_metrics(dictionary):
    for key in dictionary.keys():
        print(key)
        print(pd.DataFrame(dictionary[key]))


# In[4]:


dataset = InductiveLPDataset()


# In[5]:


model = InductiveNodePieceGNN(
        triples_factory=dataset.transductive_training,
        inference_factory=dataset.inductive_inference,
        random_seed = seed
    ).to(resolve_device())
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Space occupied: {model.num_parameter_bytes} bytes")


# In[6]:


tracker = ConsoleResultTracker()
# default training regime is negative sampling (SLCWA)
# you can also use the 1-N regime with the LCWATrainingLoop
# the LCWA loop does not need negative sampling kwargs, but accepts label_smoothing in the .train() method
training_loop = SLCWATrainingLoop(
        triples_factory=dataset.transductive_training,
        model=model,
        mode=TRAINING,  # must be specified for the inductive setup
        result_tracker=tracker,

    )


# In[7]:


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


# In[8]:


training_loop.train(
        triples_factory=dataset.transductive_training,
        num_epochs=2,
        callbacks="evaluation",
        callback_kwargs=dict(
            evaluator=valid_evaluator,
            evaluation_triples=dataset.inductive_validation.mapped_triples,
            prefix="validation",
            frequency=1,
            additional_filter_triples=dataset.inductive_inference.mapped_triples,
        ),
        
    )


# In[9]:


# train
show_metrics(train_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.transductive_training.mapped_triples,
        additional_filter_triples=[
        dataset.transductive_training.mapped_triples,
    ]
    ).to_dict())


# In[10]:


# validation
show_metrics(valid_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_validation.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.inductive_inference.mapped_triples
        ],
    ).to_dict())


# In[11]:


# result on the test set
show_metrics(test_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_testing.mapped_triples,
        additional_filter_triples=[
            # filtering of other positive triples
            dataset.inductive_inference.mapped_triples,
            dataset.inductive_validation.mapped_triples,
        ],
    ).to_dict())


# In[ ]:




