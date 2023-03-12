#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pykeen.pipeline import pipeline
from pykeen.datasets import Nations, get_dataset
import torch
from pykeen.evaluation import evaluate, RankBasedEvaluator
from pykeen.metrics.ranking import HitsAtK
import pandas as pd
from pykeen.trackers import ConsoleResultTracker

seed = 1234


# In[2]:


def show_metrics(dictionary):
    for key in dictionary.keys():
        print(key)
        print(pd.DataFrame(dictionary[key]))


# For transE, and therefore for transductive link prediction, it is necessary that all the entities and relations are present in the train set

# In[3]:


batch_size = 8


# In[4]:


from pykeen.hpo import hpo_pipeline
import os
from pykeen.triples import TriplesFactory

TRAIN_PATH = "MSCallGraph_train_transductive.tsv"
TEST_PATH = "MSCallGraph_test_transductive.tsv"
VALIDATE_PATH = "MSCallGraph_validation_transductive.tsv"


training = TriplesFactory.from_path(TRAIN_PATH)
testing = TriplesFactory.from_path(
    TEST_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)
validation = TriplesFactory.from_path(
    VALIDATE_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)

pipeline_result = pipeline(
#    n_trials=3,  # you probably want more than this
    training=training,
    testing=testing,
    validation=validation,
    model='TransE',
    epochs=5,  # short epochs for testing - you should go higher
    random_seed = seed,
)
pipeline_result.save_to_directory(os.getcwd()+'/MSCallGraph_transE_transductive')


# In[5]:


# result on the test set
print("Result on the test set at the end of training")
show_metrics(pipeline_result.metric_results.to_dict())


# In[6]:


# pipeline_result.plot_losses()


# ## Evaluation
# If the results are the same, it means that the traces are the same

# ### Rank Based Evaluator

# inverse harmonic mean rank == mean reciprocal rank https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html

# Pykeen does not allow for a raw evaluation, filtered only.

# In[7]:


model = pipeline_result.model


# In[8]:


results_training_filtered = evaluate(model=model,mapped_triples=pipeline_result.training.mapped_triples,
                            evaluator = RankBasedEvaluator(filtered = True,metrics = ['meanreciprocalrank', HitsAtK(1),
                                                                                      HitsAtK(3), HitsAtK(5), HitsAtK(10)],
                                                          add_defaults = False),mode=None,
                            additional_filter_triples=[pipeline_result.training.mapped_triples],
                                    batch_size = batch_size)
print("Training filtered evaluation")
show_metrics(results_training_filtered.to_dict())


# In[9]:


results_validation_filtered = evaluate(model=model,mapped_triples=validation.mapped_triples,
                            evaluator = RankBasedEvaluator(filtered = True,metrics = ['meanreciprocalrank', HitsAtK(1),
                                                                                      HitsAtK(3), HitsAtK(5), HitsAtK(10)],
                                                          add_defaults = False),mode=None,
                            additional_filter_triples=[pipeline_result.training.mapped_triples],
                                      batch_size = batch_size)
print("Validation filtered evaluation")
show_metrics(results_validation_filtered.to_dict())


# In[10]:


results_testing_filtered = evaluate(model=model,mapped_triples=testing.mapped_triples,
                            evaluator = RankBasedEvaluator(filtered = True,metrics = ['meanreciprocalrank', HitsAtK(1),
                                                                                      HitsAtK(3), HitsAtK(5), HitsAtK(10)],
                                                          add_defaults = False),mode=None,
                            additional_filter_triples=[pipeline_result.training.mapped_triples,
                                                      validation.mapped_triples],
                                   batch_size = batch_size)
print("Testing filtered evaluation")
show_metrics(results_testing_filtered.to_dict())


# In[ ]:




