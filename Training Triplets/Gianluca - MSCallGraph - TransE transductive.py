#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pykeen.pipeline import pipeline
from pykeen.datasets import Nations, get_dataset
import torch
from pykeen.models import predict
from pykeen.evaluation import evaluate, RankBasedEvaluator
from pykeen.metrics.ranking import HitsAtK
import pandas as pd
from pykeen.trackers import ConsoleResultTracker

seed = 1234


# In[2]:


def show_metrics(dictionary):
    for key in dictionary.keys():
        print(key)
        display(pd.DataFrame(dictionary[key]))


# For transE, and therefore for transductive link prediction, it is necessary that all the entities and relations are present in the train set

# In[27]:


batch_size = 8


# In[7]:


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


# In[8]:


# result on the test set
show_metrics(pipeline_result.metric_results.to_dict())


# In[9]:


pipeline_result.plot_losses()


# ## Evaluation
# If the results are the same, it means that the traces are the same

# ### Rank Based Evaluator

# inverse harmonic mean rank == mean reciprocal rank https://cthoyt.com/2021/04/19/pythagorean-mean-ranks.html

# Pykeen does not allow for a raw evaluation, filtered only.

# In[11]:


model = pipeline_result.model


# In[28]:


results_training_filtered = evaluate(model=model,mapped_triples=pipeline_result.training.mapped_triples,
                            evaluator = RankBasedEvaluator(filtered = True,metrics = ['meanreciprocalrank', HitsAtK(1),
                                                                                      HitsAtK(3), HitsAtK(5), HitsAtK(10)],
                                                          add_defaults = False),mode=None,
                            additional_filter_triples=[pipeline_result.training.mapped_triples],
                                    batch_size = batch_size)
print("Training filtered evaluation")
show_metrics(results_training_filtered.to_dict())


# In[29]:


results_validation_filtered = evaluate(model=model,mapped_triples=validation.mapped_triples,
                            evaluator = RankBasedEvaluator(filtered = True,metrics = ['meanreciprocalrank', HitsAtK(1),
                                                                                      HitsAtK(3), HitsAtK(5), HitsAtK(10)],
                                                          add_defaults = False),mode=None,
                            additional_filter_triples=[pipeline_result.training.mapped_triples],
                                      batch_size = batch_size)
print("Validation filtered evaluation")
show_metrics(results_validation_filtered.to_dict())


# In[30]:


results_testing_filtered = evaluate(model=model,mapped_triples=testing.mapped_triples,
                            evaluator = RankBasedEvaluator(filtered = True,metrics = ['meanreciprocalrank', HitsAtK(1),
                                                                                      HitsAtK(3), HitsAtK(5), HitsAtK(10)],
                                                          add_defaults = False),mode=None,
                            additional_filter_triples=[pipeline_result.training.mapped_triples,
                                                      validation.mapped_triples],
                                   batch_size = batch_size)
print("Testing filtered evaluation")
show_metrics(results_testing_filtered.to_dict())


# ## Prediction Inspection

# In[ ]:


model = pipeline_result.model
# Predict tails and see if they are in the train set
predicted_tails_df = predict.get_prediction_df(
    model=model, head_label = "84f9f68ef003a21288fffe8f9a09a5a29b05f4cc4229b8337d1e3c28b6d07923", relation_label="rpc", triples_factory=pipeline_result.training,
)


predicted_tails_df


# In[ ]:


model = pipeline_result.model
# Predict tails and see if they are in the validation set
predicted_tails_df = predict.get_prediction_df(
    model=model, head_label = "84f9f68ef003a21288fffe8f9a09a5a29b05f4cc4229b8337d1e3c28b6d07923", relation_label="rpc", triples_factory=validation,
)


predicted_tails_df


# In[ ]:


model = pipeline_result.model
# Predict tails and see if they are in the test set
predicted_tails_df = predict.get_prediction_df(
    model=model, head_label = "84f9f68ef003a21288fffe8f9a09a5a29b05f4cc4229b8337d1e3c28b6d07923", relation_label="rpc", triples_factory=testing,
)


predicted_tails_df


# In[ ]:


# Predict relations
predicted_relations_df = predict.get_prediction_df(
    model=model, head_label="84f9f68ef003a21288fffe8f9a09a5a29b05f4cc4229b8337d1e3c28b6d07923", tail_label="75e56c8fbb9336eb4dd40f5f609d5344203d374d73fd0b8d2bce19f754070f6a", triples_factory=pipeline_result.training,
)
predicted_relations_df


# In[ ]:


# Predict heads
predicted_heads_df = predict.get_prediction_df(
    model=model, relation_label="rpc", tail_label="75e56c8fbb9336eb4dd40f5f609d5344203d374d73fd0b8d2bce19f754070f6a", triples_factory=pipeline_result.training
)
predicted_heads_df


# In[ ]:


# # Score all triples (memory intensive)
# predictions_df = predict.get_all_prediction_df(model, triples_factory=pipeline_result.training)
# predictions_df


# In[ ]:


# Score top K triples (computationally expensive)
top_k_predictions_df = predict.get_all_prediction_df(model, k=10, triples_factory=pipeline_result.training)
top_k_predictions_df


# In[ ]:


# Score a given list of triples
score_df = predict.predict_triples_df(
    model=model,
    triples=[('84f9f68ef003a21288fffe8f9a09a5a29b05f4cc4229b8337d1e3c28b6d07923',
              'rpc',
              '75e56c8fbb9336eb4dd40f5f609d5344203d374d73fd0b8d2bce19f754070f6a'),
             ('01d660afcfadafd587e20ec4c04ddbc7eb0de95643ba0eec5fc1aeb15e341a85',
              'mc',
              '4ab265f54516248ee8873be7d6441912456ce17e84f39918e01ddc4210e56da5')],
    triples_factory=pipeline_result.training,
)
score_df


# In[ ]:




