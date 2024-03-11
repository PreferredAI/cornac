# Working with auxiliary data

This tutorial gives an overview of how to work with auxiliary information in recommender systems using Cornac. In our context, auxiliary data stands for information beyond user-item interactions or preferences, which often holds a clue on how users consume items. Examples of such information or modalities are item textual descriptions, user/item reviews, product images, social networks, etc.

## Modality classes and utilities

In addition to implementing readers and utilities for different types of data, the `cornac.data` module provides modality classes, namely `GraphModality`, `ImageModality` and `TextModality`. The purpose of the latter classes is to make it convenient to work with the corresponding modalities by:

- Offering a number of useful routines for data formatting, representation, manipulation, and transformation.
- Freeing users from the tedious process of aligning auxiliary data with the set of training users, items or ratings.
- Enabling cross-utilization of models designed for one modality to work with a different modality. This topic is covered by the tutorials under the [Cross-Modality](./README.md#Cross-Modality) section.    

In the following, we will discover the text modality by going through a concrete example involving text auxiliary information. The same principles would apply for the other modalities, when dealing with graph (e.g, social network) or visual (e.g., product images) auxiliary data.

## Example: text auxiliary data

For convenience, assume that we are fitting and evaluating a recommender model leveraging item textual auxiliary data.  As a running example, let's consider the Collaborative Deep Learning ([CDL](../cornac/models/cdl)) model, which is already implemented in Cornac.
 
 
### Dataset
We use the well-known MovieLens 100K dataset. It consists of user-movie interactions in the triplet format `(user_id, movie_id, rating)`, as well as movie plots in the format `(movie_id, text)`, which represent our textual auxiliary information. This dataset is already accessible from Cornac, and we can load it as follows,
```Python
import cornac
from cornac.data import Reader
from cornac.datasets import movielens

plots, movie_ids = movielens.load_plot()
rating_data = movielens.load_feedback(reader=Reader(item_set=movie_ids, bin_threshold=3))
```
where we have filtered out movies without plots and binarized the integer ratings using `cornac.data.Reader`.

### The TextModality class

With our dataset in place, the next step is to instantiate a `TextModality`, which will allow us to manipulate and represent our auxiliary data in the desired format.  
 ```Python
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer

item_text_modality = TextModality(corpus=plots, ids=movie_ids, 
                                  tokenizer=BaseTokenizer(sep='\t', stop_words='english'),
                                  max_vocab=5000, max_doc_freq=0.5)
```
In addition to the movie plots and ids, we have specified a `cornac.data.text.Tokenizer` to split text, we limited the maximum vocabulary size to 5000, as well as filtered out words occurring in more than 50% of documents (plots in our case) by setting `max_doc_freq = 0.5`. For more options/details on the `TextModality` please refer to the [docs](https://cornac.readthedocs.io/en/latest/data.html#module-cornac.data.text). 
 
 
**Bag-of-Words text representation.** CDL assumes the bag-of-words representation for text information, i.e., in the form of a document-word matrix. The good news is that we don't have to worry about how to generate such representation from our raw texts. The `TextModality` class implements the necessary routines to process and output different representations for text data, e.g., sequence, bag of words, tf-idf. That is, to get our auxiliary data under the desired format, all we need inside the `CDL` [implementation](../cornac/models/cdl/recom_cdl.py) is the following line of code:
```Python
text_feature = self.train_set.item_text.batch_bow(np.arange(n_items))
``` 
where `self.train_set.item_text` is our `item_text_modality`, `n_items` is the number of training items, and the `batch_bow()` function returns the bag-of-words vectors of the specified item indices, in our case we want the text features for all training items. In more details, the rows of `text_feature` correspond to the bag-of-words vectors of the provided item indices to `batch_bow()`.

**Note.** As evident from the above, we don't have to take extra actions to align the set of training movies with their plots. This is made possible thanks to passing `item_text_modality` through the evaluation (splitting) method as we shall see shortly. 

### Evaluation

We are now left with the standard steps to fit and evaluate a recommender model using Cornac. We use train-test splitting as our evaluation method:
```Python
from cornac.eval_methods import RatioSplit

ratio_split = RatioSplit(data=rating_data, test_size=0.2, exclude_unknowns=True,
                         item_text=item_text_modality, verbose=True,
                         seed=123, rating_threshold=0.5)
``` 
The item text modality is passed to the evaluation method. As mentioned earlier, this makes it possible to avoid the tedious process of aligning the set of training items with their auxiliary data. Moving forward, we need to instantiate the CDL model and evaluation metrics:
```Python
cdl = cornac.models.CDL(k=20, autoencoder_structure=[50], max_iter=100,
                        learning_rate=0.01, lambda_u=0.01, lambda_v=0.01,
                        lambda_w=0.01, lambda_n=1000, vocab_size=5000)
                        
rec_100 = cornac.metrics.Recall(k=100)
``` 
Finally, we put everything together and run our experiment.
```Python
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[cdl],
                        metrics=[rec_100])
exp.run()
```
Output:
```
    | Recall@100 | Train (s) | Test (s)
--- + ---------- + --------- + --------
CDL |     0.5494 |   42.1279 |   0.3018
```

## Other Modality classes

The usage of the `GraphModality` and `ImageModality`, to deal with graph and visual auxiliary data, follows the same principles as above. The [c2pf_example](../examples/c2pf_example.py) and [mcf_example](../examples/mcf_office.py) involve the `GraphModality` class to handle item network. For the `ImageModality`, one may refer to the [vbpr_tradesy](../examples/vbpr_tradesy.py) example. The `cornac.data` module's [documentation](https://cornac.readthedocs.io/en/v2.0.0/api_ref/data.html#module-cornac.data.modality) is also a good resource to know more about the modality classes. 
