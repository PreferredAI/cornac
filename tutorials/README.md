# Tutorials

If you are new to Cornac, the [Getting Started](#getting-started) tutorials are a good beginning point to learn how to use the framework. There is also a rich collection of [examples](../examples#cornac-examples-directory) on how to use specific models, run your evaluation on an existing data split, etc.

## Getting Started

- [Installation](../README.md#installation)
- [Your first Cornac experiment](../README.md#getting-started-your-first-cornac-experiment) 
- [Hyperparameter search for VAECF](./param_search_vaecf.ipynb)
- [Approximate nearest neighbor (ANN) search](./ann_hnswlib.ipynb)
- [Introduction to BPR, using Cornac with Microsoft Recommenders](https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb)
- [Tutorials on recommender systems by Preferred.AI](https://github.com/PreferredAI/tutorials/tree/master/recommender-systems)
- [BiVAE model with Microsoft Recommenders](https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bivae_deep_dive.ipynb)

## Contributing

- [Add a new recommender model](./add_model.md)
- [Add an evaluation metric](./add_metric.md)

## Multimodality
An important focus of Cornac is making it convenient to work with auxiliary data. Tutorials within this section elaborate more on this aspect, including how to work with auxiliary information, perform cross-modality transformations (e.g., from visual or textual features to graph), as well as use models designed for a specific modality (e.g., images) with a different one (e.g., texts).

- [Working with auxiliary data](./working_with_auxiliary_data.md)
- [Text to Graph transformation](./text_to_graph.ipynb)
- [Visual recommendation algorithm with text data](./vbpr_text.ipynb)
- [Multimodality and Cross-Modal Utilization (Tutorial at RecSys 2021)](https://github.com/PreferredAI/tutorials/tree/master/multimodal-recsys)
