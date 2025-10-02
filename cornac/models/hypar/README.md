# HypAR changes
We've had to make some changes to the HypAR model to ensure compatibility with numpy 2.x. 
The main change is replacing the gensim Word2Vec logic with `sentence-transformers`, 
which provides high-quality embeddings and is compatible with numpy 2.x. We therefore do not learn embeddings from 
data anymore, but use a pre-trained model instead.
Furthermore, we've updated the requirements file to accomodate the new version.

To validate the new implementation, we ran the original experiments on the Cellphone and Computer datasets. 
The table below shows the results before and after the changes. We observe that these changes do slightly affect the 
performance. If you want to use the original implementation, use an older version of Cornac (before v2.3.0).


| Dataset   | Model Version | AUC        | MAP        | NDCG     |
|-----------|---------------|------------|------------|----------|
| Cellphone | Original      | **0.7533** | 0.0517     | 0.2054   |
|           | Updated       | 0.7493     | **0.0597** | **0.2124** |
| Computer  | Original      | **0.7278** | 0.0194     | **0.1473** |
|           | Updated       | 0.7214     | **0.0201** | 0.1462   |

