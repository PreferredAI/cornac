# Models

This directory includes the implementation of all the models supported in Cornac. 
Additional dependencies (CPU versions) and examples for each model are also listed accordingly.

| Model and paper | Additional dependencies | Examples |
| --- | :---: | :---: |
| [Matrix Co-Factorization (MCF)](mcf), [paper](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1113.pdf) | N/A | 
| [Bayesian Personalized Ranking (BPR)](bpr), [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | N/A | [bpr_netflix.py](../../examples/bpr_netflix.py)
| [Collaborative Context Poisson Factorization (C2PF)](c2pf), [paper](https://www.ijcai.org/proceedings/2018/0370.pdf) | N/A | [c2pf_example.py](../../examples/c2pf_example.py)
| [Collaborative Deep Learning (CDL)](cdl), [paper](https://arxiv.org/pdf/1409.2944.pdf) | [requirements.txt](cdl/requirements.txt) | [cdl_example.py](../../examples/cdl_example.py)
| [Collaborative Ordinal Embedding (COE)](coe), [paper](http://www.hadylauw.com/publications/sdm16.pdf) | [requirements.txt](coe/requirements.txt) |
| [Hierarchical Poisson Factorization (HPF)](hpf), [paper](http://jakehofman.com/inprint/poisson_recs.pdf) | N/A |
| [Indexable Bayesian Personalized Ranking (IBPR)](ibpr), [paper](http://www.hadylauw.com/publications/cikm17a.pdf) | [requirements.txt](ibpr/requirements.txt) | [ibpr_example.py](../../examples/ibpr_example.py)
| [Matrix Factorization (MF)](mf), [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | N/A | [biased_mf.py](../../examples/biased_mf.py), [given_data.py](../../examples/given_data.py)
| [Online Indexable Bayesian Personalized Ranking (Online IBPR)](online_ibpr), [paper](http://www.hadylauw.com/publications/cikm17a.pdf) | [requirements.txt](online_ibpr/requirements.txt) |
| [Probabilistic Collaborative Representation Learning (PCRL)](pcrl), [paper](http://www.hadylauw.com/publications/uai18.pdf) | [requirements.txt](pcrl/requirements.txt) | [pcrl_example.py](../../examples/pcrl_example.py)
| [Probabilistic Matrix Factorization (PMF)](pmf), [paper](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf) | N/A | [pmf_ratio.py](../../examples/pmf_ratio.py)
| [Spherical K-means (SKM)](skm), [paper](https://www.sciencedirect.com/science/article/pii/S092523121501509X) | N/A |
| [Visual Bayesian Personalized Ranking (VBPR)](vbpr), [paper](https://arxiv.org/pdf/1510.01784.pdf) | [requirements.txt](vbpr/requirements.txt) | [vbpr_tradesy.py](../../examples/vbpr_tradesy.py)
| [Convolutional Matrix Factorization (ConvMF)](conv_mf), [paper](http://uclab.khu.ac.kr/resources/publication/C_351.pdf) | [requirements.txt](conv_mf/requirements.txt) | [conv_mf_example.py](../../examples/conv_mf_example.py)
