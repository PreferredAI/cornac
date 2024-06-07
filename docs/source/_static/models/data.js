var data = [
    {
        "year": "",
        "name": "Online Indexable Bayesian Personalized Ranking (Online IBPR)",
        "link": "cornac/models/online_ibpr",
        "paper": "http://www.hadylauw.com/publications/cikm17a.pdf",
        "type": "Collaborative Filtering",
        "requirements": "cornac/models/online_ibpr/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/ibpr_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/08_retrieval.ipynb"
    },
    {
        "year": "",
        "name": "Visual Matrix Factorization (VMF)",
        "link": "cornac/models/vmf",
        "paper": "https://dsail.kaist.ac.kr/files/WWW17.pdf",
        "type": "Content-Based / Image",
        "requirements": "cornac/models/vmf/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/vmf_clothing.py"
    },
    {
        "year": "2016",
        "name": "Collaborative Deep Ranking (CDR)",
        "link": "cornac/models/cdr",
        "paper": "http://inpluslab.com/chenliang/homepagefiles/paper/hao-pakdd2016.pdf",
        "type": "Content-Based / Text",
        "requirements": "cornac/models/cdr/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/cdr_example.py"
    },
    {
        "year": "",
        "name": "Collaborative Ordinal Embedding (COE)",
        "link": "cornac/models/coe",
        "paper": "http://www.hadylauw.com/publications/sdm16.pdf",
        "type": "Collaborative Filtering",
        "requirements": "cornac/models/coe/requirements.txt",
        "platform": "CPU / GPU"
    },
    {
        "year": "",
        "name": "Convolutional Matrix Factorization (ConvMF)",
        "link": "cornac/models/conv_mf",
        "paper": "http://uclab.khu.ac.kr/resources/publication/C_351.pdf",
        "type": "Content-Based / Text",
        "requirements": "cornac/models/conv_mf/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/conv_mf_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/09_deep_learning.ipynb"
    },
    {
        "year": "",
        "name": "Learning to Rank Features for Recommendation over Multiple Categories (LRPPM)",
        "link": "cornac/models/lrppm",
        "paper": "https://www.yongfeng.me/attach/sigir16-chen.pdf",
        "type": "Explainable",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/lrppm_example.py"
    },
    {
        "year": "",
        "name": "Session-based Recommendations With Recurrent Neural Networks (GRU4Rec)",
        "link": "cornac/models/gru4rec",
        "paper": "https://arxiv.org/pdf/1511.06939.pdf",
        "type": "Next-Item",
        "requirements": "cornac/models/gru4rec/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/gru4rec_yoochoose.py"
    },
    {
        "year": "",
        "name": "Spherical K-means (SKM)",
        "link": "cornac/models/skm",
        "paper": "https://www.sciencedirect.com/science/article/pii/S092523121501509X",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/skm_movielens.py"
    },
    {
        "year": "",
        "name": "Visual Bayesian Personalized Ranking (VBPR)",
        "link": "cornac/models/vbpr",
        "paper": "https://arxiv.org/pdf/1510.01784.pdf",
        "type": "Content-Based / Image",
        "requirements": "cornac/models/vbpr/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/vbpr_tradesy.py",
        "cross-modality": "tutorials/vbpr_text.ipynb",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/05_multimodality.ipynb"
    },
    {
        "year": "2015",
        "name": "Collaborative Deep Learning (CDL)",
        "link": "cornac/models/cdl",
        "paper": "https://arxiv.org/pdf/1409.2944.pdf",
        "type": "Content-Based / Text",
        "requirements": "cornac/models/cdl/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/cdl_example.py",
        "deep-dive": "https://github.com/lgabs/cornac/blob/luan/describe-gpu-supported-models-readme/tutorials/working_with_auxiliary_data.md"
    },
    {
        "year": "",
        "name": "Hierarchical Poisson Factorization (HPF)",
        "link": "cornac/models/hpf",
        "paper": "http://jakehofman.com/inprint/poisson_recs.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/hpf_movielens.py"
    },
    {
        "year": "",
        "name": "TriRank: Review-aware Explainable Recommendation by Modeling Aspects",
        "link": "cornac/models/trirank",
        "paper": "https://wing.comp.nus.edu.sg/wp-content/uploads/Publications/PDF/TriRank-%20Review-aware%20Explainable%20Recommendation%20by%20Modeling%20Aspects.pdf",
        "type": "Explainable",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/trirank_example.py"
    },
    {
        "year": "2014",
        "name": "Explicit Factor Model (EFM)",
        "link": "cornac/models/efm",
        "paper": "https://www.yongfeng.me/attach/efm-zhang.pdf",
        "type": "Explainable",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/efm_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/07_explanations.ipynb"
    },
    {
        "year": "",
        "name": "Social Bayesian Personalized Ranking (SBPR)",
        "link": "cornac/models/sbpr",
        "paper": "https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm14.pdf",
        "type": "Content-Based / Social",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/sbpr_epinions.py"
    },
    {
        "year": "2013",
        "name": "Hidden Factors and Hidden Topics (HFT)",
        "link": "cornac/models/hft",
        "paper": "https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf",
        "type": "Content-Based / Text",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/hft_example.py"
    },
    {
        "year": "2012",
        "name": "Weighted Bayesian Personalized Ranking (WBPR)",
        "link": "cornac/models/bpr",
        "paper": "http://proceedings.mlr.press/v18/gantner12a/gantner12a.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/bpr_netflix.py"
    },
    {
        "year": "2011",
        "name": "Collaborative Topic Regression (CTR)",
        "link": "cornac/models/ctr",
        "paper": "http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf",
        "type": "Content-Based / Text",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/ctr_example_citeulike.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/05_multimodality.ipynb"
    },
    {
        "year": "Earlier",
        "name": "Baseline Only",
        "link": "cornac/models/baseline_only",
        "paper": "http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf",
        "type": "Baseline",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/svd_example.py"
    },
    {
        "year": "",
        "name": "Bayesian Personalized Ranking (BPR)",
        "link": "cornac/models/bpr",
        "paper": "https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/bpr_netflix.py",
        "deep-dive": "https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb"
    },
    {
        "year": "",
        "name": "Factorization Machines (FM)",
        "link": "cornac/models/fm",
        "paper": "https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf",
        "type": "Collaborative Filtering / Content-Based",
        "requirements": "Linux, CPU",
        "platform": "",
        "quick-start": "examples/fm_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/06_contextual_awareness.ipynb"
    },
    {
        "year": "",
        "name": "Global Average (GlobalAvg)",
        "link": "cornac/models/global_avg",
        "paper": "https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf",
        "type": "Baseline",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/biased_mf.py"
    },
    {
        "year": "",
        "name": "Global Personalized Top Frequent (GPTop)",
        "link": "cornac/models/gp_top",
        "paper": "https://dl.acm.org/doi/pdf/10.1145/3587153",
        "type": "Next-Basket",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/gp_top_tafeng.py"
    },
    {
        "year": "",
        "name": "Item K-Nearest-Neighbors (ItemKNN)",
        "link": "cornac/models/knn",
        "paper": "https://dl.acm.org/doi/pdf/10.1145/371920.372071",
        "type": "Neighborhood-Based",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/knn_movielens.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/02_neighborhood.ipynb"
    },
    {
        "year": "",
        "name": "Matrix Factorization (MF)",
        "link": "cornac/models/mf",
        "paper": "https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU / GPU",
        "quick-start": "examples/biased_mf.py",
        "pre-split-data": "examples/given_data.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/03_matrix_factorization.ipynb"
    },
    {
        "year": "",
        "name": "Maximum Margin Matrix Factorization (MMMF)",
        "link": "cornac/models/mmmf",
        "paper": "https://link.springer.com/content/pdf/10.1007/s10994-008-5073-7.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/mmmf_exp.py"
    },
    {
        "year": "",
        "name": "Most Popular (MostPop)",
        "link": "cornac/models/most_pop",
        "paper": "https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf",
        "type": "Baseline",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/bpr_netflix.py"
    },
    {
        "year": "",
        "name": "Non-negative Matrix Factorization (NMF)",
        "link": "cornac/models/nmf",
        "paper": "http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/nmf_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/03_matrix_factorization.ipynb"
    },
    {
        "year": "",
        "name": "Probabilistic Matrix Factorization (PMF)",
        "link": "cornac/models/pmf",
        "paper": "https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/pmf_ratio.py"
    },
    {
        "year": "",
        "name": "Session Popular (SPop)",
        "link": "cornac/models/spop",
        "paper": "https://arxiv.org/pdf/1511.06939.pdf",
        "type": "Next-Item / Baseline",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/spop_yoochoose.py"
    },
    {
        "year": "",
        "name": "Singular Value Decomposition (SVD)",
        "link": "cornac/models/svd",
        "paper": "https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf",
        "type": "Collaborative Filtering",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/svd_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/03_matrix_factorization.ipynb"
    },
    {
        "year": "",
        "name": "Social Recommendation using PMF (SoRec)",
        "link": "cornac/models/sorec",
        "paper": "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.304.2464&rep=rep1&type=pdf",
        "type": "Content-Based / Social",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/sorec_filmtrust.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/05_multimodality.ipynb"
    },
    {
        "year": "",
        "name": "User K-Nearest-Neighbors (UserKNN)",
        "link": "cornac/models/knn",
        "paper": "https://arxiv.org/pdf/1301.7363.pdf",
        "type": "Neighborhood-Based",
        "requirements": "",
        "platform": "CPU",
        "quick-start": "examples/knn_movielens.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/02_neighborhood.ipynb"
    },
    {
        "year": "",
        "name": "Weighted Matrix Factorization (WMF)",
        "link": "cornac/models/wmf",
        "paper": "http://yifanhu.net/PUB/cf.pdf",
        "type": "Collaborative Filtering",
        "requirements": "cornac/models/wmf/requirements.txt",
        "platform": "CPU / GPU",
        "quick-start": "examples/wmf_example.py",
        "deep-dive": "https://github.com/PreferredAI/tutorials/blob/master/recommender-systems/04_implicit_feedback.ipynb"
    }
];