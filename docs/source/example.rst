First example
==============

This example will show you how to run your very first experiment using Cornac.
::

    import cornac as cn

    # Load MovieLens 100K dataset
    ml_100k = cn.datasets.movielens.load_100k()

    # Split data based on ratio
    ratio_split = cn.eval_methods.RatioSplit(data=ml_100k,
                                             test_size=0.2,
                                             rating_threshold=4.0,
                                             seed=123)

    # Here we are comparing: Biased MF, PMF, and BPR
    mf = cn.models.MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True)
    pmf = cn.models.PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)
    bpr = cn.models.BPR(k=10, max_iter=200, learning_rate=0.01, lambda_reg=0.01)

    # Define metrics used to evaluate the models
    mae = cn.metrics.MAE()
    rmse = cn.metrics.RMSE()
    rec_20 = cn.metrics.Recall(k=20)
    ndcg_20 = cn.metrics.NDCG(k=20)
    auc = cn.metrics.AUC()

    # Put it together into an experiment and run
    exp = cn.Experiment(eval_method=ratio_split,
                        models=[mf, pmf, bpr],
                        metrics=[mae, rmse, rec_20, ndcg_20, auc],
                        user_based=True)
    exp.run()


Output:

+-------+--------+--------+------------+----------+--------+-----------+----------+
|       |    MAE |   RMSE | Recall\@20 | NDCG\@20 |    AUC | Train (s) | Test (s) |
+=======+========+========+============+==========+========+===========+==========+
|**MF** | 0.7441 | 0.9007 |     0.0622 |   0.0534 | 0.2952 |    0.0791 |   1.3119 |
+-------+--------+--------+------------+----------+--------+-----------+----------+
|**PMF**| 0.7490 | 0.9093 |     0.0831 |   0.0683 | 0.4660 |    8.7645 |   2.1569 |
+-------+--------+--------+------------+----------+--------+-----------+----------+
|**BPR**| N/A    | N/A    |     0.0744 |   0.0657 | 0.5932 |    2.4791 |   1.2956 |
+-------+--------+--------+------------+----------+--------+-----------+----------+