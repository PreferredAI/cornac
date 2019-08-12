import cornac
from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit

rating = amazon_toy.load_rating()
sentiment = amazon_toy.load_sentiment()
md = SentimentModality(data=sentiment)

split_data = RatioSplit(data=rating,
                        test_size=0.15,
                        exclude_unknowns=True, verbose=True,
                        sentiment=md, seed=123)

efm = cornac.models.EFM(num_explicit_factors=50, num_latent_factors=50, num_most_cared_aspects=15,
                        rating_scale=5.0, alpha=0.85,
                        lambda_x=1, lambda_y=1, lambda_u=0.01, lambda_h=0.01, lambda_v=0.01,
                        max_iter=100, num_threads=1,
                        trainable=True, verbose=True, seed=123)

rmse = cornac.metrics.RMSE()
ndcg_50 = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()

exp = cornac.Experiment(eval_method=split_data,
                        models=[efm],
                        metrics=[rmse, ndcg_50, auc])
exp.run()
