import cornac
from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit

rating = amazon_toy.load_rating()
sentiment = amazon_toy.load_sentiment()
sentiment_module = SentimentModality(data=sentiment)

split_data = RatioSplit(data=rating,
                         test_size=0.15,
                         exclude_unknowns=True, verbose=True,
                         sentiment=sentiment_module, seed=123)

efm = cornac.models.EFM(verbose=True,
                        alpha=0.85,
                        max_iter=100, seed=123)

rmse = cornac.metrics.RMSE()
ndcg_50 = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()

exp = cornac.Experiment(eval_method=split_data,
                        models=[efm],
                        metrics=[rmse, ndcg_50, auc])
exp.run()
