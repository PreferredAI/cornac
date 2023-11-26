# Datasets

For easy experimentation, Cornac offers access to a number of popular recommendation benchmark datasets. These are listed below along with their basic characteristics, followed by a usage example. In addition to preference feedback, some of these datasets come with item and/or user auxiliary information, which are grouped into three main categories:
- **Text** refers to textual information associated with items or users. The usual format of this data is `(item_id, text)`, or `(user_id, text)`. Concrete examples of such information are item textual descriptions, product reviews, movie plots, and user reviews, just to name a few.
- **Graph**, for items, corresponds to a network where nodes (or vertices) are items, and links (or edges) represent relations among items. This information is typically represented by an adjacency matrix in the sparse triplet format: `(item_id, item_id, weight)`, or simply `(item_id, item_id)` in the case of unweighted edges. Relations between users (e.g., social network) are represented similarly.    
- **Image** consists of visual information paired with either users or items. The common format for this type of auxiliary data is `(object_id, ndarray)`, where `object_id` could be one of `user_id` or `item_id`, the `ndarray` may contain the raw images (pixel intensities), or some visual feature vectors extracted from the images, e.g., using deep neural nets. For instance, the Amazon clothing dataset includes product CNN visual features.

**How to cite.** If you are using one of the datasets listed below in your research, please follow the citation guidelines by the authors (the "source" link below) of each respective dataset.  
<table>
  <tr>
    <th rowspan="2"><br>Dataset</th>
    <th colspan="4">Preference Info.</th>
    <th colspan="3">Item Auxiliary Info.</th>
    <th>User Auxiliary Info.</th>
  </tr>
  <tr>
    <td>#Users</td>
    <td>#Items</td>
    <td>#Interactions</td>
    <td>Type</td>
    <td>Text</td>
    <td>Graph</td>
    <td>Image</td>
    <td align="center">Graph</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_clothing">Amazon Clothing</a><br>(<a href="http://jmcauley.ucsd.edu/data/amazon/">source</a>)</td>
    <td align="right">5,377</td>
    <td align="right">3,393</td>
    <td align="right">13,689</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_digital_music">Amazon Digital Music</a><br>(<a href="http://jmcauley.ucsd.edu/data/amazon/">source</a>)</td>
    <td align="right">5,541</td>
    <td align="right">3,568</td>
    <td align="right">64,706</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_office">Amazon Office</a><br>(<a href="http://jmcauley.ucsd.edu/data/amazon/">source</a>)</td>
    <td align="right">3,703</td>
    <td align="right">6,523</td>
    <td align="right">53,282</td>
    <td align="center">INT<br>[1,5]</td>
    <td></td>
    <td align="center">&#10004;</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_toy">Amazon Toy</a><br>(<a href="http://jmcauley.ucsd.edu/data/amazon/">source</a>)</td>
    <td align="right">19,412</td>
    <td align="right">11,924</td>
    <td align="right">167,597</td>
    <td align="center">INT<br>[1,5]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.citeulike">Citeulike</a><br>(<a href="http://www.wanghao.in/CDL.htm">source</a>)</td>
    <td align="right">5,551</td>
    <td align="right">16,980</td>
    <td align="right">210,537</td>
    <td align="center">BIN<br>{0,1}</td>
    <td align="center">&#10004;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.epinions">Epinions</a><br>(<a href="http://www.trustlet.org/downloaded_epinions.html">source</a>)</td>
    <td align="right">40,163</td>
    <td align="right">139,738</td>
    <td align="right">664,824</td>
    <td align="center">INT<br>[1,5]</td>
    <td></td>
    <td></td>
    <td></td>
    <td align="center">&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.filmtrust">FilmTrust</a><br>(<a href="https://www.librec.net/datasets.html">source</a>)</td>
    <td align="right">1,508</td>
    <td align="right">2,071</td>
    <td align="right">35,497</td>
    <td align="center">REAL<br>[0.5,4]</td>
    <td></td>
    <td></td>
    <td></td>
    <td align="center">&#10004;</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens">MovieLens 100k</a><br>(<a href="https://grouplens.org/datasets/movielens/">source</a>)</td>
    <td align="right">943</td>
    <td align="right">1,682</td>
    <td align="right">100,000</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens">MovieLens 1M</a><br>(<a href="https://grouplens.org/datasets/movielens/">source</a>)</td>
    <td align="right">6,040</td>
    <td align="right">3,706</td>
    <td align="right">1,000,209</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens">MovieLens 10M</a><br>(<a href="https://grouplens.org/datasets/movielens/">source</a>)</td>
    <td align="right">69,878</td>
    <td align="right">10,677</td>
    <td align="right">10,000,054</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens">MovieLens 20M</a><br>(<a href="https://grouplens.org/datasets/movielens/">source</a>)</td>
    <td align="right">138,493</td>
    <td align="right">26,744</td>
    <td align="right">20,000,263</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.netflix">Netflix Small</a><br>(<a href="https://www.kaggle.com/netflix-inc/netflix-prize-data/">source</a>)</td>
    <td align="right">10,000</td>
    <td align="right">5,000</td>
    <td align="right">607,803</td>
    <td align="center">INT<br>[1,5]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.netflix">Neflix Original</a><br>(<a href="https://www.kaggle.com/netflix-inc/netflix-prize-data/">source</a>)</td>
    <td align="right">480,189</td>
    <td align="right">17,770</td>
    <td align="right">100,480,507</td>
    <td align="center">INT<br>[1,5]</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.tradesy">Tradesy</a><br>(<a href="http://jmcauley.ucsd.edu/data/tradesy/">source</a>)</td>
    <td align="right">19,243</td>
    <td align="right">165,906</td>
    <td align="right">394,421</td>
    <td align="center">BIN<br>{0,1}</td>
    <td></td>
    <td></td>
    <td align="center">&#10004;</td>
    <td></td>
  </tr>
</table>

## Usage example

Assume that we are interested in the FilmTrust dataset, which comes with both `user-item ratings` and `user-user trust` information. We can load these two pieces of information as follows,
```Python
from cornac.datasets import filmtrust

ratings = filmtrust.load_feedback()
trust = filmtrust.load_trust()
```

The ranting values are in the range `[0.5,4]`, and the trust network is undirected. Here are samples from our dataset,
```
Samples from ratings: [('1', '1', 2.0), ('1', '2', 4.0), ('1', '3', 3.5)] 
Samples from trust: [('2', '966', 1.0), ('2', '104', 1.0), ('5', '1509', 1.0)]
```
Our dataset is now ready to use for model training and evaluation. A concrete example is [sorec_filmtrust](../../examples/sorec_filmtrust.py), which illustrates how to perform an experiment with the [SoRec](../models/sorec/) model on FilmTrust. More details regarding the other datasets are available in the [documentation](https://cornac.readthedocs.io/en/latest/datasets.html).

---

## Next-Basket Datasets

<table>
  <tr>
    <th rowspan="2"><br>Dataset</th>
    <th colspan="4">Preference Info.</th>
    <th rowspan="2">Extra Info.</th>
  </tr>
  <tr>
    <td>#Users</td>
    <td>#Items</td>
    <td>#Baskets</td>
    <td>#Interactions</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.tafeng">Ta Feng</a><br>(<a href="https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset/">source</a>)</td>
    <td align="right">28,297</td>
    <td align="right">22,542</td>
    <td align="right">86,403</td>
    <td align="right">817,741</td>
    <td align="center">price, quantity</td>
    <td></td>
  </tr>
</table>