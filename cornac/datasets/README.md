# Datasets

For easy experimentation, Cornac offers access to a number of popular recommendation benchmark datasets. These are listed below along with their basic characteristics, followed by a usage example. In addition to preference feedback, some of these datasets come with item and/or user auxiliary information, which are grouped into three main categories:
- **Text** refers to textual information associated with items or users. The usual format of this data is `(item_id, text)`, or `(user_id, text)`. Concrete examples of such information are item textual descriptions, product reviews, movie plots, and user reviews, just to name a few.
- **Graph**, for items, corresponds to a network where nodes (or vertices) are items, and links (or edges) represent relations among items. This information is typically represented by an adjacency matrix in the sparse triplet format: `(item_id, item_id, weight)`, or simply `(item_id, item_id)` in the case of unweighted edges. Relations between users (e.g., social network) are represented similarly.
- **Image** consists of visual information paired with either users or items. The common format for this type of auxiliary data is `(object_id, ndarray)`, where `object_id` could be one of `user_id` or `item_id`, the `ndarray` may contain the raw images (pixel intensities), or some visual feature vectors extracted from the images, e.g., using deep neural nets. For instance, the Amazon clothing dataset includes product CNN visual features.

**How to cite.** If you are using one of the datasets listed below in your research, please follow the citation guidelines by the authors (the "source" link below) of each respective dataset.
<table>
  <tr>
    <th rowspan="2">Dataset</th>
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
    <th rowspan="2">Dataset</th>
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
  </tr>
</table>

---

## Next-Item Datasets

<table>
  <tr>
    <th>Dataset</th>
    <th>Users</th>
    <th>#Items</th>
    <th>#Sessions</th>
    <th>#Interactions</th>
    <th>Extra Info.</th>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.gowalla">Gowalla</a><br>(<a href="https://snap.stanford.edu/data/loc-gowalla.html">source</a>)</td>
    <td align="center">107,092</td>
    <td align="right">1,280,969</td>
    <td align="right">2,710,119</td>
    <td align="right">6,442,892</td>
    <td align="center">Check-ins location (longitude, latitude)</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.yoochoose">YooChoose (buy)</a><br>(<a href="https://2015.recsyschallenge.com/">source</a>)</td>
    <td align="center">N/A</td>
    <td align="right">19,949</td>
    <td align="right">509,696</td>
    <td align="right">1,150,753</td>
    <td align="center">N/A</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.yoochoose">YooChoose (click)</a></td>
    <td align="center">N/A</td>
    <td align="right">52,739</td>
    <td align="right">9,249,729</td>
    <td align="right">33,003,944</td>
    <td align="center">N/A</td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.yoochoose">YooChoose (test)</a></td>
    <td align="center">N/A</td>
    <td align="right">42,155</td>
    <td align="right">2,312,432</td>
    <td align="right">8,251,791</td>
    <td align="center">N/A</td>
  </tr>
</table>

---

## Session-aware Datasets
Session-aware recommendation extends next-item (session-based) recommendation by associating sessions with identified users. While next-item datasets rely on session-level sequences (e.g., `SIT` format), session-aware datasets incorporate user identities (e.g., `USIT` format), allowing models to capture both long-term user preferences across multiple sessions and short-term session-level dynamics.

| Dataset                           | #Users | #Items | #Sessions | #Interactions | #Sessions per User | #Interactions per Item | #Interactions per Session | Density |
| :-------------------------------- | -----: | -----: | --------: | ------------: | -----------------: | ---------------------: | ------------------------: | ------: |
| [Diginetica](./diginetica.py)     |    571 |  6,008 |     2,670 |        12,146 |               4.68 |                   2.02 |                      4.55 |  0.354% |
| [RetailRocket](./retailrocket.py) |  4,249 | 36,658 |    24,732 |       230,817 |               5.82 |                   6.30 |                      9.33 |  0.148% |
| [Cosmetics](./cosmetics.py)       | 17,268 | 42,367 |   172,242 |     2,533,262 |               9.97 |                  59.79 |                     14.71 |  0.346% |

For session-based (next-item) evaluation, [Diginetica](./diginetica.py)'s `load_val()` and `load_test()` default to `mode="session-based"`, returning each user's single held-out session (`val_sbr`/`test_sbr`) with no training transitions repeated — the clean evaluation set used by session-based models such as [FPMC](../models/fpmc/) and [GRU4Rec](../models/gru4rec/). Pass `mode="session-aware"` to load the cumulative files (`val`/`test`) instead, where each user's prior sessions precede their held-out one for cross-session models.

---

## Semantic-ID Datasets
### Amazon Product Review
[Amazon Product Review](./amazon_review.py) 5-core

Each user's reviews form one chronologically-ordered sequence. Interactions are loaded via `amazon_review.load_feedback(category=...)` in `UIRT` format (user, item, rating, timestamp). No preprocessing is needed and the data is kept as-is for comparability with published results (with `leave-last-out` split).

| Dataset                  | #Users | #Items | #Interactions | Type      |
| :----------------------- | -----: | -----: | ------------: | :-------- |
| Amazon Beauty (`beauty`) | 22,363 | 12,101 |       198,502 | INT [1,5] |
| Amazon Sports (`sports`) | 35,598 | 18,357 |       296,337 | INT [1,5] |
| Amazon Toys (`toys`)     | 19,412 | 11,924 |       167,597 | INT [1,5] |

Item content text (title, price, brand, categories -- the features embedded with Sentence-T5 in the TIGER paper) is available via `amazon_review.load_text(category=...)`, built from the public product metadata and covering exactly the 5-core items. Passing `include_description=True` appends each item's product description to its text (cached separately); Paischer et al. (arXiv:2412.08604) found this beneficial for the Toys dataset, while attribute-only text works better for Beauty/Sports.

```Python
from cornac.data import FeatureModality
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation

data = amazon_review.load_feedback(category="beauty")  # UIRT tuples, chronological per user
texts, item_ids = amazon_review.load_text(category="beauty")  # item content text, aligned ids
features = ...  # embed texts, e.g. SentenceTransformer("sentence-t5-base").encode(texts)
eval_method = NextItemEvaluation.leave_last_out(
    data, fmt="UIRT", item_feature=FeatureModality(features=features, ids=item_ids)
)
```
