# Datasets

For convenience Cornac offers easy access to a number of popular benchmark datasets. These are listed below along with their basic characteristics, followed by a usage example. In addition to preference feedback, some of these datasets come with item and/or user auxiliary information, which are grouped into three main categories:
- **Text**. Refers to texts associated with items or users. The usual format of this data is `(item_id, text)`, or for users `(user_id, text)`. Concrete examples of such information are item textual descriptions, product reviews, movie plots, and user's reviews, just to name a few.
- **Graph**. For items this corresponds to a network where vertices (or nodes) are items, and edges (or links) represent relations among items. This information is typically represented by an adjacency matrix in the sparse format: `(item_id, item_id, weight)`, or simply `(item_id, item_id)` in which case the weights are assumed to be binary, i.e., 1 for observed item-pairs and 0 for missing pairs. Similarly, relations between users may be available (e.g., social network).    
- **Image**. Consists of visual information paired with either users or items. The common format for this type of auxiliary data is `(object_id, array)`, where object stands for user or item, the array may contain pixel intensities, in the case of raw images, or some continuous visual features, e.g., learned using deep neural nets. For instance, the Amazon clothing dataset includes product CNN features.    
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_clothing">Amazon Clothing</a><br><a href="http://jmcauley.ucsd.edu/data/amazon/"> source </a></td>
    <td align="right">5,377</td>
    <td align="right">3,393</td>
    <td align="right">13, 689</td>
    <td align="center">INT<br>[1,5]</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_office">Amazon Office</a><br><a href="http://jmcauley.ucsd.edu/data/amazon/"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.amazon_toy">Amazon Toy</a><br><a href="http://jmcauley.ucsd.edu/data/amazon/"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.citeulike">Citeulike</a><br><a href="http://www.wanghao.in/CDL.htm"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.epinions">Epinions</a><br><a href="http://www.trustlet.org/downloaded_epinions.html"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens">MovieLens 100k</a><br><a href="https://grouplens.org/datasets/movielens/"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.movielens">MovieLens 1M</a><br><a href="https://grouplens.org/datasets/movielens/"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.netflix">Netflix Small</a><br><a href="https://www.kaggle.com/netflix-inc/netflix-prize-data/"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.netflix">Neflix Original</a><br><a href="https://www.kaggle.com/netflix-inc/netflix-prize-data/"> source </a></td>
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
    <td><a href="https://cornac.readthedocs.io/en/latest/datasets.html#module-cornac.datasets.tradesy">Tradesy</a><br><a href="http://jmcauley.ucsd.edu/data/tradesy/"> source </a></td>
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