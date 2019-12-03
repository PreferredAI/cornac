# Datasets

For convenience Cornac offers easy access to a number of popular benchmark datasets. These are listed below along with their basic characteristics, followed by a usage example. In addition to preference feedback, some of these datasets come with item and/or user auxiliary information, which are grouped into three main categories:
- **Text**. Refers to texts associated with items or users. The usual format of this data is `(item_id, text)`, or for users `(user_id, text)`. Concrete examples of such information are item textual descriptions, product reviews, movie plots, and user's reviews, just to name a few.
- **Graph**. For items this corresponds to a network where nodes (or vertices) are items, and links (or edges) represent relations among items. This information is typically represented by an adjacency matrix in the sparse format: `(item_id, item_id, weight)`, or simply `(item_id, item_id)` in which case the weights are assumed to be binary, i.e., 1 for observed item-pairs and 0 for missing pairs. Similarly, relations between users may be available (e.g., social network).    
- **Image**. Consists of visual information paired with either users or items. The common format for this type of auxiliary data is `(object_id, array)`, where object stands for user or item, the array may contain pixel intensities, in the case of raw images, or some continuous visual features, e.g., learned using deep neural nets. For instance, the Amazon clothing dataset includes product CNN features.

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

## Usage example

Assume that we are interested in the FilmTrust dataset, which comes with both user-item ratings and user-user trust information. We can load these two pieces of information as follows,
```Python
from cornac.datasets import filmtrust

ratings = filmtrust.load_feedback()
trust = filmtrust.load_trust()
```

The rantings are in the range `[0.5,4]`, and the trust network is undirected with binary weight, i.e., `1` for observed user-user pairs, and `0` for missing ones. Here are samples, the first three lines of `rantings` and `trust`, from our dataset,
```Python
Sample of ratings: [('1', '1', 2.0), ('1', '2', 4.0), ('1', '3', 3.5)] 
Sample of trust relations: [('2', '966', 1.0), ('2', '104', 1.0), ('5', '1509', 1.0)]
```
We can now start using our dataset. The [sorec_filmtrust](../../examples/sorec_filmtrust.py) example illustrates how to perform an experiment to evaluate the [SoRec](../models/sorec/) model on FilmTrust. Useful details on how to load the different dataset are available in the [documentation](https://cornac.readthedocs.io/en/latest/datasets.html).

