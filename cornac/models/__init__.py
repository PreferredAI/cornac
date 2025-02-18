# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from .recommender import Recommender
from .recommender import NextBasketRecommender
from .recommender import NextItemRecommender

from .amr import AMR
from .ann import AnnoyANN
from .ann import FaissANN
from .ann import HNSWLibANN
from .ann import ScaNNANN
from .baseline_only import BaselineOnly
from .beacon import Beacon
from .bivaecf import BiVAECF
from .bpr import BPR
from .bpr import WBPR
from .causalrec import CausalRec
from .c2pf import C2PF
from .cdl import CDL
from .cdr import CDR
from .coe import COE
from .companion import Companion
from .comparer import ComparERObj
from .comparer import ComparERSub
from .conv_mf import ConvMF
from .ctr import CTR
from .cvae import CVAE
from .cvaecf import CVAECF
from .dmrl import DMRL
from .dnntsp import DNNTSP
from .ease import EASE
from .efm import EFM
from .fm import FM
from .gcmc import GCMC
from .global_avg import GlobalAvg
from .gp_top import GPTop
from .gru4rec import GRU4Rec
from .hft import HFT
from .hpf import HPF
from .hrdr import HRDR
from .hypar import HypAR
from .ibpr import IBPR
from .knn import ItemKNN
from .knn import UserKNN
from .lightgcn import LightGCN
from .lrppm import LRPPM
from .mcf import MCF
from .mf import MF
from .mmmf import MMMF
from .most_pop import MostPop
from .mter import MTER
from .narre import NARRE
from .ncf import GMF
from .ncf import MLP
from .ncf import NeuMF
from .ngcf import NGCF
from .nmf import NMF
from .online_ibpr import OnlineIBPR
from .pcrl import PCRL
from .pmf import PMF
from .recvae import RecVAE
from .sansa import SANSA
from .sbpr import SBPR
from .skm import SKMeans
from .sorec import SoRec
from .spop import SPop
from .svd import SVD
from .tifuknn import TIFUKNN
from .trirank import TriRank
from .upcf import UPCF
from .vaecf import VAECF
from .vbpr import VBPR
from .vmf import VMF
from .wmf import WMF
