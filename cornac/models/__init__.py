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

from .amr import AMR
from .baseline_only import BaselineOnly
from .bivaecf import BiVAECF
from .bpr import BPR
from .bpr import WBPR
from .causalrec import CausalRec
from .c2pf import C2PF
from .cdl import CDL
from .cdr import CDR
from .coe import COE
from .comparer import ComparERObj
from .comparer import ComparERSub
from .conv_mf import ConvMF
from .ctr import CTR
from .cvae import CVAE
from .cvaecf import CVAECF
from .efm import EFM
from .global_avg import GlobalAvg
from .hft import HFT
from .hpf import HPF
from .ibpr import IBPR
from .knn import ItemKNN
from .knn import UserKNN
from .mcf import MCF
from .mf import MF
from .mmmf import MMMF
from .most_pop import MostPop
from .mter import MTER
from .narre import NARRE
from .ncf import GMF
from .ncf import MLP
from .ncf import NeuMF
from .nmf import NMF
from .online_ibpr import OnlineIBPR
from .pcrl import PCRL
from .pmf import PMF
from .sbpr import SBPR
from .skm import SKMeans
from .sorec import SoRec
from .svd import SVD
from .vaecf import VAECF
from .vbpr import VBPR
from .vmf import VMF
from .wmf import WMF

try:
    from .fm import FM
except ModuleNotFoundError:
    print(
        "FM model is only supported on Linux.\n"
        + "Windows executable can be found at http://www.libfm.org."
    )

