from typing import Dict
from cornac.data.bert_text import BertTextModality
from cornac.exception import ScoreException
from cornac.metrics.ranking import Recall
from cornac.models.dmrl.dmrl import DMRLLoss, DMRLModel
from cornac.models.dmrl.pwlearning_sampler import PWLearningSampler
from cornac.models.mf.backend_pt import learn
from cornac.models.recommender import Recommender
from cornac.data.dataset import Dataset
import torch
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class DMRL(Recommender):
    """
    Disentangled multimodal representation learning for recomendation.






    References
    ----------
    * Fan Liu, Huilin Chen,  Zhiyong Cheng, Anan Liu, Liqiang Nie, Mohan Kankanhalli. DMRL: Disentangled Multimodal Representation Learning for
        Recommendation. https://arxiv.org/pdf/2203.05406.pdf.
    """
    def __init__(self, iid_map: Dict, num_users: int, num_items: int, bert_text_modality: BertTextModality, name: str = "DRML", batch_size: int = 32,
                 learning_rate: float = 1e-4, decay_c: float = 1, decay_r: float = 0.01,  epochs: int = 10, embedding_dim: int = 100, bert_text_dim: int = 384,
                 num_neg: int = 4, num_factors: int =4, trainable: bool = True, verbose: bool = False, modalities_pre_built: bool = True, log_metrics: bool = False):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.learning_rate = learning_rate
        self.decay_c = decay_c
        self.decay_r = decay_r
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.num_users = num_users
        self.num_items = num_items
        self.iid_map = iid_map
        self.embedding_dim = embedding_dim
        self.bert_text_dim = bert_text_dim
        self.num_neg = num_neg
        self.modalities_pre_built = modalities_pre_built
        self.bert_text_modality = bert_text_modality
        self.num_factors = num_factors
        self.log_metrics = log_metrics
        if log_metrics:
            self.tb_writer = SummaryWriter("temp/tb_data/run_1")

        if not self.modalities_pre_built:
            self.bert_text_modality.build(iid_map)

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            self._fit_dmrl(train_set)

        return self

    def get_item_image_embedding(self, batch: torch.Tensor):
        """
        Get the item image embeddings.
        """
        pass

    def get_item_text_embeddings(self, batch: torch.Tensor):
        """
        Get the item text embeddings.

        :param batch: user inidices in dirst column, pos item indices in second
            and all other columns are negative item indices 
        """
        shape = batch[:, 1:].shape
        all_items = batch[:, 1:].flatten()

        if not self.bert_text_modality.preencoded:
            item_text_embeddings = self.bert_text_modality.batch_encode(all_items)
            item_text_embeddings = item_text_embeddings.reshape((*shape, self.bert_text_modality.output_dim))
        else:
            item_text_embeddings = self.bert_text_modality.encoded_corpus[all_items]
            item_text_embeddings = item_text_embeddings.reshape((*shape, self.bert_text_modality.output_dim))

        return item_text_embeddings

    def get_modality_embeddings(self, batch: torch.Tensor):
        """
        Get the modality embeddings.
        """
        item_text_embeddings = self.get_item_text_embeddings(batch)
        # item_image_embeddings = self.get_item_image_embedding(batch)

        return item_text_embeddings

    # def _fit_dmrl(self, train_set: Dataset):
    #     self.model = DMRLModel(self.num_users,
    #                 self.num_items,
    #                 self.embedding_dim,
    #                 self.bert_text_dim,
    #                 self.num_neg,
    #                 self.num_factors)
    
    #     learn(
    #         model=self.model,
    #         train_set=train_set,
    #         n_epochs=self.epochs,
    #         batch_size=self.batch_size,
    #         learning_rate=self.learning_rate,
    #         reg=0.02,
    #         optimizer="sgd"
    #     )

    def _fit_dmrl(self, train_set: Dataset):
        """
        Fit the model to observations.

        :param train_set: User-Item preference data as well as additional modalities.
        :return: trained model
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {device} for training")

        self.sampler = PWLearningSampler(train_set, num_neg=self.num_neg)

        self.model = DMRLModel(self.num_users,
                          self.num_items,
                          self.embedding_dim,
                          self.bert_text_dim,
                          self.num_neg,
                          self.num_factors).to(device)

        loss_function = DMRLLoss(decay_c=1e-3, num_factors=self.num_factors, num_neg=self.num_neg)
        # loss_function = torch.nn.MSELoss(reduction="sum")
        neg_to_total_ration = (train_set.csr_matrix.A==0).sum()/train_set.csr_matrix.A.sum()

        # add hyperparams to tensorboard
        self.tb_writer.add_hparams({"learning_rate": self.learning_rate, "decay_c": self.decay_c, "decay_r": self.decay_r, "batch_size": self.batch_size,
                                    "epochs": self.epochs, "embedding_dim": self.embedding_dim, "bert_text_dim": self.bert_text_dim, "num_neg": self.num_neg,
                                    "num_factors": self.num_factors, "Neg_to_total_ratio": neg_to_total_ration}, {})

        # loss_function = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor([neg_to_total_ration]))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay_r)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay_r)
        # Create learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=15)
        dataloader = DataLoader(self.sampler, batch_size=self.batch_size, num_workers=0, shuffle=True, prefetch_factor=None)
        j = 1
        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0
            last_loss = 0
            i = 0

            batch: torch.Tensor
            all_batches = []
            for i, batch in enumerate(dataloader):
            # for item_idxs in train_set.item_iter(self.batch_size, shuffle=True):
                # batch_R = train_set.csc_matrix[:, item_idxs]
                # u_batch = torch.arange(batch_R.shape[0]).to(device)
                # i_batch = torch.from_numpy(item_idxs).to(device)
                # r_batch = torch.tensor(batch_R.A, dtype=torch.float).to(device)
                # text = self.bert_text_modality.encoded_corpus[i_batch]
                all_batches.append(batch)
                optimizer.zero_grad()


                item_text_embeddings = self.get_modality_embeddings(batch)

                # move the data to the device
                batch = batch.to(device)
                item_text_embeddings = item_text_embeddings.to(device)

                # Forward pass
                embedding_factor_lists, rating_scores = self.model(batch, item_text_embeddings)
                # preds = self.model(u_batch, i_batch, text)
                loss = loss_function(embedding_factor_lists, rating_scores)
                # loss = loss_function(preds, r_batch)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                if self.log_metrics:
                    self.model.log_gradients_and_weights()

                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                devider = 10
                if i % devider == 9:
                    last_loss = running_loss / devider # loss per batch
                    # last_loss = running_loss / (i + 1)
                    print('  batch {} loss: {}'.format(i + 1, last_loss))

                    if self.log_metrics:
                        # tb_x = epoch * len(dataloader) + i + 1

                        tb_x = j

                        self.tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                        self.tb_writer.add_scalar('Gradient Norm/train', np.mean(self.model.grad_norms), tb_x)
                        self.tb_writer.add_scalar('Param Norm/train', np.mean(self.model.param_norms), tb_x)
                        self.tb_writer.add_scalar('User-Item based rating', np.mean(self.model.ui_ratings), tb_x)
                        self.tb_writer.add_scalar('User-Text based rating', np.mean(self.model.ut_ratings), tb_x)
                        self.tb_writer.add_scalar('User-Itm Attention', np.mean(self.model.ui_attention), tb_x)
                        self.tb_writer.add_scalar('User-Text Attention', np.mean(self.model.ut_attention), tb_x)
                        for name, param in self.model.named_parameters():
                                self.tb_writer.add_scalar(name + '/grad_norm', np.mean(self.model.grad_dict[name]), tb_x)
                                self.tb_writer.add_histogram(name + '/grad', param.grad, global_step=epoch)
                        self.tb_writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], tb_x)
                        self.model.reset_grad_metrics()
                    running_loss = 0
                    
                # if i % 999== 0:
                    # scheduler.step()
                
                i +=1
                j +=1
                    
            print(f"Epoch: {epoch} is done")
            scheduler.step()

        print(f"Mean train set recall: {self.eval_train_set_performance()}")
        print("Finished training!")
    
    def eval_train_set_performance(self):
        """
        Save the performance of the model.
        """
        print("Evaluating training set performance")
        user_recall = []
        # first calculate Recall@k
        for user in range(self.train_set.csr_matrix.shape[0]):
            pos_items = self.train_set.csr_matrix[user, :].nonzero()[1]
            all_items = torch.tensor(range(0, self.train_set.num_items), dtype=torch.long)
            item_scores = self.score(user, all_items)
            ranked_items = np.array(all_items)[item_scores.argsort()[::-1]]

            user_recall.append(Recall(k=300).compute(pos_items, ranked_items))
        
        user_recall_mean = np.mean(user_recall)
        return user_recall_mean


    def score(self, user_index: int, item_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Scores a user-item pair. If item_index is None, scores for all known items.       

        :param user_idx: int, required
            The index of the user for whom to perform score prediction.

        :param item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.
        """
        self.model.num_neg = 0

        if item_indices is None:
            item_indices = torch.tensor(list(self.iid_map.values()), dtype=torch.long)
        
        user_index = user_index * torch.ones(len(item_indices), dtype=torch.long)

        if not hasattr(self.bert_text_modality, "encoded_corpus"):
            self.bert_text_modality.preencode_entire_corpus()
        
        # since the model expects as (batch size, 1 + num_neg, encoding dim) we just add one dim and repeat
        encoded_corpus = self.bert_text_modality.encoded_corpus[item_indices,:]
        encoded_corpus = encoded_corpus[:, None, :]
        
        input_tensor = torch.stack((user_index, item_indices), axis=1)

        with torch.no_grad():
            _, ratings_sum_over_mods = self.model(input_tensor, encoded_corpus)

        return np.array(ratings_sum_over_mods[:, 0].detach())
    
    # def score(self, user_idx, item_idx=None):
    #     """Predict the scores/ratings of a user for an item.

    #     Parameters
    #     ----------
    #     user_idx: int, required
    #         The index of the user for whom to perform score prediction.

    #     item_idx: int, optional, default: None
    #         The index of the item for which to perform score prediction.
    #         If None, scores for all known items will be returned.

    #     Returns
    #     -------
    #     res : A scalar or a Numpy array
    #         Relative scores that the user gives to the item or to all known items

    #     """
    #     if item_idx is None:
    #         item_idx = torch.arange(self.num_items)
    #     # text=self.bert_text_modality.encoded_corpus[item_idx]
    #     text = 1
    #     return np.array(self.model(torch.tensor(user_idx), item_idx, text).detach())

