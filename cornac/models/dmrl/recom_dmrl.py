from typing import Dict, Tuple
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
    Disentangled multimodal representation learning
    
    Parameters
    ----------
    name: string, default: 'DMRL'
        The name of the recommender model.
    
    batch_size: int, optional, default: 32
        The number of samples per batch to load.
    
    learning_rate: float, optional, default: 1e-4
        The learning rate for the optimizer.
    
    decay_c: float, optional, default: 1
        The decay for the disentangled loss term in the loss function.

    decay_r: float, optional, default: 0.01
        The decay for the regularization term in the loss function.
    
    epochs: int, optional, default: 10
        The number of epochs to train the model.
    
    embedding_dim: int, optional, default: 100
        The dimension of the embeddings.
    
    bert_text_dim: int, optional, default: 384
        The dimension of the bert text embeddings coming from the huggingface transformer model
    
    num_neg: int, optional, default: 4
        The number of negative samples to use in the training per user per batch (1 positive and num_neg negatives are used)
    
    num_factors: int, optional, default: 4
        The number of factors to use in the model.
    
    trainable: bool, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already trained.
    
    verbose: bool, optional, default: False
        When True, the model prints out more information during training.
    
    modalities_pre_built: bool, optional, default: True
        When True, the model assumes that the modalities are already built and does not build them.
    
    log_metrics: bool, optional, default: False
        When True, the model logs metrics to tensorboard.

    References
    ----------
    * Fan Liu, Huilin Chen,  Zhiyong Cheng, Anan Liu, Liqiang Nie, Mohan Kankanhalli. DMRL: Disentangled Multimodal Representation Learning for
        Recommendation. https://arxiv.org/pdf/2203.05406.pdf.
    """
    def __init__(self, name: str = "DRML", batch_size: int = 32,
                 learning_rate: float = 1e-4, decay_c: float = 1, decay_r: float = 0.01,  epochs: int = 10, embedding_dim: int = 100, bert_text_dim: int = 384,
                 image_dim: int = None, num_neg: int = 4, num_factors: int =4, trainable: bool = True, verbose: bool = False, log_metrics: bool = False):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.learning_rate = learning_rate
        self.decay_c = decay_c
        self.decay_r = decay_r
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.bert_text_dim = bert_text_dim
        self.image_dim = image_dim
        self.num_neg = num_neg
        self.num_factors = num_factors
        self.log_metrics = log_metrics
        if log_metrics:
            self.tb_writer = SummaryWriter("temp/tb_data/run_1")


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

    def get_item_image_embedding(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Get the item image embeddings from the image modality. Expect the image
        modaility to be preencded and available as a numpy array.

        :param batch: user inidices in first column, pos item indices in second
            and all other columns are negative item indices
        """
        shape = batch[:, 1:].shape
        all_items = batch[:, 1:].flatten()
        
        return torch.tensor(self.image_modality.features[all_items, :].reshape((*shape, self.image_modality.feature_dim)), dtype=torch.float32)

    def get_item_text_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Get the item text embeddings from the BERT model. Either by encoding the
        text on the fly or by using the preencoded text.

        :param batch: user inidices in first column, pos item indices in second
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

    def get_modality_embeddings(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the modality embeddings.
        """
        item_text_embeddings = self.get_item_text_embeddings(batch)
        item_image_embeddings = self.get_item_image_embedding(batch)

        return item_text_embeddings, item_image_embeddings

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
        self.bert_text_modality = train_set.item_text
        self.image_modality = train_set.item_image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {self.device} for training")

        self.sampler = PWLearningSampler(train_set, num_neg=self.num_neg)

        self.model = DMRLModel(self.num_users,
                          self.num_items,
                          self.embedding_dim,
                          self.bert_text_dim,
                          self.image_dim,
                          self.num_neg,
                          self.num_factors).to(self.device)

        loss_function = DMRLLoss(decay_c=1e-3, num_factors=self.num_factors, num_neg=self.num_neg)
        # loss_function = torch.nn.MSELoss(reduction="sum")
        neg_to_total_ration = (train_set.csr_matrix.A==0).sum()/train_set.csr_matrix.A.sum()

        # add hyperparams to tensorboard
        self.tb_writer.add_hparams({"learning_rate": self.learning_rate, "decay_c": self.decay_c, "decay_r": self.decay_r, "batch_size": self.batch_size,
                                    "epochs": self.epochs, "embedding_dim": self.embedding_dim, "bert_text_dim": self.bert_text_dim, "num_neg": self.num_neg,
                                    "num_factors": self.num_factors, "Neg_to_total_ratio": neg_to_total_ration}, {})

        # loss_function = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor([neg_to_total_ration]))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay_r, betas=(0.9, 0.999))
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay_r)
        # Create learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5)
        dataloader = DataLoader(self.sampler, batch_size=self.batch_size, num_workers=0, shuffle=True, prefetch_factor=None)
        j = 1
        stop=False
        # Training loop
        for epoch in range(self.epochs):
            if stop:
                break
            running_loss = 0
            last_loss = 0
            i = 0

            batch: torch.Tensor
            # all_batches = []
            for i, batch in enumerate(dataloader):
            # for item_idxs in train_set.item_iter(self.batch_size, shuffle=True):
                # batch_R = train_set.csc_matrix[:, item_idxs]
                # u_batch = torch.arange(batch_R.shape[0]).to(device)
                # i_batch = torch.from_numpy(item_idxs).to(device)
                # r_batch = torch.tensor(batch_R.A, dtype=torch.float).to(device)
                # text = self.bert_text_modality.encoded_corpus[i_batch]
                # all_batches.append(batch)
                optimizer.zero_grad()


                item_text_embeddings, item_image_embeddings = self.get_modality_embeddings(batch)

                # move the data to the device
                batch = batch.to(self.device)
                item_text_embeddings = item_text_embeddings.to(self.device)
                item_image_embeddings = item_image_embeddings.to(self.device)

                # Forward pass
                embedding_factor_lists, rating_scores = self.model(batch, item_text_embeddings, item_image_embeddings)
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
        Evaluate the models training set performance using Recall 300 metric.
        """
        from cornac.eval_methods.base_method import ranking_eval

        print("Evaluating training set performance")
        avg_results, user_results = ranking_eval(self, [Recall(k=300)], self.train_set, self.train_set, verbose=True) 

        print(avg_results)

        # user_recall = []
        # # first calculate Recall@k
        # for user in range(self.train_set.csr_matrix.shape[0]//2):
        #     pos_items = self.train_set.csr_matrix[user, :].nonzero()[1]
        #     all_items = torch.tensor(range(0, self.train_set.num_items), dtype=torch.long)
        #     item_scores = self.score(user, all_items)
        #     ranked_items = np.array(all_items)[item_scores.argsort()[::-1]]

        #     user_recall.append(Recall(k=300).compute(pos_items, ranked_items))
        
        # user_recall_mean = np.mean(user_recall)
        return avg_results


    def score(self, user_index: int, item_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Scores a user-item pair. If item_index is None, scores for all known
        items.       

        Parameters
        ----------
        name: user_idx
            The index of the user for whom to perform score prediction.
    
        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.
        
        return: tensor containing predictions for the user-item pairs
        """
        self.model.num_neg = 0

        if item_indices is None:
            item_indices = torch.tensor(list(self.iid_map.values()), dtype=torch.long)
        
        user_index = user_index * torch.ones(len(item_indices), dtype=torch.long)

        if not hasattr(self.bert_text_modality, "encoded_corpus"):
            self.bert_text_modality.preencode_entire_corpus()
        
        # since the model expects as (batch size, 1 + num_neg, encoding dim) we just add one dim and repeat
        encoded_text = self.bert_text_modality.encoded_corpus[item_indices,:]
        encoded_text = encoded_text[:, None, :]

        encoded_image = torch.tensor(self.image_modality.features[item_indices, :], dtype=torch.float32)
        encoded_image = encoded_image[:, None, :]

        input_tensor = torch.stack((user_index, item_indices), axis=1)
        input_tensor = input_tensor.to(self.device)
        encoded_text = encoded_text.to(self.device)
        encoded_image = encoded_image.to(self.device)

        with torch.no_grad():
            _, ratings_sum_over_mods = self.model(input_tensor, encoded_text, encoded_image)

        return np.array(ratings_sum_over_mods[:, 0].detach().cpu())
    
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

