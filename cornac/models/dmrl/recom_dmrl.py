from typing import Dict
from cornac.data.bert_text import BertTextModality
from cornac.models.dmrl.dmrl import DMRLLoss, DMRLModel
from cornac.models.dmrl.pwlearning_sampler import PWLearningSampler
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

    def _fit_dmrl(self, train_set: Dataset):
        """
        Fit the model to observations.

        :param train_set: User-Item preference data as well as additional modalities.
        :return: trained model
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {device} for training")

        self.sampler = PWLearningSampler(train_set, num_neg=self.num_neg)

        model = DMRLModel(self.num_users,
                          self.num_items,
                          self.embedding_dim,
                          self.bert_text_dim,
                          self.num_neg).to(device)

        loss_function = DMRLLoss(decay_c=1e-3, num_factors=self.num_factors, num_neg=self.num_neg)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.decay_r)
        # Create learning rate scheduler
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        dataloader = DataLoader(self.sampler, batch_size=self.batch_size, num_workers=0, shuffle=True, prefetch_factor=None)

        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0
            last_loss = 0
            # Create learning rate scheduler
            scheduler.step()
            batch: torch.Tensor
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                item_text_embeddings = self.get_modality_embeddings(batch)

                # move the data to the device
                batch = batch.to(device)
                item_text_embeddings = item_text_embeddings.to(device)

                # Forward pass
                embedding_factor_lists, rating_scores = model(batch, item_text_embeddings)
                loss = loss_function(embedding_factor_lists, rating_scores)

                # Backward pass and optimize
                loss.backward()
                model.log_gradients()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1)

                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))

                    if self.log_metrics:
                        tb_x = epoch * len(dataloader) + i + 1
                        self.tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                        self.tb_writer.add_scalar('Gradient Norm/train', np.mean(model.grad_norms), tb_x)
                        for name, param in model.named_parameters():
                                self.tb_writer.add_scalar(name + '/grad_norm', np.mean(model.grad_dict[name]), tb_x)
                                self.tb_writer.add_histogram(name + '/grad', param.grad, global_step=epoch)
                        
                        model.reset_grad_metrics()
                        running_loss = 0
            print(f"Epoch: {epoch} is done")

        print("Finished training!")