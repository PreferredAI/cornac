import torch
import torch.nn as nn


optimizer_dict = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}

activation_functions = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "leakyrelu": nn.LeakyReLU(),
}


class GMF_torch(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
        use_pretrain: bool = False,
        use_NeuMF: bool = False,
        pretrained_GMF=None,
    ):
        super(GMF_torch, self).__init__()

        self.pretrained_GMF = pretrained_GMF
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF

        self.pretrained_GMF = pretrained_GMF

        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)

        self.predict_layer = nn.Linear(num_factors, 1)
        self.Sigmoid = nn.Sigmoid()

        if use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight, std=1e-2)
            nn.init.normal_(self.item_embedding.weight, std=1e-2)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight, std=1e-2)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(self.pretrained_GMF.user_embedding.weight)
        self.item_embedding.weight.data.copy_(self.pretrained_GMF.item_embedding.weight)

    def forward(self, users, items):
        embedding_elementwise = self.user_embedding(users) * self.item_embedding(items)
        if not self.use_NeuMF:
            output = self.predict_layer(embedding_elementwise)
            output = self.Sigmoid(output)
            output = output.view(-1)
        else:
            output = embedding_elementwise

        return output

    def predict(self, users, items):
        with torch.no_grad():
            preds = (self.user_embedding(users) * self.item_embedding(items)).sum(
                dim=1, keepdim=True
            )
        return preds.squeeze()


class MLP_torch(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = None,
        layers=None,
        act_fn="relu",
        use_pretrain: bool = False,
        use_NeuMF: bool = False,
        pretrained_MLP=None,
    ):
        super(MLP_torch, self).__init__()

        if layers is None:
            layers = [64, 32, 16, 8]
        if num_factors is None:
            num_factors = layers[-1]

        assert layers[-1] == num_factors

        self.pretrained_MLP = pretrained_MLP
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)
        self.use_NeuMF = use_NeuMF
        MLP_layers = []

        for idx, factor in enumerate(layers[:-1]):
            MLP_layers.append(nn.Linear(factor, layers[idx + 1]))
            MLP_layers.append(activation_functions[act_fn.lower()])

        # unpacking layers in to torch.nn.Sequential
        self.MLP_model = nn.Sequential(*MLP_layers)

        self.predict_layer = nn.Linear(num_factors, 1)
        self.Sigmoid = nn.Sigmoid()

        if self.use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight, std=1e-2)
            nn.init.normal_(self.item_embedding.weight, std=1e-2)
            for layer in self.MLP_model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight, std=1e-2)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(self.pretrained_MLP.user_embedding.weight)
        self.item_embedding.weight.data.copy_(self.pretrained_MLP.item_embedding.weight)
        for layer, pretrained_layer in zip(
            self.MLP_model, self.pretrained_MLP.MLP_model
        ):
            if isinstance(layer, nn.Linear) and isinstance(pretrained_layer, nn.Linear):
                layer.weight.data.copy_(pretrained_layer.weight)
                layer.bias.data.copy_(pretrained_layer.bias)

    def forward(self, user, item):
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_input = torch.cat((embed_user, embed_item), dim=-1)
        output = self.MLP_model(embed_input)

        if not self.use_NeuMF:
            output = self.predict_layer(output)
            output = self.Sigmoid(output)
            output = output.view(-1)

        return output

    def predict(self, users, items):
        with torch.no_grad():
            embed_user = self.user_embedding(users)
            if len(users) == 1:
                # replicate user embedding to len(items)
                embed_user = embed_user.repeat(len(items), 1)
            embed_item = self.item_embedding(items)
            embed_input = torch.cat((embed_user, embed_item), dim=-1)
            output = self.MLP_model(embed_input)

            output = self.predict_layer(output)
            output = self.Sigmoid(output)
            output = output.view(-1)
        return output

    def __call__(self, *args):
        return self.forward(*args)


class NeuMF_torch(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = None,
        layers=None,  # layer for MLP
        act_fn="relu",
        use_pretrain: bool = False,
        pretrained_GMF=None,
        pretrained_MLP=None,
    ):
        super(NeuMF_torch, self).__init__()

        self.use_pretrain = use_pretrain
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        # layer for MLP
        if layers is None:
            layers = [64, 32, 16, 8]
        if num_factors is None:
            num_factors = layers[-1]

        assert layers[-1] == num_factors

        self.predict_layer = nn.Linear(num_factors * 2, 1)
        self.Sigmoid = nn.Sigmoid()

        self.GMF = GMF_torch(
            num_users,
            num_items,
            num_factors,
            use_pretrain=use_pretrain,
            use_NeuMF=True,
            pretrained_GMF=self.pretrained_GMF,
        )
        self.MLP = MLP_torch(
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
            layers=layers,
            act_fn=act_fn,
            use_pretrain=use_pretrain,
            use_NeuMF=True,
            pretrained_MLP=self.pretrained_MLP,
        )

        if self.use_pretrain:
            self._load_pretrain_model()

        if not self.use_pretrain:
            nn.init.normal_(self.predict_layer.weight, std=1e-2)

    def _load_pretrain_model(self):
        predict_weight = torch.cat(
            [
                self.pretrained_GMF.predict_layer.weight,
                self.pretrained_MLP.predict_layer.weight,
            ],
            dim=1,
        )
        predict_bias = (
            self.pretrained_GMF.predict_layer.bias
            + self.pretrained_MLP.predict_layer.bias
        )
        self.predict_layer.weight.data.copy_(0.5 * predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        before_last_layer_output = torch.cat(
            (self.GMF(user, item), self.MLP(user, item)), dim=-1
        )
        output = self.predict_layer(before_last_layer_output)
        output = self.Sigmoid(output)
        return output.view(-1)

    def predict(self, users, items):
        with torch.no_grad():
            if len(users) == 1:
                # replicate user embedding to len(items)
                users = users.repeat(len(items))
            before_last_layer_output = torch.cat(
                (self.GMF(users, items), self.MLP(users, items)), dim=-1
            )
            preds = self.predict_layer(before_last_layer_output)
            preds = self.Sigmoid(preds)
        return preds.view(-1)
