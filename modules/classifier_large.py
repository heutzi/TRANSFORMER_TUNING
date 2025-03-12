import torch
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from typing import Optional


class DenseClassifierConfig(PretrainedConfig):

    def __init__(
        self,
        base_model: str,
        input_embed_size: int,
        num_classes: int,
        classifier_dropout: float = 0.5,
        classifier_hidden_size: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.input_embed_size = input_embed_size
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_hidden_size = classifier_hidden_size


class DenseClassifier(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer_model = AutoModel.from_pretrained(
            config.base_model,
            trust_remote_code=True
            )

        # LE CHIFFRE EST ICI DIRECTEMENT MODIFIÉ
        # POUR FAIRE VARIER LES COUCHES GELÉES
        l_pres = [f".{i}." for i in range(10)]
        for name, param in self.transformer_model.named_parameters():
            if any([l_pre in name for l_pre in l_pres]):
                param.requires_grad = False
        if not config.classifier_hidden_size:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(config.classifier_dropout),
                torch.nn.Linear(config.input_embed_size,
                                config.num_classes),
                )
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(config.classifier_dropout),
                torch.nn.Linear(config.input_embed_size,
                                config.classifier_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.classifier_hidden_size,
                                config.num_classes)
                )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Apply weight initialization to each layer.
        """
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    # make predictions
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
            ) -> 'dict':

        outputs = self.transformer_model(
            input_ids,
            attention_mask=attention_mask
            ).last_hidden_state

        logits = self.classifier(outputs[:, 0])

        if labels is not None:
            lossFunction = torch.nn.CrossEntropyLoss()
            loss = lossFunction(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
