import torch
from transformers import Trainer

class MyTrainer(Trainer):
    def create_optimizer(self):
        new_module_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name)
                continue

            if name.startswith("model.pooler"):
                new_module_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": decoder_params,
                    "lr": 1e-5,
                },
                {
                    "params": new_module_params,
                    "lr": 1e-3,
                },
            ],
            weight_decay=self.args.weight_decay,
        )

        return self.optimizer