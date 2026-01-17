import logging
import os, glob
import sys
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field
from fairseq.tasks import register_task

import torch

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask

else:
    from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask

from fairseq.checkpoint_utils import load_checkpoint_to_cpu

logger = logging.getLogger(__name__)

@dataclass
class AVHubertUnitPretrainingConfig(AVHubertPretrainingConfig):
    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, start from pretrained model without label_embs_concat",
        },
    )

@register_task("av_hubert_unit_pretraining", dataclass=AVHubertUnitPretrainingConfig)
class AVHubertUnitPretrainingTask(AVHubertPretrainingTask):
    def build_model(self, args):
        model = super(AVHubertPretrainingTask, self).build_model(args)
        if self.cfg.pretrained_checkpoint:
            state = load_checkpoint_to_cpu(self.cfg.pretrained_checkpoint)
            state_dict = state["model"]
            model_state_dict = model.state_dict()

            for w in ["label_embs_concat"]:
                state_dict[w] = model_state_dict[w]
            for w in ["final_proj.weight", "final_proj.bias"]:
                if state_dict[w].shape != model_state_dict[w].shape:
                    assert len(state_dict[w]) * 2 == len(model_state_dict[w])
                    state_dict[w] = torch.cat([state_dict[w].detach().clone(), state_dict[w].detach().clone()])

            model.load_state_dict(state_dict)

        return model
