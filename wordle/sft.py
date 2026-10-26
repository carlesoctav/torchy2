import logging
import os
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from torchy_core.callback import build_protected_weights, SSTProjectionCallback
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

exp_dir = Path(__file__).resolve().parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main(config: sws.FinalConfig, unused_argv: list[str] | None = None):
    model_config = config.train.model.to_dict()
    sft_config = config.train.sft.to_dict()
    output_dir = config.output_dir
    data_config = config.train.data.to_dict()

    eval_config = config.train.eval.to_dict() if config.train.get("eval") else None
    resume_checkpoint = config.get("resume_checkpoint", False)


    last_checkpoint = None

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    trl_logger = logging.getLogger("trl")
    if not trl_logger.handlers:
        trl_logger.addHandler(handler)
    trl_logger.setLevel(logging.INFO)
    core_logger = logging.getLogger("torchy_core")
    if not core_logger.handlers:
        core_logger.addHandler(handler)
    core_logger.setLevel(logging.INFO)


    if "peft_config" in model_config:
        model_config["peft_config"] = LoraConfig(**model_config["peft_config"])

    args = SFTConfig(**sft_config)

    config_json = Path(f"{output_dir}/config.json")
    config_json.parent.mkdir(parents=True, exist_ok=True)
    with config_json.open("w") as f:
        f.write(config.to_json())


    train_dataset = load_dataset(**data_config)
    eval_dataset = load_dataset(**eval_config) if eval_config else None



    model_config["pretrained_model_name_or_path"] = model_config.pop("model")
    print(f"DEBUGPRINT[13]: sft.py:68: model_config={model_config}")
    model = AutoModelForCausalLM.from_pretrained(
        **model_config
    )
    sst_callback = SSTProjectionCallback(build_protected_weights(model, 99))

    trainer = SFTTrainer(
        model, 
        args=args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, 
    )

    if resume_checkpoint:
        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        logger.info(f"[resume] Resuming from checkpoint: {last_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info(f"[done] Finished training. global_step={trainer.state.global_step}")
    finally:
        logger.info(f"save final-checkpoint training. global_step={trainer.state.global_step}")
        trainer._save_checkpoint(trainer.model, trial=None)




if __name__ == "__main__":
    sws.run(main, forward_extras=True)
