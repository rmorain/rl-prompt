import os

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from sentiment_helpers import (
    FewShotClassificationDatasetConfig,
    PromptedClassificationRewardConfig,
    make_sentiment_reward,
)

from rlprompt.models import (
    LMAdaptorModelConfig,
    SinglePromptModelConfig,
    make_input_conditioned_prompt_model,
    make_lm_adaptor_model,
)
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.utils.utils import (
    colorful_print,
    compose_hydra_config_store,
    get_hydra_output_dir,
)

# Compose default config
config_list = [
    PromptedClassificationRewardConfig,
    FewShotClassificationDatasetConfig,
    LMAdaptorModelConfig,
    SinglePromptModelConfig,
    SQLModuleConfig,
    TrainerConfig,
]
cs = compose_hydra_config_store("base_fsc", config_list)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


@hydra.main(version_base=None, config_path="./", config_name="sentiment_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg="red")
    output_dir = get_hydra_output_dir()

    train_dataset = load_from_disk(os.path.join("data", config.dataset))

    print("Train Size:", len(train_dataset))
    print("Examples:", train_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_input_conditioned_prompt_model(policy_model, config)
    reward = make_sentiment_reward(config)
    algo_module = make_sql_module(prompt_model, reward, config)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, None, config, collator)
    trainer.train(config=config)


if __name__ == "__main__":
    main()
