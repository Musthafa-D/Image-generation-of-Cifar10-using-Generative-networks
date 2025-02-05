import os
import ccbdl
import torch
from ccbdl.config_loader.loaders import ConfigurationLoader
from optuna_hyp import Optuna


ccbdl.utils.logging.del_logger(source=__file__)

config_path = os.path.join(os.getcwd(), "config.yaml")
config = ConfigurationLoader().read_config("config.yaml")

# Get Configurations
network_config = config["network"]
optimizer_config = config["optimized"]
data_config = config["data"]
learner_config = config["learning"]
study_config = config["study"]

study_path = ccbdl.storages.storages.generate_train_folder("", generate = True)

opti = Optuna(study_config,
              optimizer_config,
              network_config,
              data_config,
              learner_config,
              config,
              study_path,
              comment="Study for Testing",
              config_path=config_path,
              debug=False,
              logging=True)

# Run Parameter Optimizer
opti.start_study()

opti.eval_study()

# Summarize Study
opti.summarize_study()

# Saving the study
torch.save(opti.study, os.path.join(study_path, "study.pt"))

handler = ccbdl.evaluation.additional.notebook_handler("./StudySummary.ipynb", 
                                                       study_location = os.path.join(study_path, "study.pt"))
handler.save_as_html(directory = study_path, html_name = "study_analysis.html")

# Calculate Metrics for all trials
# opti.eval_metrics()
