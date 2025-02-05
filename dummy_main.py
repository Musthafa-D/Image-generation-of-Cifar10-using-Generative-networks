import os
import ccbdl
from ccbdl.config_loader.loaders import ConfigurationLoader
from dummy_run import Normal_run
from datetime import datetime

ccbdl.utils.logging.del_logger(source=__file__)

# config_path = os.path.join(os.getcwd(), "dummy_config.yaml")
# config = ConfigurationLoader().read_config("dummy_config.yaml")

config_path = os.path.join(os.getcwd(), "dummy_config.yaml")
config = ConfigurationLoader().read_config("dummy_config.yaml")

# Get Configurations
network_config = config["network"]
data_config = config["data"]
learner_config = config["learning"]
task = data_config["task"]

def generate_train_folder(name: str = " ", generate: bool = False, location: str = False):
    """
    Returns the name of the training folder.

    Args:
        name (str, optional): DESCRIPTION. Defaults to " ":.
        generate (bool, optional): Enable to directly create the folder instead of only returning the path.\n. Defaults to False.
        location (str, optional): Alternative location of the folder. 00_Runs-folder will be automatically created at this location. Defaults to False.

    Returns:
        folder (TYPE): Train folder.

    """
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not location:
        location = os.getcwd()
    folder = os.path.join(location, "00_Runs", time_str + name)
    if generate:
        if not os.path.exists(folder):
            os.makedirs(folder)
    return folder

study_path = generate_train_folder("", generate = True)

normal = Normal_run(task,
              network_config,
              data_config,
              learner_config,
              config,
              study_path,
              comment="Study for Testing",
              config_path=config_path,
              debug=False,
              logging=True)

normal.execute()
# normal.eval_metrics()