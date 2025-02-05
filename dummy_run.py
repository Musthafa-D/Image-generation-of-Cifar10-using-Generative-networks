from networks import CGAN, GAN
from learner import Learner, Conditional_Learner
from ccbdl.utils import DEVICE
from datetime import datetime
from data_loader import prepare_data
from metrics import Metrics
import os
import torch


class Normal_run:
    def __init__(self,
                 task,
                 network_config: dict,
                 data_config: dict,
                 learner_config: dict,
                 config,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.result_folder = study_path
        self.config = config
        self.task = task
    
    def execute(self):
        start_time = datetime.now()
        
        print("\n\n******* Run is started*******")
        
        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)
        
        if self.learner_config["model"] == 'CGAN':
            model = CGAN("CGAN_Test",
                         self.learner_config["noise_dim"],
                         **self.network_config).to(DEVICE)
            
            self.learner = Conditional_Learner(self.result_folder,
                                   model,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   self.network_config,
                                   logging=True)

        elif self.learner_config["model"] == 'GAN':
            model = GAN("GAN_Test",
                        self.learner_config["noise_dim"],
                        **self.network_config).to(DEVICE)
            
            self.learner = Learner(self.result_folder,
                                   model,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   self.network_config,
                                   logging=True)
        
        else:
            raise ValueError(
                f"Invalid value for model: {self.learner_config['model']}, it should be 'GAN', or 'CGAN'")
            
        self.learner.fit(test_epoch_step=self.learner_config["testevery"])

        self.learner.parameter_storage.write("Current config:-\n")
        self.learner.parameter_storage.store(self.config)

        self.learner.parameter_storage.write(
            f"Start Time of gan training and evaluation in this run: {start_time.strftime('%H:%M:%S')}")

        self.learner.parameter_storage.write("\n")

        print("\n\n******* Run is completed*******")

        end_time = datetime.now()

        self.learner.parameter_storage.write(
            f"End Time of gan training and evaluation in this run: {end_time.strftime('%H:%M:%S')}\n")

        self.duration_trial = end_time - start_time
        self.durations=self.duration_trial

        self.learner.parameter_storage.write(
            f"Duration of gan training and evaluation in this run: {str(self.durations)[:-7]}\n")

        return self.learner.best_values["GenLoss"]
    
    def eval_metrics(self):
        """
        eval_metrics function of the run
            --> evaluates the metric values for test_data in the run
                and provides final results such as average 
                infidelity and sensitivityof the attributions used.
        Returns
            None.
        """
    
        model_path = os.path.join(self.result_folder, "net_best.pt")
        
        if self.learner_config["model"] == 'CGAN':
            model = CGAN("CGAN_Test",
                         self.learner_config["noise_dim"],
                         **self.network_config).to(DEVICE)

        elif self.learner_config["model"] == 'GAN':
            model = GAN("GAN_Test",
                        self.learner_config["noise_dim"],
                        **self.network_config).to(DEVICE)
        
        else:
            raise ValueError("Invalid values, it's either CGAN or GAN")
        
        # Load the model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        train_data, test_data, val_data = prepare_data(self.data_config)

        test_metrics = Metrics(model=model.discriminator, test_data=test_data, result_folder=self.result_folder,
                                    model_type=self.learner_config["model"], best_trial_check=0)
        
        test_metrics.calculations()
        duration_metrics_run = test_metrics.total_metric_duration()
        duration_per_run = self.durations + duration_metrics_run

        with open(os.path.join(self.result_folder, "ParameterStorage.txt"), "a") as file:
            file.write(f"Duration of metrics calculation of test data in this run: {str(duration_metrics_run)[:-7]}\n\n")
            file.write(f"Total duration of this run: {str(duration_per_run)[:-7]}")