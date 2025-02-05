import torch
import numpy as np
import os
import time
from plots import attributions, attribution_maps
from captum.metrics import infidelity, sensitivity_max
from tabulate import tabulate
from ccbdl.utils import DEVICE
from datetime import datetime


class Metrics:
    def __init__(self, model, test_data, result_folder, model_type, best_trial_check):
        self.model = model
        self.device = DEVICE
        self.test_data = test_data
        self.result_folder = result_folder
        self.model_type = model_type
        self.best_trial_check = best_trial_check

    def compute_metrics(self, method_name, method, method_map, inputs, labels):
        def my_perturb_func(inputs):
            noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().to(self.device)
            return noise, inputs - noise
        
        if self.model_type == 'CGAN':
            infidelity_score = infidelity(self.model, my_perturb_func, inputs, method_map, additional_forward_args=labels)
        
            if method_name == "Occlusion":
                sensitivity_score = sensitivity_max(method.attribute, inputs, sliding_window_shapes=(3, 5, 5), strides=(3, 2, 2), additional_forward_args=(labels))
            else:
                sensitivity_score = sensitivity_max(method.attribute, inputs, additional_forward_args=labels)
        else:
            infidelity_score = infidelity(self.model, my_perturb_func, inputs, method_map)
        
            if method_name == "Occlusion":
                sensitivity_score = sensitivity_max(method.attribute, inputs, sliding_window_shapes=(3, 5, 5), strides=(3, 2, 2))
            else:
                sensitivity_score = sensitivity_max(method.attribute, inputs)
    
        return infidelity_score, sensitivity_score

    def calculations(self):
        start_time = datetime.now()
        # Initialize variables to accumulate results
        if self.best_trial_check == 1:
            metrics_data = {
                "Saliency": {"infidelity": 0.0, "sensitivity": 0.0},
                "Guided Backprop": {"infidelity": 0.0, "sensitivity": 0.0},
                "Input x Gradient": {"infidelity": 0.0, "sensitivity": 0.0},
                "Deconvolution": {"infidelity": 0.0, "sensitivity": 0.0},
                "Occlusion": {"infidelity": 0.0, "sensitivity": 0.0}
            }
            total_samples = 0
            
            method_durations = {
                "Saliency": 0,
                "Guided Backprop": 0,
                "Input x Gradient": 0,
                "Deconvolution": 0,
                "Occlusion": 0
            }
            
        else:
            metrics_data = {
                "Saliency": {"infidelity": 0.0, "sensitivity": 0.0},
                "Guided Backprop": {"infidelity": 0.0, "sensitivity": 0.0},
                "Input x Gradient": {"infidelity": 0.0, "sensitivity": 0.0},
                "Deconvolution": {"infidelity": 0.0, "sensitivity": 0.0}
            }
            total_samples = 0
            
            method_durations = {
                "Saliency": 0,
                "Guided Backprop": 0,
                "Input x Gradient": 0,
                "Deconvolution": 0
            }

        
        for i, data in enumerate(self.test_data):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device).long()
            
            inputs.requires_grad = True

            if self.model_type == 'CGAN':
                attr, attr_maps = attributions(self.model, inputs), attribution_maps(self.model, inputs, labels)
            else:
                attr, attr_maps = attributions(self.model, inputs), attribution_maps(self.model, inputs)
            
            if self.best_trial_check != 1:
                attr, attr_maps = attr[:-1], attr_maps[:-1]

            for method_name, method, method_map in zip(metrics_data.keys(), attr, attr_maps):
                method_start_time = time.time()
                
                infid, sens = self.compute_metrics(method_name, method, method_map, inputs, labels)
                
                metrics_data[method_name]["infidelity"] += infid.sum().item()              
                metrics_data[method_name]["sensitivity"] += sens.sum().item()
                
                method_end_time = time.time()
                method_durations[method_name] += method_end_time - method_start_time
                print(f"{method_name}: {method_durations[method_name]}")

            print(f"{i}\n")
            total_samples += inputs.size(0)

        # Calculate average infidelity and sensitivity values
        for method_name in metrics_data:
            metrics_data[method_name]["infidelity"] /= total_samples
            metrics_data[method_name]["sensitivity"] /= total_samples

        # Create data for the table
        table_data = [[name] + list(values.values()) for name, values in metrics_data.items()]
        table_headers = ["Attribution", "Average Infidelity", "Average Sensitivity"]
        table_string = tabulate(table_data, headers=table_headers, tablefmt="grid")

        output_path = os.path.join(self.result_folder, "metric_values_of_test_dataset")
        os.makedirs(output_path, exist_ok=True)
        
        end_time = datetime.now()
        self.total_duration = end_time - start_time

        # Write the heading and table to a file
        with open(os.path.join(output_path, "metrics.txt"), "w") as file:
            file.write("Metrics of Cifar10 Test Dataset\n\n")
            file.write(table_string)
            file.write("\n\n")
            file.write(f"Total duration for calculating metrics: {str(self.total_duration)[:-7]}\n")
            
            for method_name, duration in method_durations.items():
                m, s = divmod(duration, 60)
                h, m = divmod(m, 60)
                file.write(f"Duration for {method_name}: {h} hours, {m} minutes, {s:.2f} seconds.\n")
        
    def total_metric_duration(self):
        return self.total_duration
