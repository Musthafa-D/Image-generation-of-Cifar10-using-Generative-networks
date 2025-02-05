import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
import torch
import math
from sklearn.manifold import TSNE
from ccbdl.evaluation.plotting.base import GenericPlot
from captum.attr import visualization as viz
from captum.attr import Saliency, GuidedBackprop, InputXGradient, Deconvolution, Occlusion
from ccbdl.utils.logging import get_logger
from ccbdl.evaluation.plotting import graphs, images
from sklearn.metrics import confusion_matrix
from networks import CNN
from torch.utils.data import DataLoader, Dataset
from ccbdl.config_loader.loaders import ConfigurationLoader
from matplotlib.colors import LinearSegmentedColormap
from networks import NLRL_AO
import random
import matplotlib.ticker as ticker
from setsize import set_size

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


class TimePlot(GenericPlot):
    def __init__(self, learner):
        super(TimePlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating time plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        xs, ys = zip(*self.learner.data_storage.get_item("Time", batch=True))
        
        if self.learner.learner_config["model"] == 'CGAN':
            ax.plot(xs, [y - ys[0]for y in ys], label="cgan_train_time")
        else:
            ax.plot(xs, [y - ys[0]for y in ys], label="gan_train_time")
        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$t$', fontsize=14)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(ys))
        ax.legend()
        
        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "time_plot"))
        return figs, names


class Loss_plot(GenericPlot):
    def __init__(self, learner):
        super(Loss_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating loss plot")

    def consistency_check(self):
        return True
    
    def graph_multix_multiy(self, xs, ys, colors, alphas, *args, **kwargs):
        """
        Function to plot multiple y-values over different x-values.
    
        Args:
            xs (iterable of iterables): Each element of this 2-d iterable contains the full list of x-values.
            ys (iterable of iterables): Each element of this 2-d iterable contains the full list of y-values.
            colors (list): List of colors for each plot.
            alphas (list): List of alpha (transparency) values for each plot.
            *args (iterable): Default support for args (not used).
            **kwargs (dict): Default support for kwargs (not used).
    
        Returns:
            fig (matplotlib.figure.Figure): The generated figure.
        """
        fig = plt.figure()
        
        for idx in range(len(ys)):
            if "labels" in kwargs:
                label = kwargs["labels"][idx]
            else:
                label = str(idx)
                
            x = xs[idx]
            y = ys[idx]
            plt.plot(x, y, label=label, color=colors[idx], alpha=alphas[idx])
        
        plt.xlabel("$B$", fontsize=14)
        plt.ylabel('$\\mathcal{L}$', fontsize=14)
        
        plt.xlim(left=0, right=max(x))
        plt.ylim(bottom=0)
    
        plt.legend()
        plt.grid()
        plt.tight_layout()
    
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        
        return fig

    def plot(self):
        figs = []
        names = []
        
        xs = []
        ys = []
        
        plot_names = []
        
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("gen_loss")
        xs.append(x)
        ys.append(y)
        plot_names.append("$\\mathcal{L}_{\\mathrm{gen}}$")
        xs.append(x[: -49])
        ys.append(graphs.moving_average(y, 50))
        plot_names.append("$\\mathcal{L}_{\\mathrm{gen\\_avg}}$")
        
        
        x = self.learner.data_storage.get_item("batch") 
        y = self.learner.data_storage.get_item("dis_loss")
        xs.append(x)
        ys.append(y)
        plot_names.append("$\\mathcal{L}_{\\mathrm{dis}}$")
        xs.append(x[: -49])
        ys.append(graphs.moving_average(y, 50))
        plot_names.append("$\\mathcal{L}_{\\mathrm{dis\\_avg}}$")
        
        figs.append(self.graph_multix_multiy(xs = xs,
                                           ys = ys,
                                           labels = plot_names,
                                           xlim = (0, min([x[-1] for x in xs])),
                                           # ylim = (0, max([max(y) for y in ys])),
                                           ylim = (0, 4),
                                           colors = ['red', 'darkred', 'blue', 'darkblue'],
                                           alphas = [0.5, 0.5, 0.5, 0.5]
                                           ))
        names.append(os.path.join("plots", "losses"))
        return figs, names


class Psnr_plot(GenericPlot):
    def __init__(self, learner):
        super(Psnr_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating psnr score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("psnr_score")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the PSNR scores
        ax.plot(x, y)
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{psnr}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "psnr_scores")]
        return figs, names
    

class Ssim_plot(GenericPlot):
    def __init__(self, learner):
        super(Ssim_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating psnr score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("ssim_score")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the PSNR scores
        ax.plot(x, y)
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{ssim}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "ssim_scores")]
        return figs, names


class Fid_plot_nlrl(GenericPlot):
    def __init__(self, learner):
        super(Fid_plot_nlrl, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating fid score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("fid_score_nlrl")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the FID scores
        ax.plot(x, y, label="CNN_RGB")
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{fid}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "fid_scores_nlrl")]
        return figs, names
    

class Fid_plot_linear(GenericPlot):
    def __init__(self, learner):
        super(Fid_plot_linear, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating fid score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("fid_score_linear")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the FID scores
        ax.plot(x, y, label="CNN_RGB")
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{fid}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "fid_scores_linear")]
        return figs, names
        
        
class Image_generation(GenericPlot):
    def __init__(self, learner):
        super(Image_generation, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator")
        
        self.style = {}

    def consistency_check(self):
        return True
    
    def grid_2d(self, imgs, preds, figsize = (10, 10), labels=None):
        """
        Function to create of reconstructions of img data.

        Args:
            original (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            reconstructed (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).
            labels (torch.Tensor): Tensor with N number of int values randomly from 0 to 9. Defaults to None.

        Returns:
            None.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):        
            ax = plt.subplot(rows, cols, i+1)
            # get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
                
            if labels != None:
                class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                                  6: "frog", 7: "horse", 8: "ship", 9: "truck"}
                ax.set_title(f"{class_dict[labels[i].item()]}\nPrediction: {preds[i]:.3f}", fontsize=20)
            else:
                ax.set_title(f"{i}, Prediction: {preds[i]:.3f}", fontsize=20)
            # remove axes
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            
            # show img
            plt.imshow(img)
        plt.tight_layout()
        return fig
    
    def grid_2d_labels(self, imgs, preds, labels, figsize=(10, 10)):
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
    
        # Organize images by labels
        label_dict = {}
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in label_dict:
                label_dict[label_item] = []
            label_dict[label_item].append(i)
    
        # Determine grid size
        num_labels = len(label_dict)
        max_images_per_label = max(len(label_dict[key]) for key in label_dict)
        cols = num_labels
        rows = max_images_per_label
    
        fig = plt.figure(figsize=figsize)
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
    
        for label in label_dict:
            for idx, img_idx in enumerate(label_dict[label]):
                ax = plt.subplot(rows, cols, label + idx * cols + 1)
                img = imgs[img_idx].cpu().detach().permute(1, 2, 0).squeeze()
    
                ax.set_title(f"{class_dict[labels[i].item()]}, Pred: {preds[img_idx]:.3f}")
                
                # Remove axes
                frame = plt.gca()
                frame.axes.get_xaxis().set_ticks([])
                frame.axes.get_yaxis().set_ticks([])
    
                plt.imshow(img)
    
        plt.tight_layout()
        return fig

    def plot(self):
        generated_images = self.learner.data_storage.get_item("generated_images")
        predictions = self.learner.data_storage.get_item("predictions_gen")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        
        if self.learner.learner_config["model"] == 'CGAN':
            labels = self.learner.data_storage.get_item("labels_gen")
            
        total = len(epochs)
        # Number of batches per epoch
        batches_per_epoch = int(len(predictions)/total)

        figs = []
        names = []

        for idx in range(total):
            # Calculate the index for the last batch of the current epoch
            last_batch_index = ((idx + 1) * batches_per_epoch) - 1
            
            generated_images_per_epoch = generated_images[last_batch_index]
            num = generated_images_per_epoch.size(0)
            
            self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (15, 15)})
            
            epoch = epochs[idx]
            prediction_per_epoch= predictions[last_batch_index]
            
            if self.learner.learner_config["model"] == 'CGAN':
                label = labels[last_batch_index]
                figs.append(self.grid_2d(imgs=generated_images_per_epoch, 
                                                  labels=label, 
                                                  preds=prediction_per_epoch, 
                                                  **self.style))
            else:
                figs.append(self.grid_2d(imgs=generated_images_per_epoch, 
                                           preds=prediction_per_epoch,
                                           **self.style))

            names.append(os.path.join("plots", "generated_images", f"epoch_{epoch}"))
        return figs, names


class Image_generation_dataset(GenericPlot):
    def __init__(self, learner):
        super(Image_generation_dataset, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator from train")
        
        self.style = {}

    def consistency_check(self):
        return True
    
    def grid_2d(self, imgs, preds, figsize = (10, 10), labels=None):
        """
        Function to create of reconstructions of img data.

        Args:
            original (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            reconstructed (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).
            labels (torch.Tensor): Tensor with N number of int values randomly from 0 to 9. Defaults to None.

        Returns:
            None.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):        
            ax = plt.subplot(rows, cols, i+1)
            # get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
                
            if labels != None:
                class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                                  6: "frog", 7: "horse", 8: "ship", 9: "truck"}
                ax.set_title(f"{class_dict[labels[i].item()]}\nPrediction: {preds[i]:.3f}", fontsize=20)
            else:
                ax.set_title(f"{i}, Prediction: {preds[i]:.3f}", fontsize=20)
            # remove axes
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            
            # show img
            plt.imshow(img)
        plt.tight_layout()
        return fig

    def plot(self):
        generated_images = self.learner.data_storage.get_item("fake_images")
        predictions = self.learner.data_storage.get_item("predictions_fake")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        
        if self.learner.learner_config["model"] == 'CGAN':
            labels = self.learner.data_storage.get_item("labels")
            
        total = len(epochs)
        # Number of batches per epoch
        batches_per_epoch = int(len(predictions)/total)

        figs = []
        names = []

        for idx in range(total):
            # Calculate the index for the last batch of the current epoch
            last_batch_index = ((idx + 1) * batches_per_epoch) - 1
            
            generated_images_per_epoch = generated_images[last_batch_index]
            num = generated_images_per_epoch.size(0)
            
            self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (15, 15)})
            
            epoch = epochs[idx]
            prediction_per_epoch= predictions[last_batch_index]
            
            if self.learner.learner_config["model"] == 'CGAN':
                label = labels[last_batch_index]
                figs.append(self.grid_2d(imgs=generated_images_per_epoch, 
                                                  labels=label, 
                                                  preds=prediction_per_epoch, 
                                                  **self.style))
            else:
                figs.append(self.grid_2d(imgs=generated_images_per_epoch, 
                                           preds=prediction_per_epoch,
                                           **self.style))

            names.append(os.path.join("plots", "generated_images_train", f"epoch_{epoch}"))
        return figs, names


class Confusion_matrix_gan(GenericPlot):
    def __init__(self, learner):
        super(Confusion_matrix_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating confusion matrix for real vs. fake predictions")

    def consistency_check(self):
        return True

    def plot(self):
        names = []
        figs = []
        
        fake_probs = self.learner.data_storage.get_item("predictions_fake")
        real_probs = self.learner.data_storage.get_item("predictions_real")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        threshold = self.learner.threshold
        
        total = len(epochs)
        # Number of batches per epoch
        batches_per_epoch = int(len(real_probs)/total)
        
        for idx in range(total):
            fig, ax = plt.subplots(figsize=(10, 6))
            epoch = epochs[idx]
            # Calculate the index for the last batch of the current epoch
            last_batch_index = ((idx + 1) * batches_per_epoch) - 1
            
            # Extract and flatten the predictions for the current epoch
            fake_probs_per_epoch = fake_probs[last_batch_index].flatten()
            real_probs_per_epoch = real_probs[last_batch_index].flatten()

            # Convert probabilities to binary predictions using a threshold eg 0.5
            fake_predictions = (fake_probs_per_epoch < threshold).astype(int)
            real_predictions = (real_probs_per_epoch > threshold).astype(int)

            # Concatenate predictions and true labels
            predictions = np.concatenate([fake_predictions, real_predictions])
            correct_labels = np.concatenate([np.zeros_like(fake_predictions), np.ones_like(real_predictions)])
            
            # Compute confusion matrix
            matrix = confusion_matrix(correct_labels, predictions)

            # Plot the confusion matrix using seaborn's heatmap
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', ax=ax,
                        xticklabels=['Predicted Fake', 'Predicted Real'],
                        yticklabels=['Actual Fake', 'Actual Real'])
            
            figs.append(fig)
            plt.close(fig)
            names.append(os.path.join("plots", "confusion_matrices", "gan_training_based", f"epoch_{epoch}"))
        return figs, names


class Confusion_matrix_classifier(GenericPlot):
    def __init__(self, learner, **kwargs):
        super(Confusion_matrix_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating confusion matrix based on classifier")
        
        # get default stype values
        self.style = {}
        self.style["figsize"] = self.get_value_with_default("figsize", (12, 12), kwargs)
        self.style["cmap"] = self.get_value_with_default("cmap", "Blues", kwargs)
        self.style["ticks"] = self.get_value_with_default("ticks", "auto", kwargs)
        self.style["xrotation"] = self.get_value_with_default("xrotation", "vertical", kwargs)
        self.style["yrotation"] = self.get_value_with_default("yrotation", "horizontal", kwargs)
        self.style["color_threshold"] = self.get_value_with_default("color_threshold", 50, kwargs)
        self.style["color_high"] = self.get_value_with_default("color_high", "white", kwargs)
        self.style["color_low"] = self.get_value_with_default("color_low", "black", kwargs)
    
    def consistency_check(self):
        return True
    
    def plot(self):
        labels_extractor = Tsne_plot_classifier(self.learner)
        # Load the classifier
        classifier = labels_extractor.load_classifier()
        
        # Setting concatenation false by initializing value as 0
        cat = 0
        
        names = []
        figs = []
        
        real_images = self.learner.data_storage.get_item("real_images")
        fake_images = self.learner.data_storage.get_item("fake_images")
        true_labels = self.learner.data_storage.get_item("labels")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        
        total = len(epochs)
        # Number of batches per epoch
        batches_per_epoch = int(len(real_images)/total)
        
        for idx in range(total):
            epoch = epochs[idx]
            
            # Calculate the index for the last batch of the current epoch
            last_batch_index = ((idx + 1) * batches_per_epoch) - 1
            
            # Access the last batch of real, fake, and label data for the current epoch
            real_images_last_batch = real_images[last_batch_index]
            fake_images_last_batch = fake_images[last_batch_index]
            true_labels_last_batch = true_labels[last_batch_index]
    
            # Process real images
            real_dataset = ImageTensorDataset(real_images_last_batch)
            real_data_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)
            _, real_predicted_labels = labels_extractor.process_images(real_data_loader, classifier, cat)
            
            # Process fake images
            fake_dataset = ImageTensorDataset(fake_images_last_batch)
            fake_data_loader = DataLoader(fake_dataset, batch_size=64, shuffle=False)
            _, fake_predicted_labels = labels_extractor.process_images(fake_data_loader, classifier, cat)
            
            for types in ["real", "fake"]:
                predictions = real_predicted_labels if types == 'real' else fake_predicted_labels
                correct_labels = true_labels_last_batch
                
                # Flatten the predictions list to a single tensor
                # Each tensor in the list is a single-element tensor, so we concatenate them and then flatten
                predictions_tensor = torch.cat(predictions, dim=0).flatten()
                
                # Move tensors to CPU for sklearn compatibility
                predictions_np = predictions_tensor.detach().cpu().numpy()
                correct_labels_np = correct_labels.detach().cpu().numpy()

                # Ensure all classes (0-9) are included
                all_classes = np.arange(10)
                predictions_np = np.concatenate([predictions_np, all_classes])
                correct_labels_np = np.concatenate([correct_labels_np, all_classes])
                
                figs.append(images.plot_confusion_matrix(predictions_np,
                                                         correct_labels_np,
                                                         **self.style))

                names.append(os.path.join("plots", "confusion_matrices", "classifier_training_based", f"{types}", f"epoch_{epoch}"))
        return figs, names


class Tsne_plot_classifier(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on classifier's features and classifier's decision")
    
    def consistency_check(self):
        return True
    
    def load_classifier(self):
        config = ConfigurationLoader().read_config("config.yaml")
        if self.learner.learner_config['layer'] == 'linear':
            classifier_config = config["classifier_linear"]
        
            classifier = CNN(3,"Classifier to classify real and fake images", 
                          **classifier_config).to(self.learner.device)
            checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_linear.pt")
        elif self.learner.learner_config['layer'] == 'nlrl':
            classifier_config = config["classifier_nlrl"]
        
            classifier = CNN(3,"Classifier to classify real and fake images", 
                          **classifier_config).to(self.learner.device)
            checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_nlrl.pt")
        else:
            raise ValueError(
                f"Invalid value for layer: {self.learner.learner_config['layer']}, it should be 'linear' or 'nlrl'.")
        
        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        return classifier
    
    def get_features(self, classifier, imgs):
        activation = {}
        
        def get_activation(name):
            def hook(classifier, inp, output):
                activation[name] = output.detach()
            return hook
        
        # Register the hook
        if self.learner.learner_config["layer"] == 'nlrl':
            handle = classifier.model[-2].register_forward_hook(get_activation('conv'))
        else:
            handle = classifier.model[-1].register_forward_hook(get_activation('conv'))
        _ = classifier(imgs)
        
        # Remove the hook
        handle.remove()
        return activation['conv']
    
    def compute_tsne(self, features):
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features)
        return tsne_results
    
    def process_images(self, data_loader, classifier, cat):
        all_features = []
        all_labels = []
        
        for imgs in data_loader:
            outputs = classifier(imgs)
            _, predicted_labels = torch.max(outputs, 1)
            features = self.get_features(classifier, imgs)
            features = features.view(features.size(0), -1)  # Flatten the features
            all_features.append(features)
            all_labels.append(predicted_labels)
            
        # Concatenate all the features and labels from the batches
        if cat == 1:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    def plot(self):
        # Load the classifier
        classifier = self.load_classifier()
        
        # Setting concatenation true by initializing value as 1
        cat = 1
        config = ConfigurationLoader().read_config("config.yaml")
        data_config = config["data"]
        
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
    
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images)/total)
        
        real_images = total_real_images[-batches_per_epoch:]
        real_images = torch.cat(real_images)
        real_dataset = ImageTensorDataset(real_images)
        real_data_loader = DataLoader(real_dataset, batch_size=data_config["batch_size"], shuffle=False)
        real_features, real_labels = self.process_images(real_data_loader, classifier, cat)
    
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        
        fake_images = total_fake_images[-batches_per_epoch:]
        fake_images = torch.cat(fake_images)
        fake_dataset = ImageTensorDataset(fake_images)
        fake_data_loader = DataLoader(fake_dataset, batch_size=data_config["batch_size"], shuffle=False)
        fake_features, fake_labels = self.process_images(fake_data_loader, classifier, cat)
        
        real_label_counts = [torch.sum(real_labels == i).item() for i in range(10)]
        fake_label_counts = [torch.sum(fake_labels == i).item() for i in range(10)]
    
        # Combine features for t-SNE
        combined_features = torch.cat([real_features, fake_features], dim=0)
        tsne_results = self.compute_tsne(combined_features.cpu().numpy())
    
        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:len(real_features)]
        fake_tsne = tsne_results[len(real_features):]
        
        # Define a color palette for the labels
        palette = sns.color_palette("colorblind", 10)
        palette_fake = sns.color_palette("dark", 10)
    
        # Plotting
        figs, names = [], []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        for label in range(10):  # Cifar10 dataset has 10 labels
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = (real_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"{class_dict[label]}", 
                color=palette[label],
                alpha=0.5,
                marker='o'
            )
            # Fake images scatter plot
            fake_indices = (fake_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"{class_dict[label]}",
                color=palette_fake[label],
                alpha=0.5,
                marker='^'
            )
            ax.legend()
            figs.append(fig)
            plt.close(fig)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_classifier", f"label_{label}_counts_real_{real_label_counts[label]}_fake_{fake_label_counts[label]}"))   
            
        fig, ax = plt.subplots(figsize=(16, 10))
        
        for label in range(10):  # Cifar10 dataset has 10 labels
            # Filter data points by label
            real_indices = (real_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"{class_dict[label]}", 
                color=palette[label],
                alpha=0.5,
                marker='o'
            )
            # Fake images scatter plot
            fake_indices = (fake_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"{class_dict[label]}",
                color=palette_fake[label],
                alpha=0.5,
                marker='^'
            )
        ax.legend()
        
        figs.append(fig)        
        names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_classifier", "combined"))
        plt.close(fig)

        return figs, names


class Tsne_plot_dis_gan(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_dis_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating t-SNE plots based on GAN's discriminator's features and discriminator's decision")

    def consistency_check(self):
        return True

    def get_features_and_predictions(self, layer_num, images, discriminator):
        features = []
        predictions = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator.dis[layer_num].register_forward_hook(hook)

        # Process images through the discriminator
        for imgs in images:
            pred = discriminator(imgs)
            predictions.append(pred.detach())

        handle.remove()  # Remove the hook
        return torch.cat(features), torch.cat(predictions)

    def compute_tsne(self, features):
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())

    def plot(self):
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)

        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images) / total)
        real_images = total_real_images[-batches_per_epoch:]

        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-batches_per_epoch:]

        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-batches_per_epoch:]

        # Plotting
        figs, names = [], []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}

        # Load the discriminator
        for model_type in ["initial", "best"]:
            if model_type == "initial":
                discriminator = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                discriminator = self.learner._load_best()  # Load the best epoch's model with the respective weights

            # Determine the layer number based on the configuration
            if self.learner.learner_config["layer"] == 'nlrl':
                layer_num = -2
            else:
                layer_num = -3

            labels_cat = torch.cat(labels, dim=0)

            # Extract features and predictions from the discriminator for real and fake images
            real_features, real_predictions = self.get_features_and_predictions(layer_num, real_images, discriminator)
            fake_features, fake_predictions = self.get_features_and_predictions(layer_num, fake_images, discriminator)

            # Flatten the features
            real_features = real_features.view(real_features.size(0), -1)
            fake_features = fake_features.view(fake_features.size(0), -1)

            # Combine features for t-SNE
            combined_features = torch.cat([real_features, fake_features], dim=0)
            tsne_results = self.compute_tsne(combined_features)

            # Split t-SNE results back into real and fake
            half = len(tsne_results) // 2
            real_tsne = tsne_results[:half]
            fake_tsne = tsne_results[half:]

            # Prepare data for plotting
            real_pred_label = (real_predictions > 0.5).cpu().numpy().astype(int).flatten()
            fake_pred_label = (fake_predictions > 0.5).cpu().numpy().astype(int).flatten()

            # Combined plot for real and fake images
            fig, ax = plt.subplots(figsize=(18, 10))
            palette = sns.color_palette("colorblind", 10)

            # Plot real images with different colors for each label and mark based on predictions
            for label in range(10):
                label_indices = (labels_cat == label).cpu().numpy()  # Get a boolean array for images with the current label
                real_label_indices = np.where(label_indices)[0]  # Get indices of images with the current label

                correct_real_indices = real_label_indices[real_pred_label[real_label_indices] == 0]  # Indices of correctly classified real images
                incorrect_real_indices = real_label_indices[real_pred_label[real_label_indices] == 1]  # Indices of incorrectly classified real images

                sns.scatterplot(
                    ax=ax,
                    x=real_tsne[correct_real_indices, 0],
                    y=real_tsne[correct_real_indices, 1],
                    color=palette[label],
                    marker='o',
                    label=f"R_Real {class_dict[label]}", # classified as real
                    alpha=0.5
                )
                sns.scatterplot(
                    ax=ax,
                    x=real_tsne[incorrect_real_indices, 0],
                    y=real_tsne[incorrect_real_indices, 1],
                    color='blue',
                    marker='o',
                    label=f"F_Real {class_dict[label]}", # classified as fake
                    alpha=0.5
                )

            # Plot fake images
            correct_fake_indices = np.where(fake_pred_label == 0)[0]
            incorrect_fake_indices = np.where(fake_pred_label == 1)[0]

            sns.scatterplot(
                ax=ax,
                x=fake_tsne[correct_fake_indices, 0],
                y=fake_tsne[correct_fake_indices, 1],
                color='gray',
                marker='X',
                label="F_Fake", # classified as fake
                alpha=0.5
            )
            sns.scatterplot(
                ax=ax,
                x=fake_tsne[incorrect_fake_indices, 0],
                y=fake_tsne[incorrect_fake_indices, 1],
                color='green',
                marker='X',
                label="R_Fake", # classified as real
                alpha=0.5
            )
            ax.legend(title='Predictions', loc='best')

            figs.append(fig)
            plt.close(fig)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_gan_discriminator", f"{model_type}_combined_plot"))

            # Separate plot for real images
            for label in range(10):
                fig, ax = plt.subplots(figsize=(16, 10))  # Re-initialize figure and axis
                label_indices = (labels_cat == label).cpu().numpy()  # Get a boolean array for images with the current label
                real_label_indices = np.where(label_indices)[0]  # Get indices of images with the current label

                correct_real_indices = real_label_indices[real_pred_label[real_label_indices] == 0]  # Indices of correctly classified real images
                incorrect_real_indices = real_label_indices[real_pred_label[real_label_indices] == 1]  # Indices of incorrectly classified real images

                sns.scatterplot(
                    ax=ax,
                    x=real_tsne[correct_real_indices, 0],
                    y=real_tsne[correct_real_indices, 1],
                    color=palette[label],
                    marker='o',
                    label=f"R_Real {class_dict[label]}",
                    alpha=0.5
                )
                sns.scatterplot(
                    ax=ax,
                    x=real_tsne[incorrect_real_indices, 0],
                    y=real_tsne[incorrect_real_indices, 1],
                    color='blue',
                    marker='o',
                    label=f"F_Real {class_dict[label]}",
                    alpha=0.5
                )
                ax.legend(title='Predictions', loc='best')

                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_gan_discriminator", f"{model_type}_label_{label}_real"))

            # Separate plot for fake images
            fig, ax = plt.subplots(figsize=(16, 10))  # Re-initialize figure and axis

            sns.scatterplot(
                ax=ax,
                x=fake_tsne[correct_fake_indices, 0],
                y=fake_tsne[correct_fake_indices, 1],
                color='gray',
                marker='X',
                label="F_Fake",
                alpha=0.5
            )
            sns.scatterplot(
                ax=ax,
                x=fake_tsne[incorrect_fake_indices, 0],
                y=fake_tsne[incorrect_fake_indices, 1],
                color='green',
                marker='X',
                label="R_Fake",
                alpha=0.5
            )
            ax.legend(title='Predictions', loc='best')

            figs.append(fig)
            plt.close(fig)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_gan_discriminator", f"{model_type}_fake"))

        return figs, names
    

class Tsne_plot_dis_cgan(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_dis_cgan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating t-SNE plots based on CGAN's discriminator's features and discriminator's decision")

    def consistency_check(self):
        return True

    def get_features_and_predictions(self, layer_num, images, discriminator, labels):
        features = []
        predictions = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator.dis[layer_num].register_forward_hook(hook)

        # Process images through the discriminator
        for imgs, lbls in zip(images, labels):
            pred = discriminator(imgs, lbls)
            predictions.append(pred.detach())

        handle.remove()  # Remove the hook
        return torch.cat(features), torch.cat(predictions)

    def compute_tsne(self, features):
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())

    def plot(self):
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)

        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images) / total)
        real_images = total_real_images[-batches_per_epoch:]

        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-batches_per_epoch:]

        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-batches_per_epoch:]

        # Plotting
        figs, names = [], []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}

        # Load the discriminator
        for model_type in ["initial", "best"]:
            if model_type == "initial":
                discriminator = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                discriminator = self.learner._load_best()  # Load the best epoch's model with the respective weights

            # Determine the layer number based on the configuration
            if self.learner.learner_config["layer"] == 'nlrl':
                layer_num = -2
            else:
                layer_num = -3

            labels_cat = torch.cat(labels, dim=0)

            # Extract features and predictions from the discriminator for real and fake images
            real_features, real_predictions = self.get_features_and_predictions(layer_num, real_images, discriminator, labels)
            fake_features, fake_predictions = self.get_features_and_predictions(layer_num, fake_images, discriminator, labels)

            # Flatten the features
            real_features = real_features.view(real_features.size(0), -1)
            fake_features = fake_features.view(fake_features.size(0), -1)

            # Combine features for t-SNE
            combined_features = torch.cat([real_features, fake_features], dim=0)
            tsne_results = self.compute_tsne(combined_features)

            # Split t-SNE results back into real and fake
            half = len(tsne_results) // 2
            real_tsne = tsne_results[:half]
            fake_tsne = tsne_results[half:]

            # Prepare data for plotting
            real_pred_label = (real_predictions > 0.5).cpu().numpy().astype(int).flatten()
            fake_pred_label = (fake_predictions > 0.5).cpu().numpy().astype(int).flatten()

            # Combined plot for real and fake images
            fig_combined, ax_combined = plt.subplots(figsize=(18, 10))
            palette = sns.color_palette("colorblind", 10)
            palette_fake = sns.color_palette("dark", 10)

            # Plot real images with different colors for each label and mark based on predictions
            for label in range(10):
                label_indices = (labels_cat == label).cpu().numpy()
                real_label_indices = np.where(label_indices)[0]

                correct_real_indices = real_label_indices[real_pred_label[real_label_indices] == 0]
                incorrect_real_indices = real_label_indices[real_pred_label[real_label_indices] == 1]

                sns.scatterplot(
                    ax=ax_combined,
                    x=real_tsne[correct_real_indices, 0],
                    y=real_tsne[correct_real_indices, 1],
                    color=palette[label],
                    marker='o',
                    label=f"R_Real {class_dict[label]}",
                    alpha=0.5
                )
                sns.scatterplot(
                    ax=ax_combined,
                    x=real_tsne[incorrect_real_indices, 0],
                    y=real_tsne[incorrect_real_indices, 1],
                    color='blue',
                    marker='o',
                    label=f"F_Real {class_dict[label]}",
                    alpha=0.5
                )

                fake_label_indices = np.where(label_indices)[0]

                correct_fake_indices = fake_label_indices[fake_pred_label[fake_label_indices] == 0]
                incorrect_fake_indices = fake_label_indices[fake_pred_label[fake_label_indices] == 1]

                sns.scatterplot(
                    ax=ax_combined,
                    x=fake_tsne[correct_fake_indices, 0],
                    y=fake_tsne[correct_fake_indices, 1],
                    color=palette_fake[label],
                    marker='X',
                    label=f"F_Fake {class_dict[label]}",
                    alpha=0.5
                )
                sns.scatterplot(
                    ax=ax_combined,
                    x=fake_tsne[incorrect_fake_indices, 0],
                    y=fake_tsne[incorrect_fake_indices, 1],
                    color='green',
                    marker='X',
                    label=f"R_Fake {class_dict[label]}",
                    alpha=0.5
                )
            ax_combined.legend(title='Predictions', loc='best')

            figs.append(fig_combined)
            plt.close(fig_combined)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_cgan_discriminator", f"{model_type}_combined_plot"))

            # Separate plot for real images
            for label in range(10):
                fig_real, ax_real = plt.subplots(figsize=(16, 10))  # Re-initialize figure and axis
                label_indices = (labels_cat == label).cpu().numpy()  # Get a boolean array for images with the current label
                real_label_indices = np.where(label_indices)[0]  # Get indices of images with the current label

                correct_real_indices = real_label_indices[real_pred_label[real_label_indices] == 0]  # Indices of correctly classified real images
                incorrect_real_indices = real_label_indices[real_pred_label[real_label_indices] == 1]  # Indices of incorrectly classified real images

                sns.scatterplot(
                    ax=ax_real,
                    x=real_tsne[correct_real_indices, 0],
                    y=real_tsne[correct_real_indices, 1],
                    color=palette[label],
                    marker='o',
                    label=f"R_Real {class_dict[label]}",
                    alpha=0.5
                )
                sns.scatterplot(
                    ax=ax_real,
                    x=real_tsne[incorrect_real_indices, 0],
                    y=real_tsne[incorrect_real_indices, 1],
                    color='blue',
                    marker='o',
                    label=f"F_Real {class_dict[label]}",
                    alpha=0.5
                )
                ax_real.legend(title='Predictions', loc='best')

                figs.append(fig_real)
                plt.close(fig_real)
                names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_cgan_discriminator", f"{model_type}_label_{label}_real"))

            # Separate plot for fake images
            for label in range(10):
                fig_fake, ax_fake = plt.subplots(figsize=(16, 10))  # Re-initialize figure and axis
                label_indices = (labels_cat == label).cpu().numpy()  # Get a boolean array for images with the current label
                fake_label_indices = np.where(label_indices)[0]  # Get indices of images with the current label

                correct_fake_indices = fake_label_indices[fake_pred_label[fake_label_indices] == 0]  # Indices of correctly classified fake images
                incorrect_fake_indices = fake_label_indices[fake_pred_label[fake_label_indices] == 1]  # Indices of incorrectly classified fake images

                sns.scatterplot(
                    ax=ax_fake,
                    x=fake_tsne[correct_fake_indices, 0],
                    y=fake_tsne[correct_fake_indices, 1],
                    color=palette_fake[label],
                    marker='X',
                    label=f"F_Fake {class_dict[label]}",
                    alpha=0.5
                )
                sns.scatterplot(
                    ax=ax_fake,
                    x=fake_tsne[incorrect_fake_indices, 0],
                    y=fake_tsne[incorrect_fake_indices, 1],
                    color='green',
                    marker='X',
                    label=f"R_Fake {class_dict[label]}",
                    alpha=0.5
                )
                ax_fake.legend(title='Predictions', loc='best')

                figs.append(fig_fake)
                plt.close(fig_fake)
                names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_cgan_discriminator", f"{model_type}_label_{label}_fake"))

        return figs, names


class Tsne_plot_images_combined(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_images_combined, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating combined tsne plots")

    def consistency_check(self):
        return True

    def plot(self):
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        labels = self.learner.data_storage.get_item("labels")

        num_classes = 10
        # Create color map for real and fake images
        colors = cm.rainbow(np.linspace(0, 1, num_classes))

        tsne = TSNE(n_components=2)

        fig, ax = plt.subplots(figsize=(16, 10))

        # Storage for transformed data
        real_transformed_list = []
        fake_transformed_list = []

        for lb in range(num_classes):
            idx = labels == lb
            real_image = real_images[idx].view(real_images[idx].size(0), -1).cpu().numpy()  # real_images of current class
            fake_image = fake_images[idx].view(fake_images[idx].size(0), -1).cpu().numpy()  # generated_images of current class

            # Compute TSNE for each class and store
            real_transformed_list.append(tsne.fit_transform(real_image))
            fake_transformed_list.append(tsne.fit_transform(fake_image))

        # Plot each class for real and fake images
        for lb in range(num_classes):
            ax.scatter(real_transformed_list[lb][:, 0], real_transformed_list[lb][:, 1], c=[colors[lb]], label=f'{lb}')
            ax.scatter(fake_transformed_list[lb][:, 0], fake_transformed_list[lb][:, 1], c=[colors[lb]], marker='x', label=f'{lb}')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "analysis_plots", "tsne_plots", "combined_tsne_plot")]
        return figs, names


class Tsne_plot_images_separate(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_images_separate, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots for each label")

    def consistency_check(self):
        return True

    def plot(self):
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        labels = self.learner.data_storage.get_item("labels")

        figs = []
        names = []

        num_classes = 10

        for lb in range(num_classes):
            idx = labels == lb
            real_image = real_images[idx]  # real_images of current class  
            fake_image = fake_images[idx] # generated_images of current class

            real_image = real_image.view(real_image.size(0), -1).cpu().numpy()
            fake_image = fake_image.view(fake_image.size(0), -1).cpu().numpy()

            # compute TSNE
            tsne = TSNE(n_components=2)

            real_transformed = tsne.fit_transform(real_image)
            fake_transformed = tsne.fit_transform(fake_image)

            fig, ax = plt.subplots(figsize=(16, 10))

            ax.scatter(
                real_transformed[:, 0], real_transformed[:, 1], label='real images')

            ax.scatter(
                fake_transformed[:, 0], fake_transformed[:, 1], label='fake images')

            ax.legend()

            figs.append(fig)
            plt.close(fig)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "seperate_plots", f"label_{lb}"))          
        return figs, names


class Attribution_plots(GenericPlot):
    def __init__(self, learner):
        super(Attribution_plots, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for some grayscale images of mnist")

    def consistency_check(self):
        return True
    
    def safe_visualize(self, attr, title, fig, ax, label, img_name, types, cmap, check):
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if check == 0:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         title=title,
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
            else:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot
        
        # Plotting
        figs, names = [], []
        
        # Load the discriminator  
        for models in ["initial", "best"]:
            if models == "initial":
                model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model = self.learner._load_best()  # Load the best epoch's model with the respective weights
        
            # Custom cmap for better visulaization
            cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])
    
            fake_images = self.learner.data_storage.get_item("fake_images")
            real_images = self.learner.data_storage.get_item("real_images")
    
            for types in ["real", "fake"]:
                inputs = real_images[-1].clone() if types == 'real' else fake_images[-1].clone()
        
                inputs.requires_grad = True  # Requires gradients set true
                
                preds = model(inputs)
                # Get attribution maps for different techniques
                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps(model, inputs)
    
                attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
    
                # Process all input images
                total_indices = list(range(inputs.shape[0]))
                subsets = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]
    
                for subset in subsets:
                    num_rows = len(subset)
                    num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4 * num_rows))
                
                    # Adjust the shape of axs array if needed
                    if num_rows == 1 and num_cols == 1:
                        axs = np.array([[axs]])
                    elif num_rows == 1:
                        axs = axs[np.newaxis, :]
                    elif num_cols == 1:
                        axs = axs[:, np.newaxis]
                
                    count = 0
                    for idx in subset:
                        img = (inputs[idx].cpu().detach().permute(1, 2, 0)).numpy()   
                        pred = preds[idx].item()
                        
                        results = [
                            np.transpose(saliency_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(guided_backprop_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(input_x_gradient_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(deconv_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(occlusion_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        ]
                
                        # Display the original image
                        axs[count, 0].imshow(img)
                        axs[count, 0].set_title(f"Prediction: {pred:.3f}", fontsize=17)
                        axs[count, 0].axis("off")
                
                        # Display each of the attribution maps next to the original image
                        for col, (attr, res) in enumerate(zip(attrs, results)):
                            title = f"{attr}" 
                            if len(subset) > 1:
                                if idx == subset[0]:
                                    check = 0
                                else:
                                    check = 1
                            else:
                                check = 0
                            
                            # Call the visualization function, passing None for label and img_name since they are not applicable
                            self.safe_visualize(res, title, fig, axs[count, col + 1], None, None, types, cmap, check)
                
                        count += 1
                        
                    # Add a single colorbar for all subplots below the grid
                    fig.subplots_adjust(bottom=0.15)
                    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
                    norm = plt.Normalize(vmin=-1, vmax=1)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=17)  # Set colorbar tick label font size
                
                    # Store the figure with an appropriate name
                    figs.append(fig)
                    plt.close(fig)
                    names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{models}_{types}_subsets_{subsets.index(subset) + 1}"))
        return figs, names


class Attribution_plots_conditional(GenericPlot):
    def __init__(self, learner):
        super(Attribution_plots_conditional, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for some grayscale images of mnist")

    def consistency_check(self):
        return True
    
    def safe_visualize(self, attr, title, fig, ax, label, img_name, types, cmap, check):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if check == 0:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         title=title,
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
            else:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot
        
        # Plotting
        figs, names = [], []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        # Load the discriminator  
        for models in ["initial", "best"]:
            if models == "initial":
                model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model = self.learner._load_best()  # Load the best epoch's model with the respective weights
        
            # Custom cmap for better visulaization
            cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])
            
            imp_values = Softmax_plot_classifier_conditional(self.learner)
            
            real_images, fake_images, labels = imp_values.values()
    
            for types in ["real", "fake"]:
                inputs = real_images.clone() if types == 'real' else fake_images.clone()
        
                inputs.requires_grad = True  # Requires gradients set true
                
                class_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                              6: "6", 7: "7", 8: "8", 9: "9"}
                preds = model(inputs, labels)
                # Get attribution maps for different techniques
                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps(model, inputs, labels)
    
                attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
    
                # Process all input images
                total_indices = list(range(inputs.shape[0]))
                subsets = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]
    
                for subset in subsets:
                    num_rows = len(subset)
                    num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4 * num_rows))
                
                    # Adjust the shape of axs array if needed
                    if num_rows == 1 and num_cols == 1:
                        axs = np.array([[axs]])
                    elif num_rows == 1:
                        axs = axs[np.newaxis, :]
                    elif num_cols == 1:
                        axs = axs[:, np.newaxis]
                
                    count = 0
                    for idx in subset:
                        img = (inputs[idx].cpu().detach().permute(1, 2, 0)).numpy()   
                        label = labels[idx].cpu().detach()
                        pred = preds[idx].item()
                        
                        results = [
                            np.transpose(saliency_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(guided_backprop_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(input_x_gradient_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(deconv_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                            np.transpose(occlusion_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        ]
                
                        # Display the original image
                        axs[count, 0].imshow(img)
                        axs[count, 0].set_title(f"Actual: {class_dict[label.item()]}\nPrediction: {pred:.3f}", fontsize=17)
                        axs[count, 0].axis("off")
                
                        # Display each of the attribution maps next to the original image
                        for col, (attr, res) in enumerate(zip(attrs, results)):
                            title = f"{attr}"
                            if len(subset) > 1:
                                if idx == subset[0]:
                                    check = 0
                                else:
                                    check = 1
                            else:
                                check = 0
                            
                            self.safe_visualize(res, title, fig, axs[count, col + 1], label, class_dict[label.item()], types, cmap, check)
                
                        count += 1
                        
                    # Add a single colorbar for all subplots below the grid
                    fig.subplots_adjust(bottom=0.15)
                    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
                    norm = plt.Normalize(vmin=-1, vmax=1)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=17)  # Set colorbar tick label font size
                
                    # Store the figure with an appropriate name
                    figs.append(fig)
                    plt.close(fig)
                    names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{models}_{types}_subsets_{subsets.index(subset) + 1}"))
        return figs, names


# Custom Dataset class to handle lists of tensors
class ImageTensorDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

# Other functions
def attributions(model, inputs):
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    # Initialize the Occlusion object
    occlusion = Occlusion(model)   
    return saliency, guided_backprop, input_x_gradient, deconv, occlusion

def attribution_maps(model, inputs, labels=None):
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model, inputs)
    
    if labels is not None:
        saliency_maps = saliency.attribute(inputs, additional_forward_args=labels)
        guided_backprop_maps = guided_backprop.attribute(inputs, additional_forward_args=labels)
        input_x_gradient_maps = input_x_gradient.attribute(inputs, additional_forward_args=labels)
        deconv_maps = deconv.attribute(inputs, additional_forward_args=labels)
        occlusion_maps = occlusion.attribute(inputs, additional_forward_args=labels, sliding_window_shapes=(3, 5, 5), strides=(3, 2, 2))
    else:
        # For GAN, we do not pass the labels
        saliency_maps = saliency.attribute(inputs)
        guided_backprop_maps = guided_backprop.attribute(inputs)
        input_x_gradient_maps = input_x_gradient.attribute(inputs)
        deconv_maps = deconv.attribute(inputs)
        occlusion_maps = occlusion.attribute(inputs, sliding_window_shapes=(3, 5, 5), strides=(3, 2, 2))
    return saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps


class Attribution_plots_classifier(GenericPlot):
    def __init__(self, learner):
        """
        init method of the attribution plots based on pre trained classifier

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Attribution_plots_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for grayscale images")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def safe_visualize(self, attr, title, fig, ax, label, img_name, types, cmap, check):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if check == 0:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         title=title,
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
            else:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        """
        method to plot the attribution plots.

        Returns
        -------
        figs of the attribution plots, names of the attribution plots (for saving the plots with this name).

        """
        names = []
        figs = []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot
        
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        
        labels_extractor = Tsne_plot_classifier(self.learner)
        # Load the classifier
        classifier = labels_extractor.load_classifier()
        
        # Load the model
        model = classifier
                
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])

        for types in ["real", "fake"]:
            inputs = real_images[-1].clone().detach().requires_grad_(True) if types == 'real' else fake_images[-1].clone().detach().requires_grad_(True) # Requires gradients set true
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)           

            # Get attribution maps for different techniques
            saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_classifier(model, 
                                                                                                                                  inputs, 
                                                                                                                                  preds)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
            
            # Process all input images
            total_indices = list(range(inputs.shape[0]))
            subsets = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for subset in subsets:
                num_rows = len(subset)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in subset:
                    img = (inputs[idx].cpu().detach().permute(1, 2, 0)).numpy()  
                    pred = preds[idx].cpu().detach()
                    
                    results = [
                        np.transpose(saliency_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(guided_backprop_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(input_x_gradient_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(deconv_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(occlusion_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                    ]

                    # Display the original image
                    axs[count, 0].imshow(img)
                    axs[count, 0].set_title(f"Predicted: {class_dict[pred.item()]}", fontsize=17)
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        if len(subset) > 1:
                            if idx == subset[0]:
                                check = 0
                            else:
                                check = 1
                        else:
                            check = 0
                            
                        # Call the visualization function, passing None for label and img_name since they are not applicable
                        self.safe_visualize(res, title, fig, axs[count, col + 1], None, None, types, cmap, check)
            
                    count += 1
                    
                # Add a single colorbar for all subplots below the grid
                fig.subplots_adjust(bottom=0.15)
                cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
                norm = plt.Normalize(vmin=-1, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=17)  # Set colorbar tick label font size
       
                # Store the figure with an appropriate name
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "analysis_plots", "attribution_plots_classifier", f"{types}_subsets_{subsets.index(subset) + 1}"))
        return figs, names


# Other functions
def attributions_classifier(model, inputs, labels):
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    # Initialize the Occlusion object
    occlusion = Occlusion(model)    
    return saliency, guided_backprop, input_x_gradient, deconv, occlusion

def attribution_maps_classifier(model, inputs, labels):
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions_classifier(model, inputs, labels)
    
    saliency_maps = saliency.attribute(inputs, target=labels)
    guided_backprop_maps = guided_backprop.attribute(inputs, target=labels)
    input_x_gradient_maps = input_x_gradient.attribute(inputs, target=labels)
    deconv_maps = deconv.attribute(inputs, target=labels)
    occlusion_maps = occlusion.attribute(inputs, target=labels, sliding_window_shapes=(3, 5, 5), strides=(3, 2, 2))    
    return saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps


class Hist_plot(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None

    def plot(self):
        figs=[]
        names=[]
        
        labels = ['negation', 'relevancy', 'selection']
        # Load the discriminator  
        for models in ["initial", "best"]:
            if models == "initial":
                model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model = self.learner._load_best()  # Load the best epoch's model with the respective weights
            params, init_params = self.extract_parameters(model)
        
            for i, (param, init_param) in enumerate(zip(params, init_params)):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
                ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
                
                ax.set_xlabel('$\sigma(W)$', fontsize=14) # sigmoid of the learnable weight matrices
                ax.set_ylabel('$|W|$', fontsize=14) # number of parameters
                ax.legend(loc='upper right')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "histogram_plots", f"{models}_{labels[i]}"))
        return figs, names


class Softmax_plot_classifier_conditional(GenericPlot):
    def __init__(self, learner):
        super(Softmax_plot_classifier_conditional, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating softmax bar plots")

    def consistency_check(self):
        return True
    
    def values(self):
        real_images_list = self.learner.data_storage.get_item("real_images")
        fake_images_list = self.learner.data_storage.get_item("fake_images")
        labels_list = self.learner.data_storage.get_item("labels")
        
        real_images = torch.cat(real_images_list, dim=0)
        fake_images = torch.cat(fake_images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Dictionary to hold indices for each label
        label_indices = {i: [] for i in range(10)}  # labels from 0 to 9
        
        # Populate the dictionary with indices for each label
        for idx, label in enumerate(labels):
            label_indices[label.item()].append(idx)
        
        # List to store the selected indices
        selected_indices = []
    
        # Sample 5 images from each label group
        for label, indices in label_indices.items():
            if len(indices) >= 5:
                selected_indices.extend(random.sample(indices, 5))
            else:
                selected_indices.extend(indices)
        
        # Extract the selected real images, fake images and labels
        selected_real_images = real_images[selected_indices]
        selected_fake_images = fake_images[selected_indices]
        selected_labels = labels[selected_indices]
        return selected_real_images, selected_fake_images, selected_labels

    def plot(self):
        names = []
        figs = []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        max_images_per_plot = 5 # Define a constant for the maximum number of images per plot
        real_images, fake_images, labels = self.values()
        for types in ["real", "fake"]:
            if types == "real":
                inputs = real_images
            else:
                inputs = fake_images
            sm = torch.nn.Softmax(dim=0)
            classifier = Tsne_plot_classifier(self.learner)
            model = classifier.load_classifier()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            
            correct_indices = (labels == preds).nonzero(as_tuple=True)[0].tolist()
            incorrect_indices = (labels != preds).nonzero(as_tuple=True)[0].tolist()

            for label_type, indices in [("correctly_classified", correct_indices), ("misclassified", incorrect_indices)]:
                subsets = [indices[i:i + max_images_per_plot] for i in range(0, len(indices), max_images_per_plot)]
                
                for subset in subsets:
                    fig, axs = plt.subplots(2, len(subset), figsize=(5 * len(subset), 8), squeeze=False)
                    
                    for idx, image_idx in enumerate(subset):
                        img = (inputs[image_idx].cpu().detach().permute(1, 2, 0)).numpy()
                        label = labels[image_idx].cpu().detach()
                        pred = preds[image_idx].cpu().detach()

                        num_classes = outputs[image_idx].cpu().detach().shape[0]
                        output_softmax = sm(outputs[image_idx].cpu().detach()).numpy()

                        axs[0, idx].imshow(img)
                        axs[0, idx].set_title(f"Actual: {class_dict[label.item()]}\nPredicted: {class_dict[pred.item()]}", fontsize=17)
                        axs[0, idx].axis("off")
                        
                        axs[1, idx].bar(range(num_classes), output_softmax)
                        axs[1, idx].set_xticks(range(num_classes))
                        if idx == 0:
                            axs[1, idx].set_ylabel("$P$", fontsize=17) # class probability P(y/x)
                        if len(subset) == 5:
                            axs[1, 2].set_xlabel("$y$", fontsize=17) # class
                        if len(subset) == 4:
                            axs[1, 1].set_xlabel("$y$", fontsize=17)
                        if len(subset) == 3:
                            axs[1, 1].set_xlabel("$y$", fontsize=17)
                        if len(subset) == 2:
                            axs[1, 0].set_xlabel("$y$", fontsize=17)
                        if len(subset) == 1:
                            axs[1, idx].set_xlabel("$y$", fontsize=17)
                        axs[1, idx].set_ylim((0, 1))
                        axs[1, idx].set_yticks(torch.arange(0, 1.1, 0.1).tolist())


                    if label_type == "correctly_classified":
                        fig.suptitle("classified correctly")
                        dir_name = f"{types}_{label_type}_label_plots_{subsets.index(subset) + 1}"
                    else:
                        fig.suptitle("misclassified")
                        dir_name = f"{types}_{label_type}_label_plots_{subsets.index(subset) + 1}"

                    names.append(os.path.join("plots", "analysis_plots", "softmax_plots_classifier", dir_name))
                    figs.append(fig)
                    plt.close(fig)
        return figs, names


class Softmax_plot_classifier(GenericPlot):
    def __init__(self, learner):
        super(Softmax_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating softmax bar plots")

    def consistency_check(self):
        return True
    
    def values(self):
        real_images_list = self.learner.data_storage.get_item("real_images")
        fake_images_list = self.learner.data_storage.get_item("fake_images")
        
        real_images = torch.cat(real_images_list, dim=0)
        fake_images = torch.cat(fake_images_list, dim=0)
        
        selected_real_images = real_images[:50]
        selected_fake_images = fake_images[:50]       
        return selected_real_images, selected_fake_images

    def plot(self):
        names = []
        figs = []
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        max_images_per_plot = 5 # Define a constant for the maximum number of images per plot
        real_images, fake_images = self.values()
        for types in ["real", "fake"]:
            if types == "real":
                inputs = real_images
            else:
                inputs = fake_images
            sm = torch.nn.Softmax(dim=0)
            classifier = Tsne_plot_classifier(self.learner)
            model = classifier.load_classifier()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            
            # Create subsets for plotting
            subsets = [range(i, min(i + max_images_per_plot, len(inputs))) for i in range(0, len(inputs), max_images_per_plot)]
            
            for subset in subsets:
                fig, axs = plt.subplots(2, len(subset), figsize=(5 * len(subset), 8), squeeze=False)
                
                for idx, image_idx in enumerate(subset):
                    img = (inputs[image_idx].cpu().detach().permute(1, 2, 0)).numpy()
                    pred = preds[image_idx].cpu().detach()
    
                    num_classes = outputs[image_idx].cpu().detach().shape[0]
                    output_softmax = sm(outputs[image_idx].cpu().detach()).numpy()
    
                    axs[0, idx].imshow(img)
                    axs[0, idx].set_title(f"Predicted: {class_dict[pred.item()]}", fontsize=17)
                    axs[0, idx].axis("off")
                    
                    axs[1, idx].bar(range(num_classes), output_softmax)
                    axs[1, idx].set_xticks(range(num_classes))
                    if len(subset) == 5:
                        axs[1, 2].set_xlabel("$y$", fontsize=17)
                    if idx == 0:
                        axs[1, idx].set_ylabel("$P$", fontsize=17)
                    axs[1, idx].set_ylim((0, 1))
                    axs[1, idx].set_yticks(torch.arange(0, 1.1, 0.1).tolist())
    
                dir_name = f"{types}_{subsets.index(subset) + 1}"
                names.append(os.path.join("plots", "analysis_plots", "softmax_plots_classifier", dir_name))
                figs.append(fig)
                plt.close(fig)        
        return figs, names


class Attribution_plots_classifier_conditional(GenericPlot):
    def __init__(self, learner):
        """
        init method of the attribution plots based on pre trained classifier

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Attribution_plots_classifier_conditional, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for grayscale images")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def safe_visualize(self, attr, title, fig, ax, label, img_name, types, cmap, check):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if check == 0:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         title=title,
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
            else:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        """
        method to plot the attribution plots.

        Returns
        -------
        figs of the attribution plots, names of the attribution plots (for saving the plots with this name).

        """
        names = []
        figs = []
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot
        
        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
        
        labels_extractor = Tsne_plot_classifier(self.learner)
        # Load the classifier
        classifier = labels_extractor.load_classifier()
        
        # Load the model
        model = classifier
                
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])
        
        imp_values = Softmax_plot_classifier_conditional(self.learner)
        
        real_images, fake_images, labels = imp_values.values()

        for types in ["real", "fake"]:
            inputs = real_images.clone().detach().requires_grad_(True) if types == 'real' else fake_images.clone().detach().requires_grad_(True) # Requires gradients set true
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Get attribution maps for different techniques
            saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_classifier(model, 
                                                                                                                                  inputs, 
                                                                                                                                  preds)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
            
            # Process all input images
            total_indices = list(range(inputs.shape[0]))
            subsets = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for subset in subsets:
                num_rows = len(subset)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in subset:
                    img = (inputs[idx].cpu().detach().permute(1, 2, 0)).numpy()  
                    label = labels[idx].cpu().detach()
                    pred = preds[idx]
                    
                    results = [
                        np.transpose(saliency_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(guided_backprop_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(input_x_gradient_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(deconv_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                        np.transpose(occlusion_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                    ]
            
                    # Display the original image
                    axs[count, 0].imshow(img, cmap)
                    axs[count, 0].set_title(f"Actual: {class_dict[label.item()]}\nPredicted: {class_dict[pred.item()]}", fontsize=17)
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        if len(subset) > 1:
                            if idx == subset[0]:
                                check = 0
                            else:
                                check = 1
                        else:
                            check = 0
                        self.safe_visualize(res, title, fig, axs[count, col + 1], label, class_dict[label.item()], types, cmap, check)
            
                    count += 1
                    
                # Add a single colorbar for all subplots below the grid
                fig.subplots_adjust(bottom=0.15)
                cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
                norm = plt.Normalize(vmin=-1, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=17)  # Set colorbar tick label font size
            
                # Store the figure with an appropriate name
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "analysis_plots", "attribution_plots_classifier", f"{types}_subsets_{subsets.index(subset) + 1}"))
        return figs, names
