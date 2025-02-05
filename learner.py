import torch
from plots import *
from ccbdl.utils import DEVICE
from ccbdl.learning.gan import BaseGANLearning, BaseCGANLearning
from fid_custom import FrechetInceptionDistance
#from torcheval.metrics import FrechetInceptionDistance
from ccbdl.config_loader.loaders import ConfigurationLoader
from networks import CNN
import os
from ignite.metrics import PSNR, SSIM


class Learner(BaseGANLearning):
    def __init__(self,
                 trial_path: str,
                 model,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 network_config: dict,
                 logging):

        super().__init__(train_data, test_data, val_data, trial_path, learner_config,
                         data_storage_names=[
                             "epoch", "batch", "train_acc", "test_acc", "dis_loss", "gen_loss", "fid_score_nlrl", "fid_score_linear", "psnr_score", "ssim_score"],
                         task=task, logging=logging)

        self.model = model
        
        self.device = DEVICE
        print(self.device)
        
        self.figure_storage.dpi = 200
        
        def load_classifier():
            config = ConfigurationLoader().read_config("config.yaml")

            classifier_linear_config = config["classifier_linear"]
        
            classifier_linear = CNN(3,"Classifier to classify real and fake images", 
                          **classifier_linear_config)
            checkpoint_path_linear = os.path.join("Saved_networks", "cnn_net_best_linear.pt")

            classifier_nlrl_config = config["classifier_nlrl"]
        
            classifier_nlrl = CNN(3,"Classifier to classify real and fake images", 
                          **classifier_nlrl_config)
            checkpoint_path_nlrl = os.path.join("Saved_networks", "cnn_net_best_nlrl.pt")
            
            checkpoint_linear = torch.load(checkpoint_path_linear)
            classifier_linear.load_state_dict(checkpoint_linear['model_state_dict'])
            
            checkpoint_nlrl = torch.load(checkpoint_path_nlrl)
            classifier_nlrl.load_state_dict(checkpoint_nlrl['model_state_dict'])
            return classifier_nlrl, classifier_linear
        
        self.classifier_nlrl, self.classifier_linear = load_classifier()

        self.criterion_name = learner_config["criterion"]
        self.noise_dim = learner_config["noise_dim"]
        self.lr_exp = learner_config["learning_rate_exp"]
        self.lr_exp_l = learner_config["learning_rate_exp_l"]
        self.threshold = learner_config["threshold"]
        self.learner_config = learner_config
        self.network_config= network_config
        
        self.lr = 10**self.lr_exp
        self.lr_l = 10**self.lr_exp_l

        self.criterion = getattr(torch.nn, self.criterion_name)(reduction='mean').to(self.device)
        
        # Get the last layer's name
        last_layer_name_parts = list(self.model.discriminator.named_parameters())[-1][0].split('.')
        last_layer_name = last_layer_name_parts[0] + '.' + last_layer_name_parts[1]
        # print("Last layer name:", last_layer_name)
        
        # Separate out the parameters based on the last layer's name
        fc_params = [p for n, p in self.model.discriminator.named_parameters() if last_layer_name + '.' in n]  # Parameters of the last layer
        rest_params = [p for n, p in self.model.discriminator.named_parameters() if not last_layer_name + '.' in n]  # Parameters of layers before the last layer
        
        # print("FC Params:")
        # for p in fc_params:
        #     print(p.shape)
        # print("\nRest Params:")
        # for p in rest_params:
        #     print(p.shape)
        
        # print(self.model)

        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr / 2, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(rest_params, lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_fc = torch.optim.Adam(fc_params, lr=self.lr_l)

        self.train_data = train_data
        self.test_data = test_data

        self.result_folder = trial_path

        self.plotter.register_default_plot(TimePlot(self))
        self.plotter.register_default_plot(Image_generation(self))
        #self.plotter.register_default_plot(Image_generation_dataset(self))
        # self.plotter.register_default_plot(Tsne_plot_classifier(self))
        # self.plotter.register_default_plot(Attribution_plots(self))
        self.plotter.register_default_plot(Fid_plot_nlrl(self))
        self.plotter.register_default_plot(Fid_plot_linear(self))
        self.plotter.register_default_plot(Psnr_plot(self))
        self.plotter.register_default_plot(Ssim_plot(self))
        self.plotter.register_default_plot(Loss_plot(self))
        # self.plotter.register_default_plot(Attribution_plots_classifier(self))
        
        # if self.network_config["final_layer"] == 'nlrl':
        #     self.plotter.register_default_plot(Hist_plot(self))
        
        # self.plotter.register_default_plot(Softmax_plot_classifier(self))
        self.plotter.register_default_plot(Confusion_matrix_gan(self))
        # self.plotter.register_default_plot(Tsne_plot_dis_gan(self))

        self.parameter_storage.store(self)
        
        self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters:")
        self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters: ")
        
        self.fid_metric = FrechetInceptionDistance(model=self.classifier_nlrl, feature_dim=10, device=self.device)
        self.fid_metric_linear = FrechetInceptionDistance(model=self.classifier_linear, feature_dim=10, device=self.device)
        
        self.psnr_metric = PSNR(data_range=1.0, device=self.device)
        
        self.ssim_metric = SSIM(data_range=1.0, device=self.device)
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')

    def _train_epoch(self, train=True):
        if self.epoch == 0:
            torch.save({'epoch': self.epoch,
                        'batch': 0,
                        'model_state_dict': self.model.state_dict()},
                       self.initial_save_path)
        self.model.train()
        for i, data in enumerate(self.train_data):
            inputs, labels = data
            
            self.real_data = inputs.to(self.device)
            self.labels = labels.to(self.device).long()
            
            # min_value = self.real_data.min()
            # max_value = self.real_data.max()
            
            # print(f"Min value: {min_value.item()}")
            # print(f"Max value: {max_value.item()}")
            
            self.optimizer_D.zero_grad()
            self.optimizer_D_fc.zero_grad()

            # Train discriminator on real data
            real_target = torch.ones(len(inputs), device=self.device)

            predictions_real = self._discriminate(self.real_data)

            diss_real = self.criterion(predictions_real, real_target)

            # Train discriminator on fake data
            noise = torch.randn(self.real_data.size(0), self.noise_dim, device=self.device)

            fake = self._generate(noise)
                
            fake_target = torch.zeros(len(inputs), device=self.device)
            
            predictions_fake = self._discriminate(fake.detach())

            dis_fake = self.criterion(predictions_fake, fake_target)

            self.loss_disc = diss_real + dis_fake
            
            self.loss_disc.backward()
            self.optimizer_D.step()
            self.optimizer_D_fc.step()

            self.optimizer_G.zero_grad()

            real_acc = 100.0 * (predictions_real > self.threshold).sum() / real_target.shape[0]
            fake_acc = 100.0 * (predictions_fake < self.threshold).sum() / fake_target.shape[0]
            self.train_accuracy = (real_acc + fake_acc)/2

            output = self._discriminate(fake)
            # min_value = output.min()
            # max_value = output.max()
            
            # print(f"Min value: {min_value.item()}")
            # print(f"Max value: {max_value.item()}")

            self.loss_gen = self.criterion(output, real_target)
            
            self.loss_gen.backward()
            self.optimizer_G.step()        

            # Update the metric for real images and fake images of RGB
            self.fid_metric.update(self.real_data, is_real=True)
            self.fid_metric.update(fake.detach(), is_real=False)
            
            self.fid = self.fid_metric.compute()
            
            # Update the metric for real images and fake images of RGB
            self.fid_metric_linear.update(self.real_data, is_real=True)
            self.fid_metric_linear.update(fake.detach(), is_real=False)
            
            self.fid_linear = self.fid_metric_linear.compute()
            
            self.psnr_metric.reset()
            self.psnr_metric.update((fake.detach(), self.real_data))
            self.psnr = self.psnr_metric.compute()
            
            self.ssim_metric.reset()
            self.ssim_metric.update((fake.detach(), self.real_data))
            self.ssim = self.ssim_metric.compute()

            self.data_storage.store(
                [self.epoch, self.batch, self.train_accuracy, self.test_accuracy, self.loss_disc, 
                 self.loss_gen, self.fid, self.fid_linear,  self.psnr, self.ssim])

            if train:
                self.batch += 1
                if len(self.train_data) -1: 
                    self.data_storage.dump_store("predictions_real", predictions_real.detach().cpu().numpy())
                    self.data_storage.dump_store("predictions_fake", predictions_fake.detach().cpu().numpy())
                    #self.data_storage.dump_store("fake_images", fake.detach())
                    #self.data_storage.dump_store("real_images", self.real_data)
                    #self.data_storage.dump_store("labels", self.labels)
                if self.epoch == self.learner_config["num_epochs"] - 1:
                    self.data_storage.dump_store("fake_images_", fake.detach())
                    self.data_storage.dump_store("real_images_", self.real_data)
                    self.data_storage.dump_store("labels_", self.labels)

    def _test_epoch(self, test=True):
        self.model.eval()
        loss, samples, corrects = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                inputs, labels = data
                
                real_data = inputs.to(self.device)
                labels = labels.to(self.device).long()
                real_target = torch.ones(len(inputs), device=self.device)

                # Classify Images
                predictions = self._discriminate(real_data)
                # min_value = predictions.min()
                # max_value = predictions.max()
                
                # print(f"Min value: {min_value.item()}")
                # print(f"Max value: {max_value.item()}")

                loss += self.criterion(predictions, real_target).item()
                corrects += (predictions > self.threshold).sum()
                samples += inputs.size(0)
                
        self.test_loss = loss / len(self.test_data)
        self.test_accuracy = 100.0 * corrects / samples

    def _validate_epoch(self):
        pass

    def _generate(self, ins):
        return self.model.generator(ins)

    def _discriminate(self, ins):
        return self.model.discriminator(ins).float().squeeze()

    def _update_best(self):
        if self.fid < self.best_values["FidScore"]:
            self.best_values["GenLoss"] = self.loss_gen.item()
            self.best_values["DisLoss"] = self.loss_disc.item()
            self.best_values["FidScore"] = self.fid.item()
            self.best_values["FidScore_Linear"] = self.fid_linear.item()
            self.best_values["PsnrScore"] = self.psnr
            self.best_values["SsimScore"] = self.ssim
            self.best_values["Batch"] = self.batch

            self.other_best_values = {'testloss':      self.test_loss,
                                      "test_acc":      self.test_accuracy.item(),
                                      "train_acc":     self.train_accuracy.item(), }

            self.best_state_dict = self.model.state_dict()

    def evaluate(self):
        self.end_values = {"GenLoss":       self.loss_gen.item(),
                           "DisLoss":       self.loss_disc.item(),
                           "FidScore":      self.fid.item(),
                           "FidScore_Linear":      self.fid_linear.item(),
                           "PsnrScore": self.psnr,
                           "SsimScore": self.ssim,
                           'testloss':      self.test_loss,
                           "test_acc":      self.test_accuracy.item(),
                           "train_acc":     self.train_accuracy.item(),
                           "Batch":         self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"GenLoss":       self.loss_gen.item(),
                                "DisLoss":       self.loss_disc.item(),
                                "FidScore":      self.fid.item(),
                                "FidScore_Linear":      self.fid_linear.item(),
                                "PsnrScore": self.psnr,
                                "SsimScore": self.ssim,
                                'testloss':      self.test_loss,
                                "test_acc":      self.test_accuracy.item(),
                                "train_acc":     self.train_accuracy.item(),
                                "Batch":         self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'GenLoss': self.loss_gen.item(),
                                    'DisLoss': self.loss_disc.item(),
                                    'FidScore': self.fid.item(),
                                    'FidScore_Linear': self.fid_linear.item(),
                                    "PsnrScore": self.psnr,
                                    "SsimScore": self.ssim,
                                    'model_state_dict': self.model.state_dict()}
            self.test_noise = torch.randn(30, self.noise_dim, device=self.device)

        # Saving generated images
        with torch.no_grad(): 
            pos_epoch = self.epoch

            generated_images = self._generate(self.test_noise)
            predictions_gen = self._discriminate(generated_images.detach())

            self.data_storage.dump_store("generated_images", generated_images)
            self.data_storage.dump_store("epochs_gen", pos_epoch)
            self.data_storage.dump_store("predictions_gen", predictions_gen.detach().cpu().numpy())

    def _save(self):
        torch.save(self.init_state_dict, self.init_save_path)
        
        torch.save({'epoch': self.epoch,
                    'best_values': self.best_values,
                    'best_acc_loss_values': self.other_best_values,
                    'model_state_dict': self.best_state_dict},
                   self.best_save_path)

        torch.save({'epoch': self.epoch,
                    'batch': self.batch,
                    'GenLoss': self.loss_gen.item(),
                    'DisLoss': self.loss_disc.item(),
                    'FidScore': self.fid.item(),
                    'FidScore_Linear': self.fid_linear.item(),
                    "PsnrScore": self.psnr,
                    "SsimScore": self.ssim,
                    'model_state_dict': self.model.state_dict()},
                   self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.other_best_values, "best_acc_loss_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")
        # if self.best_values["FidScore"] <= 7.0:
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))
    
    def _load_initial(self):
        checkpoint = torch.load(self.initial_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator
    
    def _load_best(self):
        checkpoint = torch.load(self.best_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator


class Conditional_Learner(BaseCGANLearning):
    def __init__(self,
                  trial_path: str,
                  model,
                  train_data,
                  test_data,
                  val_data,
                  task,
                  learner_config: dict,
                  network_config: dict,
                  logging):

        super().__init__(train_data, test_data, val_data, trial_path, learner_config,
                         data_storage_names=[
                             "epoch", "batch", "train_acc", "test_acc", "dis_loss", "gen_loss", "fid_score_nlrl", "fid_score_linear", "psnr_score", "ssim_score"],
                         task=task, logging=logging)

        self.model = model
        
        self.device = DEVICE
        print(self.device)
        
        self.figure_storage.dpi = 200
        
        def load_classifier():
            config = ConfigurationLoader().read_config("config.yaml")

            classifier_linear_config = config["classifier_linear"]
        
            classifier_linear = CNN(3,"Classifier to classify real and fake images", 
                          **classifier_linear_config)
            checkpoint_path_linear = os.path.join("Saved_networks", "cnn_net_best_linear.pt")

            classifier_nlrl_config = config["classifier_nlrl"]
        
            classifier_nlrl = CNN(3,"Classifier to classify real and fake images", 
                          **classifier_nlrl_config)
            checkpoint_path_nlrl = os.path.join("Saved_networks", "cnn_net_best_nlrl.pt")
            
            checkpoint_linear = torch.load(checkpoint_path_linear)
            classifier_linear.load_state_dict(checkpoint_linear['model_state_dict'])
            
            checkpoint_nlrl = torch.load(checkpoint_path_nlrl)
            classifier_nlrl.load_state_dict(checkpoint_nlrl['model_state_dict'])
            return classifier_nlrl, classifier_linear
        
        self.classifier_nlrl, self.classifier_linear = load_classifier()

        self.criterion_name = learner_config["criterion"]
        self.classification_criterion_name = learner_config["classification_criterion"]
        self.noise_dim = learner_config["noise_dim"]
        self.threshold = learner_config["threshold"]
        self.lr_exp = learner_config["learning_rate_exp"]
        self.lr_exp_l = learner_config["learning_rate_exp_l"]
        self.learner_config = learner_config
        self.network_config = network_config
        
        self.lr = 10**self.lr_exp
        self.lr_l = 10**self.lr_exp_l

        self.criterion = getattr(torch.nn, self.criterion_name)(reduction='mean').to(self.device)
        self.classification_criterion = getattr(torch.nn, self.classification_criterion_name)().to(self.device)

        # Get the last layer's name
        last_layer_name_parts = list(self.model.discriminator.named_parameters())[-1][0].split('.')
        last_layer_name = last_layer_name_parts[0] + '.' + last_layer_name_parts[1]
        print("Last layer name:", last_layer_name)
        
        # Separate out the parameters based on the last layer's name
        fc_params = [p for n, p in self.model.discriminator.named_parameters() if last_layer_name + '.' in n]  # Parameters of the last layer
        rest_params = [p for n, p in self.model.discriminator.named_parameters() if not last_layer_name + '.' in n]  # Parameters of layers before the last layer
        
        print("FC Params:")
        for p in fc_params:
            print(p.shape)
        print("\nRest Params:")
        for p in rest_params:
            print(p.shape)
        
        print(self.model)

        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr / 2, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(rest_params, lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_fc = torch.optim.Adam(fc_params, lr=self.lr_l, betas=(0.5, 0.999))

        self.train_data = train_data
        self.test_data = test_data

        self.result_folder = trial_path

        self.plotter.register_default_plot(TimePlot(self))               
        self.plotter.register_default_plot(Image_generation(self))
        #self.plotter.register_default_plot(Image_generation_dataset(self))       
        # self.plotter.register_default_plot(Attribution_plots_conditional(self))
        self.plotter.register_default_plot(Fid_plot_nlrl(self))
        self.plotter.register_default_plot(Fid_plot_linear(self))
        self.plotter.register_default_plot(Psnr_plot(self))
        self.plotter.register_default_plot(Ssim_plot(self))
        self.plotter.register_default_plot(Confusion_matrix_gan(self))
        # self.plotter.register_default_plot(Attribution_plots_classifier_conditional(self))
        
        # if self.network_config["final_layer"] == 'nlrl':
        #     self.plotter.register_default_plot(Hist_plot(self))
            
        # self.plotter.register_default_plot(Tsne_plot_images_separate(self))
        # self.plotter.register_default_plot(Tsne_plot_images_combined(self))  
        # self.plotter.register_default_plot(Tsne_plot_classifier(self))
        # self.plotter.register_default_plot(Tsne_plot_dis_cgan(self))      
        # self.plotter.register_default_plot(Softmax_plot_classifier_conditional(self))
        self.plotter.register_default_plot(Loss_plot(self))
        # self.plotter.register_default_plot(Confusion_matrix_classifier(self, **{"ticks": torch.arange(0, 10, 1).numpy()}))

        self.parameter_storage.store(self)
        
        self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters:")
        self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters: ")
        
        self.fid_metric = FrechetInceptionDistance(model=self.classifier_nlrl, feature_dim=10, device=self.device)
        self.fid_metric_linear = FrechetInceptionDistance(model=self.classifier_linear, feature_dim=10, device=self.device)
        
        self.psnr_metric = PSNR(data_range=1.0, device=self.device)
        
        self.ssim_metric = SSIM(data_range=1.0, device=self.device)
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')

    def _train_epoch(self, train=True):
        if self.epoch == 0:
            torch.save({'epoch': self.epoch,
                        'batch': 0,
                        'model_state_dict': self.model.state_dict()},
                       self.initial_save_path)
        self.model.train()
        for i, data in enumerate(self.train_data):
            inputs, labels = data
            
            self.real_data = inputs.to(self.device)
            self.labels = labels.to(self.device).long()
            
            self.optimizer_D.zero_grad()
            self.optimizer_D_fc.zero_grad()

            # Train discriminator on real data
            real_target = torch.ones(len(inputs), device=self.device)

            predictions_real = self._discriminate(self.real_data, self.labels)

            diss_real = self.criterion(predictions_real, real_target)

            # Train discriminator on fake data
            noise = torch.randn(self.real_data.size(0), self.noise_dim, device=self.device)

            fake = self._generate(noise, self.labels)
                
            fake_target = torch.zeros(len(inputs), device=self.device)

            predictions_fake = self._discriminate(fake.detach(), self.labels)

            dis_fake = self.criterion(predictions_fake, fake_target)

            self.loss_disc = diss_real + dis_fake
            
            self.loss_disc.backward()
            self.optimizer_D.step()
            self.optimizer_D_fc.step()

            self.optimizer_G.zero_grad()

            real_acc = 100.0 * (predictions_real > self.threshold).sum() / real_target.shape[0]
            fake_acc = 100.0 * (predictions_fake < self.threshold).sum() / fake_target.shape[0]
            self.train_accuracy = (real_acc + fake_acc)/2

            # Fooling the discriminator with fake
            output = self._discriminate(fake, self.labels)
            
            if self.learner_config["cnn_feed_back"] != "Yes":
                self.loss_gen = self.criterion(output, real_target)
            else:
                class_output = self._classify(fake)
                self.loss_gen = self.criterion(output, real_target) + self.classification_criterion(class_output, self.labels)

            self.loss_gen.backward()
            self.optimizer_G.step()

            # Update the metric for real images and fake images of RGB
            self.fid_metric.update(self.real_data, is_real=True)
            self.fid_metric.update(fake.detach(), is_real=False)
            
            self.fid = self.fid_metric.compute()
            
            # Update the metric for real images and fake images of RGB
            self.fid_metric_linear.update(self.real_data, is_real=True)
            self.fid_metric_linear.update(fake.detach(), is_real=False)
            
            self.fid_linear = self.fid_metric_linear.compute()
            
            self.psnr_metric.reset()
            self.psnr_metric.update((fake.detach(), self.real_data))
            self.psnr = self.psnr_metric.compute()
            
            self.ssim_metric.reset()
            self.ssim_metric.update((fake.detach(), self.real_data))
            self.ssim = self.ssim_metric.compute()

            self.data_storage.store(
                [self.epoch, self.batch, self.train_accuracy, self.test_accuracy, self.loss_disc, 
                 self.loss_gen, self.fid, self.fid_linear,  self.psnr, self.ssim])
            
            if train:
                self.batch += 1
                if len(self.train_data) -1: 
                    self.data_storage.dump_store("predictions_real", predictions_real.detach().cpu().numpy())
                    self.data_storage.dump_store("predictions_fake", predictions_fake.detach().cpu().numpy())
                    #self.data_storage.dump_store("fake_images", fake.detach())
                    #self.data_storage.dump_store("real_images", self.real_data)
                    #self.data_storage.dump_store("labels", self.labels)
                if self.epoch == self.learner_config["num_epochs"] - 1:
                    self.data_storage.dump_store("fake_images_", fake.detach())
                    self.data_storage.dump_store("real_images_", self.real_data)
                    self.data_storage.dump_store("labels_", self.labels)

    def _test_epoch(self, test=True):        
        self.model.eval()
        loss, samples, corrects = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                inputs, labels = data
                
                real_data = inputs.to(self.device)
                labels = labels.to(self.device).long()
                real_target = torch.ones(len(inputs), device=self.device)

                # Classify Images
                predictions = self._discriminate(real_data, labels)

                loss += self.criterion(predictions, real_target).item()
                corrects += (predictions > self.threshold).sum()
                samples += inputs.size(0)
                
        self.test_loss = loss / len(self.test_data)
        self.test_accuracy = 100.0 * corrects / samples

    def _validate_epoch(self):
        pass

    def _generate(self, ins, labels):
        return self.model.generator(ins, labels)

    def _discriminate(self, ins, labels):
        return self.model.discriminator(ins, labels).float().squeeze()
    
    def _classify(self, ins):
       	if self.learner_config["layer"] == "nlrl":
       		classifier = self.classifier_nlrl
       	else:
       		classifier = self.classifier_linear
        return classifier(ins)

    def _update_best(self):
        if self.fid < self.best_values["FidScore"]:
            self.best_values["GenLoss"] = self.loss_gen.item()
            self.best_values["DisLoss"] = self.loss_disc.item()
            self.best_values["FidScore"] = self.fid.item()
            self.best_values["FidScore_Linear"] = self.fid_linear.item()
            self.best_values["PsnrScore"] = self.psnr
            self.best_values["SsimScore"] = self.ssim
            self.best_values["Batch"] = self.batch

            self.other_best_values = {'testloss':      self.test_loss,
                                      "test_acc":      self.test_accuracy.item(),
                                      "train_acc":     self.train_accuracy.item(), }

            self.best_state_dict = self.model.state_dict()

    def evaluate(self):
        self.end_values = {"GenLoss":       self.loss_gen.item(),
                           "DisLoss":       self.loss_disc.item(),
                           "FidScore":      self.fid.item(),
                           "FidScore_Linear":      self.fid_linear.item(),
                           "PsnrScore": self.psnr,
                           "SsimScore": self.ssim,
                           'testloss':      self.test_loss,
                           "test_acc":      self.test_accuracy.item(),
                           "train_acc":     self.train_accuracy.item(),
                           "Batch":         self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"GenLoss":       self.loss_gen.item(),
                                "DisLoss":       self.loss_disc.item(),
                                "FidScore":      self.fid.item(),
                                "FidScore_Linear":      self.fid_linear.item(),
                                "PsnrScore": self.psnr,
                                "SsimScore": self.ssim,
                                'testloss':      self.test_loss,
                                "test_acc":      self.test_accuracy.item(),
                                "train_acc":     self.train_accuracy.item(),
                                "Batch":         self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'GenLoss': self.loss_gen.item(),
                                    'DisLoss': self.loss_disc.item(),
                                    'FidScore': self.fid.item(),
                                    'FidScore_Linear': self.fid_linear.item(),
                                    "PsnrScore": self.psnr,
                                    "SsimScore": self.ssim,
                                    'model_state_dict': self.model.state_dict()}
            
            self.test_noise = torch.randn(30, self.noise_dim, device=self.device)
            self.test_labels = torch.tensor([i % 10 for i in range(30)], device=self.device)

        # Saving generated images
        with torch.no_grad():         
            pos_epoch = self.epoch

            generated_images = self._generate(self.test_noise, self.test_labels)
            predictions_gen = self._discriminate(generated_images.detach(), self.test_labels)
            
            self.data_storage.dump_store("generated_images", generated_images)
            self.data_storage.dump_store("epochs_gen", pos_epoch)
            self.data_storage.dump_store("predictions_gen", predictions_gen.detach().cpu().numpy())
            self.data_storage.dump_store("labels_gen", self.test_labels)

    def _save(self):
        torch.save(self.init_state_dict, self.init_save_path)
        
        torch.save({'epoch': self.epoch,
                    'best_values': self.best_values,
                    'best_acc_loss_values': self.other_best_values,
                    'model_state_dict': self.best_state_dict},
                    self.best_save_path)

        torch.save({'epoch': self.epoch,
                    'batch': self.batch,
                    'GenLoss': self.loss_gen.item(),
                    'DisLoss': self.loss_disc.item(),
                    'FidScore': self.fid.item(),
                    'FidScore_Linear': self.fid_linear.item(),
                    "PsnrScore": self.psnr,
                    "SsimScore": self.ssim,
                    'model_state_dict': self.model.state_dict()},
                    self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.other_best_values, "best_acc_loss_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")
        # if self.best_values["FidScore"] <= 0.8:
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))
    
    def _load_initial(self):
        checkpoint = torch.load(self.initial_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator
    
    def _load_best(self):
        checkpoint = torch.load(self.best_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator

