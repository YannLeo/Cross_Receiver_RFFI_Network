from collections import Counter
from copy import deepcopy
import itertools
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from torch.utils import tensorboard
import datasets
import models


class MinePseudoClassWeightTrainer:
    """
    > It is the model of our paper, which is a combination of MINE, pseudo labeling and class weight.

      The most important part are the functions train_epoch() and test_epoch(), which are called by function 
      train(). Just ignore the basic variables, loggers and helper functions stared with '__' if they bother you. 

    """

    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        """
        > Initialize the model, optimizer, scheduler, dataloader, and logger.

        :param info: dict of configs from toml file
        :param resume: path to checkpoint
        :param path: the path to the folder where the model will be saved
        :param device: the device to run the model on

        ---
        Tips: This function is highly dependent on the config file, where the data, models, optimizers and 
              schedulers are defined.              
        """
        # Basic variables
        self.info = info  # dict of configs from toml file
        self.resume = resume  # path to checkpoint
        self.device = device
        self.max_epoch = info['epochs']
        self.num_classes = info['model']['args']['num_classes']
        self.log_path = path / 'log' / 'log.txt'
        self.model_path = path / 'model'
        self.confusion_path = path / 'confusion'
        self.save_period = info['save_period']
        self.min_valid_loss = np.inf
        self.min_valid_pretrain_loss = np.inf
        # The prior distribution of the labels on the **source** domain
        self.distribution_prior = torch.tensor(info.get("distribution_prior", torch.ones(self.num_classes)))
        self.distribution_prior = self.distribution_prior / sum(self.distribution_prior)
        # Dataloaders
        self.__prepare_dataloaders(info)
        # Defination of models
        self.model = self.__get_object(models, info['model']['name'], info['model']['args'])
        self.mine = models.MINE()
        # Optimizers and schedulers
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.opt_MINE = torch.optim.Adam(params=self.mine.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self.__get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                              {'optimizer': self.opt, **info['lr_scheduler']['args']})
        self.lr_scheduler_MINE = self.__get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                                   {'optimizer': self.opt_MINE, **info['lr_scheduler']['args']})
        # Prepare for resuming models
        self.__resuming_models()
        self.model = self.model.to(self.device)
        self.mine = self.mine.to(self.device)
        # loggers
        self.__get_logger()  # txt logger
        self.metric_writer = tensorboard.SummaryWriter(path / 'log')  # tensorboard logger

    def __prepare_dataloaders(self, info):
        self.dataset_source = self.__get_object(datasets, info['dataloader_source']['dataset']['name'],
                                                info['dataloader_source']['dataset']['args'])
        self.dataset_target = self.__get_object(datasets, info['dataloader_target']['dataset']['name'],
                                                info['dataloader_target']['dataset']['args'])
        self.dataset_valid = self.__get_object(datasets, info['dataloader_valid']['dataset']['name'],
                                               info['dataloader_valid']['dataset']['args'])
        self.dataset_test = self.__get_object(datasets, info['dataloader_test']['dataset']['name'],
                                              info['dataloader_test']['dataset']['args'])
        self.dataloader_source = torch.utils.data.DataLoader(dataset=self.dataset_source,
                                                             **info['dataloader_source']['args'])
        self.dataloader_target = torch.utils.data.DataLoader(dataset=self.dataset_target,
                                                             **info['dataloader_target']['args'])
        self.dataloader_valid = torch.utils.data.DataLoader(dataset=self.dataset_valid,
                                                            **info['dataloader_valid']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test,
                                                           **info['dataloader_test']['args'])

    def __resuming_models(self):
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict['model'])
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0

    def __reset_grad(self):
        self.opt.zero_grad()
        self.opt_MINE.zero_grad()

    def train(self):  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        """
        self.batch_size = min(self.dataloader_source.batch_size, self.dataloader_target.batch_size)
        self.num_train = len(self.dataset_source)
        self.num_target = len(self.dataset_target)
        self.num_test = len(self.dataset_test)
        self.num_train_batch = min(self.num_train // self.dataloader_source.batch_size,
                                   self.num_target // self.dataloader_target.batch_size)
        self.num_test_batch = self.num_test // self.dataloader_test.batch_size
        self.train_display = 1 if self.num_train_batch < 10 else self.num_train_batch // 10
        self.test_display = 1 if self.num_test_batch < 10 else self.num_test_batch // 10
        begin_epoch = self.epoch
        num_pseudo, num_correct = 0, 0

        for epoch in range(begin_epoch, self.max_epoch):
            time_begin = time.time()
            print(f'epoch: {epoch + 1}\t| ', end='')

            '''Training epoch'''
            train_class_loss, train_kl_loss, train_acc, train_predicts, train_labels, pseudo_labels, predicted_labels = self.train_epoch(
                epoch+1)
            print(f'train_loss: {train_class_loss:.6f} | train_kl_loss: {train_kl_loss:.6f} | train_acc: {train_acc:6f} | ', end='')
            print('testing...' + '\b' * len('testing...'), end='', flush=True)

            num_pseudo, num_correct, num_pseudo_list, num_correct_list, num_list = self.get_pseudo_statistics(pseudo_labels)
            pred_counter = Counter(predicted_labels.tolist())
            pred_counter.pop(-1)
            pred_distribution = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                if pred_counter[i] == 0:
                    pred_distribution[i] = -1
                else:
                    pred_distribution[i] = (1 / self.num_classes) / (pred_counter[i] / sum(pred_counter.values()))

            print(f'pseudo_rate: {num_pseudo_list / num_list} | correct_pseudo_rate: {num_correct_list / num_list} | '
                  f'pred_distribution: {pred_distribution} | pred_counter: {pred_counter} | ')

            '''Testing epoch'''
            test_class_loss, test_acc, test_predicts, test_labels = self.test_epoch(epoch+1)
            time_end = time.time()
            print(
                f'test_class_loss: {test_class_loss:6f} | test_acc: {test_acc:6f} | pseudo_rate: {num_pseudo/self.num_target:.6f} | '
                f'pseudo_correct: {num_correct/(num_pseudo+1e-5):.6f} | time: {int(time_end - time_begin)}s', end='')

            '''Logging results'''
            best = self.__save_model_by_valid_loss(epoch + 1, test_class_loss)
            self.metric_writer.add_scalar("test_acc", test_acc, epoch)
            self.logger.info(
                f'epoch: {epoch + 1}\t| train_class_loss: {train_class_loss:.6f} | train_kl_loss: {train_kl_loss:.6f} | '
                f'train_acc: {train_acc:.6f} | pseudo_rate: {num_pseudo/self.num_target:.6f} | '
                f'pseudo_correct: {num_correct/(num_pseudo+1e-5):.6f} | test_class_loss: {test_class_loss:.6f} | '
                f'test_acc: {test_acc:.6f}{" | saving best model..." if best else ""}')
            self.epoch += 1

            if 'confusion' in self.info:
                if self.info['confusion'].get('train', False):
                    self.__plot_confusion_matrix(photo_path=self.confusion_path / f'train-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png',
                                                 labels=train_labels, predicts=train_predicts, classes=list(range(self.num_classes)), normalize=True)

                if self.info['confusion'].get('test', False):
                    self.__plot_confusion_matrix(photo_path=self.confusion_path / f'test-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png',
                                                 labels=test_labels, predicts=test_predicts, classes=list(range(self.num_classes)), normalize=True)
        # End of training
        self.metric_writer.close()
        
    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        Main training process
        """
        train_loss, train_kl_loss = 0, 0
        train_acc_num = 0
        train_num = 0
        predict, labels = [], []
        pseudo_labels = torch.ones(len(self.dataset_target)).long() * (-1)
        predicted_labels = torch.ones(len(self.dataset_target)).long() * (-1)

        self.model.train()
        self.mine.train()
        for batch, data_pack in enumerate(zip(self.dataloader_source, self.dataloader_target)):
            # `index_t` records the index of target data in the whole dataset, for pseudo labeling.
            # We do not use label_t here.
            (data_s, label_s, _), (data_t, _, index_t) = data_pack
            data_s, data_t, label_s = data_s.to(self.device), data_t.to(self.device), label_s.to(self.device)

            self.__reset_grad()
            output_s, feat_s = self.model(data_s)
            output_t, feat_t = self.model(data_t)

            """1. MINE"""
            if self.info["use_mine"] and (epoch > self.info["start_epoch"]):
                for _ in range(self.info['mine_steps']):
                    kl, ma_et, loss_kl = self.learn_KL(feat_s.detach(), feat_t.detach())
                    # gp = self.compute_gradient_penalty(feat_s.detach(), feat_t.detach())
                    self.__reset_grad()
                    loss = - 0.5 * loss_kl  # + 1.0 * gp
                    loss.backward()
                    self.opt_MINE.step()

                *_, kl_loss = self.learn_KL(feat_s, feat_t, ma_et)
                kl_loss = self.info["kl_weight"] * kl_loss
            else:
                kl_loss = torch.tensor(0)

            """2. on target domain"""
            predicted_labels = self.set_pred_labels(output_t, predicted_labels, index_t)
            if (self.info['use_class_weight']) and (epoch > self.info["start_epoch"]):
                class_weight = self.get_class_weight(predicted_labels)
            else:
                class_weight = torch.ones(self.num_classes, dtype=torch.float32).to(self.device)

            loss_content_tgt = torch.tensor(0.0)
            if self.info["use_pseudo"] and (epoch > self.info["start_epoch"]):
                output_selected, label_selected, pseudo_labels = \
                    self.set_pseudo_labels(output_t, pseudo_labels, index_t, self.info["pseudo_threshold"])
                if len(output_selected) > 0:
                    loss_content_tgt = nn.CrossEntropyLoss(weight=class_weight)(output_selected, label_selected) * 0.5

            """3. on source domain"""
            if (self.info['use_class_weight']) and (epoch > self.info["start_epoch"]):
                loss_content_src = nn.CrossEntropyLoss(weight=class_weight)(output_s, label_s) * 0.5
            else:  # pretraining
                loss_content_src = nn.CrossEntropyLoss(label_smoothing=self.info["label_smoothing"])(output_s, label_s)

            """4. total loss"""
            loss = loss_content_src + loss_content_tgt + kl_loss

            self.__reset_grad()
            loss.backward()
            self.opt.step()
            # logging the learning rate
            self.metric_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], epoch*self.num_train_batch+batch)

            predict.append(torch.argmax(output_s, dim=1).cpu().detach().numpy())
            labels.append(label_s.cpu().detach().numpy())
            train_num += data_s.shape[0]
            train_loss += loss.item()
            train_kl_loss += kl_loss.item()
            train_acc_num += np.sum(predict[-1] == labels[-1])

            self.lr_scheduler.step()
            self.lr_scheduler_MINE.step()

            if batch % self.train_display == 0:
                print('training... batch: {}/{} | total_loss: {:6f} | kl_loss: {:6f}'.format(batch, self.num_train_batch, loss.item(), kl_loss.item()) +
                      '\b' * len('training... batch: {}/{} | total_loss: {:6f} | kl_loss: {:6f}'.format(batch, self.num_train_batch, loss.item(), kl_loss.item())), end='', flush=True)

        predict = np.concatenate(predict, axis=0)
        labels = np.concatenate(labels, axis=0)
        return train_loss / self.num_train_batch, train_kl_loss / self.num_train_batch, train_acc_num / train_num, \
            predict, labels, pseudo_labels, predicted_labels

    def test_epoch(self, epoch):
        # on target domain
        test_class_loss = 0
        test_acc_num = 0
        test_num = 0
        predict, label = [], []
        self.model.eval()
        self.mine.eval()
        with torch.no_grad():
            for data, targets, _ in self.dataloader_test:
                data, targets = data.to(self.device), targets.to(self.device)
                label.append(targets.cpu().detach().numpy())
                test_num += data.shape[0]

                output, _ = self.model(data)

                loss = nn.CrossEntropyLoss()(output, targets)
                predict.append(torch.argmax(output, dim=1).cpu().detach().numpy())
                test_acc_num += torch.sum(torch.argmax(output, dim=1) == targets).item()
                test_class_loss += loss.item()

        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return test_class_loss / self.num_test_batch, test_acc_num / test_num, predict, label

    def set_pseudo_labels(self, output_t, pseudo_labels, index_t, threshold):
        pseudo_counter = Counter(pseudo_labels.tolist())
        pseudo_t = torch.argmax(output_t, dim=1)
        max_prob = torch.max(output_t, dim=1)[0]

        if max(pseudo_counter.values()) < len(self.dataloader_target):
            classwise_acc = torch.zeros((self.num_classes)).to(self.device)
            wo_negative_one = deepcopy(pseudo_counter)
            if -1 in wo_negative_one.keys():
                wo_negative_one.pop(-1)
            for i in range(self.num_classes):
                classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            idx = torch.where(max_prob > threshold * (classwise_acc[pseudo_t] / (2. - classwise_acc[pseudo_t])))[0]
        else:
            idx = torch.where(max_prob > threshold)[0]

        label_selected = pseudo_t[idx].to(self.device)
        output_selected = output_t[idx]
        pseudo_labels[index_t[idx.cpu()]] = pseudo_t[idx].cpu()

        return output_selected, label_selected, pseudo_labels

    def get_pseudo_statistics(self, pseudo_labels):
        num_pseudo = len(torch.where(pseudo_labels >= 0)[0])
        num_correct = len(torch.where(pseudo_labels == self.dataset_target.targets)[0])
        num_pseudo_list = [0] * self.num_classes
        num_correct_list = [0] * self.num_classes
        num_list = [0] * self.num_classes

        for i in range(self.num_classes):
            num_pseudo_list[i] = torch.sum(pseudo_labels == i).item()
            num_correct_list[i] = len(torch.where((pseudo_labels == i) & (self.dataset_target.targets == i))[0])
            num_list[i] = torch.sum(self.dataset_target.targets == i).item()

        return num_pseudo, num_correct, np.array(num_pseudo_list, dtype=np.float32), np.array(num_correct_list, dtype=np.float32), np.array(num_list, dtype=np.float32)+1e-5

    def set_pred_labels(self, output, predicted_labels, index):
        predicted_labels[index] = torch.argmax(output, dim=1).cpu()  # tag the prediction on the entire target domain
        return predicted_labels

    def get_class_weight(self, predicted_labels):
        distribution_pred = Counter(predicted_labels.tolist())
        distribution_pred.pop(-1)
        class_weight = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            if distribution_pred[i] == 0:
                class_weight[i] = 1
            else:
                class_weight[i] = self.distribution_prior[i] / (distribution_pred[i] / sum(distribution_pred.values()))
        return class_weight.to(self.device)

    def estimate_KL_divergence(self, p, q):
        t = self.mine(p)
        et = torch.exp(self.mine(q))
        kl = torch.mean(t) - torch.log(torch.mean(et)+1e-4)
        return kl, t, et

    def learn_KL(self, p, q, ma_et=1.0, ma_rate=0.01):
        kl, t, et = self.estimate_KL_divergence(p, q)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
        loss = torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et)
        return kl, ma_et, loss

    def compute_gradient_penalty(self, domain_0, domain_1):
        """Calculates the gradient penalty loss for WGAN GP
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L119
        """
        # Get random interpolation between 2 different domains
        alpha = torch.rand(domain_0.shape[0], 1).cuda()  # Random weight of size (B, 1)
        interpolates = (alpha * domain_0 + ((1 - alpha) * domain_1)).requires_grad_(True)  # auto broadcasting
        # out_interpolates = Discriminator(interpolates)
        # TODO: check if this is correct
        *_, out_interpolates = self.learn_KL(interpolates, domain_1, 1.0)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=out_interpolates,  # (B,)
            inputs=interpolates,  # (B, ...)
            grad_outputs=torch.ones_like(out_interpolates).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # scaler; dim=1 means by row

    def __plot_confusion_matrix(self, photo_path, labels, predicts, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
        FONT_SIZE = 9
        cm = confusion_matrix(labels, predicts, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        plt.figure(figsize=(8*2, 6*2))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, classes, fontsize=FONT_SIZE)
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE+3,
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(photo_path)

    def __get_object(self, module, s: str, parameter: dict):
        return getattr(module, s)(**parameter)

    def __get_logger(self):
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info(f'model: {type(self.model).__name__}')

    def __save_model_by_valid_loss(self, epoch, valid_loss):
        flag = 0
        if valid_loss < self.min_valid_loss:
            flag = 1
            self.min_valid_loss = valid_loss
            if epoch % self.save_period == 0:
                print(' | saving best model and checkpoint...')
                self.__save_checkpoint(epoch, True)
                self.__save_checkpoint(epoch, False)
            else:
                print(' | saving best model...')
                self.__save_checkpoint(epoch, True)
        elif epoch % self.save_period == 0:
            print(' | saving checkpoint...')
            self.__save_checkpoint(epoch, False)
        else:
            print()
        return flag

    def __save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'loss_best': self.min_valid_pretrain_loss,
        }
        if save_best:
            best_path = str(self.model_path / ('model_best.pth'))
            torch.save(state, best_path)
        else:
            path = str(self.model_path / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, path)
