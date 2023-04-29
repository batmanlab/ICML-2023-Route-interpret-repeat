import os.path
import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utils import *
import utils


class Logger:
    """
    This class creates manages different parameters based on each run.
    """

    def __init__(self, checkpoint_path, tb_path, output_path, train_loader, val_loader, n_classes):
        """
        Initialized each parameters of each run.
        """
        self.checkpoint_path = checkpoint_path
        self.tb_path = tb_path
        self.output_path = output_path
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epoch_id = 0
        self.best_epoch_id = 0
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0
        self.epoch_id_total_train_correct = 0
        self.epoch_id_total_val_correct = 0
        self.best_val_accuracy = 0
        self.epoch_start_time = None
        self.best_val_loss = 1000000

        self.run_params = None
        self.run_id = 0
        self.run_data = []
        self.run_start_time = None
        self.epoch_duration = None

        self.tb = None
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.n_classes = n_classes

        self.n_attributes = 0

        # for explainer g
        self.best_val_g_accuracy = 0
        self.val_g_accuracy = 0
        self.val_g_incorrect_accuracy = 0
        self.epoch_id_total_val_g_correct = 0

        self.epoch_train_selective_loss = 0
        self.epoch_train_emp_coverage = 0
        self.epoch_train_distillation_risk = 0
        self.epoch_train_CE_risk = 0
        self.epoch_train_KD_risk = 0
        self.epoch_train_entropy_risk = 0
        self.epoch_train_emp_risk = 0
        self.epoch_train_cov_penalty = 0
        self.epoch_train_aux_loss = 0

        self.train_selective_loss = 0
        self.train_emp_coverage = 0
        self.train_distillation_risk = 0
        self.train_CE_risk = 0
        self.train_KD_risk = 0
        self.train_entropy_risk = 0
        self.train_emp_risk = 0
        self.train_cov_penalty = 0
        self.train_aux_loss = 0

        self.epoch_val_selective_loss = 0
        self.epoch_val_emp_coverage = 0
        self.epoch_val_distillation_risk = 0
        self.epoch_val_CE_risk = 0
        self.epoch_val_KD_risk = 0
        self.epoch_val_entropy_risk = 0
        self.epoch_val_emp_risk = 0
        self.epoch_val_cov_penalty = 0
        self.epoch_val_aux_loss = 0

        self.val_selective_loss = 0
        self.val_emp_coverage = 0
        self.val_distillation_risk = 0
        self.val_CE_risk = 0
        self.val_KD_risk = 0
        self.val_entropy_risk = 0
        self.val_emp_risk = 0
        self.val_cov_penalty = 0
        self.val_aux_loss = 0

        # for tracking pi
        self.val_out_put_sel_proba = None
        self.val_out_put_class = None
        self.val_out_put_target = None

        self.acc_selected = 0
        self.acc_rejected = 0
        self.n_rejected = 0
        self.n_selected = 0
        self.coverage = 0

        # For saving in csv file
        self.arr_epoch_id = []
        self.arr_epoch_duration = []
        self.arr_total_train_loss = []
        self.arr_total_val_loss = []
        self.arr_train_acc = []
        self.arr_val_acc = []

        self.arr_train_emp_coverage = []
        self.arr_train_distillation_risk = []
        self.arr_train_CE_risk = []
        self.arr_train_KD_risk = []
        self.arr_train_entropy_risk = []
        self.arr_train_emp_risk = []
        self.arr_train_cov_penalty = []
        self.arr_train_selective_loss = []
        self.arr_train_aux_loss = []

        self.arr_val_emp_coverage = []
        self.arr_val_distillation_risk = []
        self.arr_val_CE_risk = []
        self.arr_val_KD_risk = []
        self.arr_val_entropy_risk = []
        self.arr_val_emp_risk = []
        self.arr_val_cov_penalty = []
        self.arr_val_selective_loss = []
        self.arr_val_aux_loss = []

        self.arr_val_g_accuracy = []
        self.arr_val_g_incorrect_accuracy = []
        self.arr_n_selected = []
        self.arr_n_rejected = []
        self.arr_coverage = []

        self.arr_best_epoch_id = []
        self.arr_best_val_g_acc = []
        self.arr_best_val_acc = []

        self.performance_dict = {}

    def set_n_attributes(self, n_attributes):
        # for multilabel classification
        self.n_attributes = n_attributes

    def begin_run(self, run):
        """
        Records all the parameters at the start of each run.

        :param run:
        :param network: cnn model
        :param loader: pytorch data loader
        :param device: {cpu or gpu}
        :param type_of_bn: whether {batch normalization, no batch normalization or dropout}

        :return: none
        """
        self.run_start_time = time.time()

        self.run_id += 1
        self.run_params = run
        self.tb = SummaryWriter(f"{self.tb_path}/{run}")

    def end_run(self):
        """
        Records all the parameters at the end of each run.

        :return: none
        """
        self.tb.close()
        self.epoch_id = 0

    def begin_epoch(self):
        """
        Records all the parameters at the start of each epoch.

        :return: none
        """
        self.epoch_start_time = time.time()
        self.epoch_id += 1
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0
        self.epoch_id_total_train_correct = 0
        self.epoch_id_total_val_correct = 0

        self.acc_selected = 0
        self.acc_rejected = 0
        self.n_selected = 0
        self.n_rejected = 0

        # for explainer g
        self.train_selective_loss = 0
        self.train_emp_coverage = 0
        self.train_distillation_risk = 0
        self.train_CE_risk = 0
        self.train_KD_risk = 0
        self.train_entropy_risk = 0
        self.train_emp_risk = 0
        self.train_cov_penalty = 0

        self.epoch_id_total_val_g_correct = 0
        self.val_selective_loss = 0
        self.val_emp_coverage = 0
        self.val_distillation_risk = 0
        self.val_CE_risk = 0
        self.val_KD_risk = 0
        self.val_entropy_risk = 0
        self.val_emp_risk = 0
        self.val_cov_penalty = 0

        self.val_out_put_sel_proba = torch.FloatTensor().cuda()
        self.val_out_put_class = torch.FloatTensor().cuda()
        self.val_out_put_target = torch.FloatTensor().cuda()

    def evaluate_g_correctly(self, selection_threshold):
        prediction_result = self.val_out_put_class.argmax(dim=1)
        t = self.val_out_put_target.detach()
        selection_result = None
        if self.val_out_put_sel_proba is not None:
            condition = self.val_out_put_sel_proba >= selection_threshold
            selection_result = torch.where(
                condition,
                torch.ones_like(self.val_out_put_sel_proba),
                torch.zeros_like(self.val_out_put_sel_proba),
            ).view(-1)

        h_rjc = torch.masked_select(prediction_result, selection_result.bool())
        t_rjc = torch.masked_select(self.val_out_put_target, selection_result.bool())
        t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())

        acc = float(t / (t + f + 1e-12)) * 100
        self.epoch_id_total_val_g_correct = t
        self.val_g_accuracy = acc

    def evaluate_g_incorrectly(self, selection_threshold):
        prediction_result = self.val_out_put_class.argmax(dim=1)
        t = self.val_out_put_target.detach()
        selection_result = None
        if self.val_out_put_sel_proba is not None:
            condition = self.val_out_put_sel_proba < selection_threshold
            selection_result = torch.where(
                condition,
                torch.ones_like(self.val_out_put_sel_proba),
                torch.zeros_like(self.val_out_put_sel_proba),
            ).view(-1)
        h_rjc = torch.masked_select(prediction_result, selection_result.bool())
        t_rjc = torch.masked_select(self.val_out_put_target, selection_result.bool())
        t = float(torch.where(h_rjc == t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc != t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())

        acc = float(t / (t + f + 1e-12)) * 100
        self.val_g_incorrect_accuracy = acc

    def evaluate_coverage_stats(self, selection_threshold):
        prediction_result = self.val_out_put_class.argmax(dim=1)
        selection_result = None
        if self.val_out_put_sel_proba is not None:
            condition = self.val_out_put_sel_proba >= selection_threshold
            selection_result = torch.where(
                condition,
                torch.ones_like(self.val_out_put_sel_proba),
                torch.zeros_like(self.val_out_put_sel_proba),
            ).view(-1)
        condition_true = prediction_result == self.val_out_put_target
        condition_false = prediction_result != self.val_out_put_target
        condition_acc = selection_result == torch.ones_like(selection_result)
        condition_rjc = selection_result == torch.zeros_like(selection_result)
        ta = float(
            torch.where(
                condition_true & condition_acc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )
        tr = float(
            torch.where(
                condition_true & condition_rjc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )
        fa = float(
            torch.where(
                condition_false & condition_acc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )
        fr = float(
            torch.where(
                condition_false & condition_rjc,
                torch.ones_like(prediction_result),
                torch.zeros_like(prediction_result),
            ).sum()
        )

        rejection_rate = float((tr + fr) / (ta + tr + fa + fr + 1e-12))

        # rejection precision - not used in our code
        rejection_pre = float(tr / (tr + fr + 1e-12))

        self.n_rejected = tr + fr
        self.n_selected = len(self.val_loader.dataset) - (tr + fr)
        self.coverage = (1 - rejection_rate)

    def end_epoch(self, model, multi_label=False, track_explainer_loss=False, save_model_wrt_g_performance=False):
        """
        Records all the parameters at the end of each epoch.

        :return: none
        """
        self.epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        self.train_loss = self.epoch_train_loss / len(self.train_loader.dataset)
        self.val_loss = self.epoch_val_loss / len(self.val_loader.dataset)

        if not multi_label:
            # for multiclass classification
            self.train_accuracy = (self.epoch_id_total_train_correct / len(self.train_loader.dataset)) * 100
            self.val_accuracy = (self.epoch_id_total_val_correct / len(self.val_loader.dataset)) * 100
        else:
            # for multilabel classification
            self.train_accuracy = (self.epoch_id_total_train_correct / (
                    len(self.train_loader.dataset) * self.n_attributes)) * 100
            self.val_accuracy = (self.epoch_id_total_val_correct / (
                    len(self.val_loader.dataset) * self.n_attributes)) * 100

        self.tb.add_scalar("Epoch_stats_model/Train_correct", self.epoch_id_total_train_correct, self.epoch_id + 1)
        self.tb.add_scalar("Epoch_stats_model/Val_correct", self.epoch_id_total_val_correct, self.epoch_id + 1)
        self.tb.add_scalar("Epoch_stats_model/Train_accuracy", self.train_accuracy, self.epoch_id + 1)
        self.tb.add_scalar("Epoch_stats_model/Val_accuracy", self.val_accuracy, self.epoch_id + 1)

        self.tb.add_scalar("Epoch_Loss/Train_Loss", self.train_loss, self.epoch_id + 1)
        self.tb.add_scalar("Epoch_Loss/Val_Loss", self.val_loss, self.epoch_id + 1)

        # for logging in csv file
        self.arr_epoch_id.append(self.epoch_id + 1)
        self.arr_epoch_duration.append(self.epoch_duration)
        self.arr_total_train_loss.append(self.train_loss)
        self.arr_total_val_loss.append(self.val_loss)
        self.arr_train_acc.append(self.train_accuracy)
        self.arr_val_acc.append(self.val_accuracy)

        self.performance_dict["epoch_id"] = self.arr_epoch_id
        self.performance_dict["epoch_duration"] = self.arr_epoch_duration
        self.performance_dict["train_loss"] = self.arr_total_train_loss
        self.performance_dict["val_loss"] = self.arr_total_val_loss
        self.performance_dict["train_acc"] = self.arr_train_acc
        self.performance_dict["val_acc"] = self.arr_val_acc

        # individual losses for g's
        if track_explainer_loss:
            self.track_g_loss_stats()

        # for name, param in model.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_id)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_id)

        torch.save(
            model.state_dict(), os.path.join(self.checkpoint_path, f"model_seq_epoch_{self.epoch_id + 1}.pth.tar")
        )

        if save_model_wrt_g_performance:
            self.save_model_g(model)
        else:
            self.save_model(model)

        performance_df = pd.DataFrame(self.performance_dict)
        performance_df.to_csv(os.path.join(output_path, "train_val_stats") + ".csv")

    def track_g_loss_stats(self):
        self.train_emp_coverage = self.epoch_train_emp_coverage / len(self.train_loader.dataset)
        self.train_distillation_risk = self.epoch_train_distillation_risk / len(self.train_loader.dataset)
        self.train_CE_risk = self.epoch_train_CE_risk / len(self.train_loader.dataset)
        self.train_KD_risk = self.epoch_train_KD_risk / len(self.train_loader.dataset)
        self.train_entropy_risk = self.epoch_train_entropy_risk / len(self.train_loader.dataset)
        self.train_emp_risk = self.epoch_train_emp_risk / len(self.train_loader.dataset)
        self.train_cov_penalty = self.epoch_train_cov_penalty / len(self.train_loader.dataset)
        self.train_selective_loss = self.epoch_train_selective_loss / len(self.train_loader.dataset)
        self.train_aux_loss = self.epoch_train_aux_loss / len(self.train_loader.dataset)

        self.val_emp_coverage = self.epoch_val_emp_coverage / len(self.val_loader.dataset)
        self.val_distillation_risk = self.epoch_val_distillation_risk / len(self.val_loader.dataset)
        self.val_CE_risk = self.epoch_val_CE_risk / len(self.val_loader.dataset)
        self.val_KD_risk = self.epoch_val_KD_risk / len(self.val_loader.dataset)
        self.val_entropy_risk = self.epoch_val_entropy_risk / len(self.val_loader.dataset)
        self.val_emp_risk = self.epoch_val_emp_risk / len(self.val_loader.dataset)
        self.val_cov_penalty = self.epoch_val_cov_penalty / len(self.val_loader.dataset)
        self.val_selective_loss = self.epoch_val_selective_loss / len(self.val_loader.dataset)
        self.val_aux_loss = self.epoch_val_aux_loss / len(self.val_loader.dataset)

        self.tb.add_scalar("Loss_g_train/Empirical_Coverage ", self.train_emp_coverage, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_train/Distillation_Risk", self.train_distillation_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_train/CE_Risk", self.train_CE_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_train/KD_Risk (Distillation + CE)", self.train_KD_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_train/Entropy_Risk", self.train_entropy_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_train/Emp_Risk (KD + Entropy)", self.train_emp_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_train/Cov_Penalty", self.train_cov_penalty, self.epoch_id + 1)
        self.tb.add_scalar(
            "Loss_g_train/Selective_Loss (Emp + Cov)", self.train_selective_loss, self.epoch_id + 1
        )
        self.tb.add_scalar("Loss_g_train/Aux_Loss", self.train_aux_loss, self.epoch_id + 1)

        self.tb.add_scalar("Loss_g_val/Empirical_Coverage ", self.val_emp_coverage, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_val/Distillation_Risk", self.val_distillation_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_val/CE_Risk", self.val_CE_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_val/KD_Risk (Distillation + CE)", self.val_KD_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_val/Entropy_Risk", self.val_entropy_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_val/Emp_Risk (KD + Entropy)", self.val_emp_risk, self.epoch_id + 1)
        self.tb.add_scalar("Loss_g_val/Cov_Penalty", self.val_cov_penalty, self.epoch_id + 1)
        self.tb.add_scalar(
            "Loss_g_val/Selective_Loss (Emp + Cov)", self.val_selective_loss, self.epoch_id + 1
        )
        self.tb.add_scalar("Loss_g_val/Aux_Loss", self.val_aux_loss, self.epoch_id + 1)

        self.tb.add_scalar(
            "Epoch_stats_g/Accuracy_Correctly_Selected (pi >= 0.5)", self.val_g_accuracy, self.epoch_id + 1
        )
        self.tb.add_scalar(
            "Epoch_stats_g/Accuracy_Correctly_Rejected (pi < 0.5)", self.val_g_incorrect_accuracy, self.epoch_id + 1
        )

        self.tb.add_scalar("Pi_stats/N_Selected", self.n_selected, self.epoch_id + 1)
        self.tb.add_scalar("Pi_stats/N_Rejected", self.n_rejected, self.epoch_id + 1)
        self.tb.add_scalar("Pi_stats/coverage", self.coverage, self.epoch_id + 1)

        # for logging in csv file
        self.arr_train_emp_coverage.append(self.train_emp_coverage)
        self.arr_train_distillation_risk.append(self.train_distillation_risk)
        self.arr_train_CE_risk.append(self.train_CE_risk)
        self.arr_train_KD_risk.append(self.train_KD_risk)
        self.arr_train_entropy_risk.append(self.train_entropy_risk)
        self.arr_train_emp_risk.append(self.train_emp_risk)
        self.arr_train_cov_penalty.append(self.train_cov_penalty)
        self.arr_train_selective_loss.append(self.train_selective_loss)
        self.arr_train_aux_loss.append(self.train_aux_loss)

        self.arr_val_emp_coverage.append(self.val_emp_coverage)
        self.arr_val_distillation_risk.append(self.val_distillation_risk)
        self.arr_val_CE_risk.append(self.val_CE_risk)
        self.arr_val_KD_risk.append(self.val_KD_risk)
        self.arr_val_entropy_risk.append(self.val_entropy_risk)
        self.arr_val_emp_risk.append(self.val_emp_risk)
        self.arr_val_cov_penalty.append(self.val_cov_penalty)
        self.arr_val_selective_loss.append(self.val_selective_loss)
        self.arr_val_aux_loss.append(self.val_aux_loss)

        self.arr_n_selected.append(self.n_selected)
        self.arr_n_rejected.append(self.n_rejected)
        self.arr_coverage.append(self.coverage)
        self.arr_val_g_accuracy.append(self.val_g_accuracy)
        self.arr_val_g_incorrect_accuracy.append(self.val_g_incorrect_accuracy)

        self.performance_dict["train_emp_coverage"] = self.arr_train_emp_coverage
        self.performance_dict["train_distillation_risk"] = self.arr_train_distillation_risk
        self.performance_dict["train_CE_risk"] = self.arr_train_CE_risk
        self.performance_dict["train_KD_risk"] = self.arr_train_KD_risk
        self.performance_dict["train_entropy_risk"] = self.arr_train_entropy_risk
        self.performance_dict["train_emp_risk"] = self.arr_train_emp_risk
        self.performance_dict["train_cov_penalty"] = self.arr_train_cov_penalty
        self.performance_dict["train_selective_loss"] = self.arr_train_selective_loss
        self.performance_dict["train_aux_loss"] = self.arr_train_aux_loss

        self.performance_dict["val_emp_coverage"] = self.arr_val_emp_coverage
        self.performance_dict["val_distillation_risk"] = self.arr_val_distillation_risk
        self.performance_dict["val_CE_risk"] = self.arr_val_CE_risk
        self.performance_dict["val_KD_risk"] = self.arr_val_KD_risk
        self.performance_dict["val_entropy_risk"] = self.arr_val_entropy_risk
        self.performance_dict["val_emp_risk"] = self.arr_val_emp_risk
        self.performance_dict["val_cov_penalty"] = self.arr_val_cov_penalty
        self.performance_dict["val_selective_loss"] = self.arr_val_selective_loss
        self.performance_dict["val_aux_loss"] = self.arr_val_aux_loss

        self.performance_dict["n_selected"] = self.arr_n_selected
        self.performance_dict["n_rejected"] = self.arr_n_rejected
        self.performance_dict["coverage"] = self.arr_coverage
        self.performance_dict["val_g_accuracy (pi >= 0.5)"] = self.arr_val_g_accuracy
        self.performance_dict["g_incorrect_accuracy (pi < 0.5)"] = self.arr_val_g_incorrect_accuracy

    def save_model_g(self, model):
        if self.val_g_accuracy > self.best_val_g_accuracy:
            torch.save(
                model.state_dict(),
                os.path.join(self.checkpoint_path, f"model_g_best_model_epoch_{self.epoch_id + 1}.pth.tar")
            )
            print(f"\n Old best val accuracy of g : {self.best_val_g_accuracy} (%) || "
                  f"New best val accuracy: {self.val_g_accuracy} (%) , and new model saved..\n")

            self.best_epoch_id = self.epoch_id
            self.best_val_g_accuracy = self.val_g_accuracy

            self.arr_best_epoch_id.append(self.best_epoch_id)
            self.arr_best_val_g_acc.append(self.best_val_g_accuracy)

            self.performance_dict["best_epoch_id"] = self.arr_best_epoch_id
            self.performance_dict["best_val_g_acc"] = self.arr_best_val_g_acc

    def save_model(self, model):
        if self.val_accuracy > self.best_val_accuracy:
            torch.save(
                model.state_dict(),
                os.path.join(self.checkpoint_path, f"g_best_model_epoch_{self.epoch_id + 1}.pth.tar")
            )
            print(f"\n Old best val accuracy: {self.best_val_accuracy} (%) || "
                  f"New best val accuracy: {self.val_accuracy} (%) , and new model saved..\n")

            self.best_epoch_id = self.epoch_id
            self.best_val_accuracy = self.val_accuracy

            self.arr_best_epoch_id.append(self.best_epoch_id)
            self.arr_best_val_acc.append(self.best_val_accuracy)

            self.performance_dict["best_epoch_id"] = self.arr_best_epoch_id
            self.performance_dict["val_acc"] = self.arr_best_val_acc

    def track_train_loss(self, loss):
        """
        Calculates the loss at the each iteration of batch.

        :param loss:

        :return: calculated loss
        """
        self.epoch_train_loss += loss * self.train_loader.batch_size

    def track_train_losses_wrt_g(
            self, train_emp_coverage, train_distillation_risk, train_CE_risk,
            train_KD_risk, train_entropy_risk, train_emp_risk,
            train_cov_penalty, train_selective_loss, train_aux_loss
    ):
        self.epoch_train_emp_coverage += train_emp_coverage * self.train_loader.batch_size
        self.epoch_train_distillation_risk += train_distillation_risk * self.train_loader.batch_size
        self.epoch_train_CE_risk += train_CE_risk * self.train_loader.batch_size
        self.epoch_train_KD_risk += train_KD_risk * self.train_loader.batch_size
        self.epoch_train_entropy_risk += train_entropy_risk * self.train_loader.batch_size
        self.epoch_train_emp_risk += train_emp_risk * self.train_loader.batch_size
        self.epoch_train_cov_penalty += train_cov_penalty * self.train_loader.batch_size
        self.epoch_train_selective_loss += train_selective_loss * self.train_loader.batch_size
        self.epoch_train_aux_loss += train_aux_loss * self.train_loader.batch_size

    def track_total_train_correct_per_epoch(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.epoch_id_total_train_correct += get_correct(preds, labels, self.n_classes)

    def track_total_train_correct_multilabel_per_epoch(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.epoch_id_total_train_correct += get_correct_multi_label(preds, labels)

    def track_val_loss(self, loss):
        """
        Calculates the loss at the each iteration of batch.

        :param loss:

        :return: calculated loss
        """
        self.epoch_val_loss += loss * self.val_loader.batch_size

    def track_val_losses_wrt_g(
            self, val_emp_coverage, val_distillation_risk, val_CE_risk,
            val_KD_risk, val_entropy_risk, val_emp_risk,
            val_cov_penalty, val_selective_loss, val_aux_loss
    ):
        self.epoch_val_emp_coverage += val_emp_coverage * self.val_loader.batch_size
        self.epoch_val_distillation_risk += val_distillation_risk * self.val_loader.batch_size
        self.epoch_val_CE_risk += val_CE_risk * self.val_loader.batch_size
        self.epoch_val_KD_risk += val_KD_risk * self.val_loader.batch_size
        self.epoch_val_entropy_risk += val_entropy_risk * self.val_loader.batch_size
        self.epoch_val_emp_risk += val_emp_risk * self.val_loader.batch_size
        self.epoch_val_cov_penalty += val_cov_penalty * self.val_loader.batch_size
        self.epoch_val_selective_loss += val_selective_loss * self.val_loader.batch_size
        self.epoch_val_aux_loss += val_aux_loss * self.val_loader.batch_size

    def track_total_val_correct_per_epoch(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.epoch_id_total_val_correct += get_correct(preds, labels, self.n_classes)

    def track_val_outputs(self, out_select, out_class, val_y):
        self.val_out_put_sel_proba = torch.cat((self.val_out_put_sel_proba, out_select), dim=0)
        self.val_out_put_class = torch.cat((self.val_out_put_class, out_class), dim=0)
        self.val_out_put_target = torch.cat((self.val_out_put_target, val_y), dim=0)

    def track_total_val_correct_multilabel_per_epoch(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.epoch_id_total_val_correct += get_correct_multi_label(preds, labels)

    def get_final_val_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.val_loss

    def get_final_val_KD_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.val_KD_risk

    def get_final_val_entropy_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.val_entropy_risk

    def get_final_val_aux_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.val_aux_loss

    def get_final_train_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.train_loss

    def get_final_train_KD_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.train_KD_risk

    def get_final_train_entropy_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.train_entropy_risk

    def get_final_train_aux_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.train_aux_loss

    def get_final_best_val_accuracy(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.best_val_accuracy

    def get_final_val_accuracy(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.val_accuracy

    def get_final_G_val_accuracy(self):
        return self.val_g_accuracy

    def get_final_G_val_incorrect_accuracy(self):
        return self.val_g_incorrect_accuracy

    def get_final_best_G_val_accuracy(self):
        return self.best_val_g_accuracy

    def get_final_train_accuracy(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.train_accuracy

    def get_n_selected(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.n_selected

    def get_n_rejected(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.n_rejected

    def get_acc_selected(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.acc_selected

    def get_acc_rejected(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.acc_rejected

    def get_coverage(self):
        return self.coverage

    def get_epoch_duration(self):
        return self.epoch_duration

    def get_best_epoch_id(self):
        return self.best_epoch_id
