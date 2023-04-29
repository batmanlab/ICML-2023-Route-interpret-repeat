import torch
import torch.nn.functional as F

from Explainer.models.entropy_layer import EntropyLinear


def entropy_loss(model: torch.nn.Module):
    """
    Entropy loss function to get simple logic explanations.

    :param model: pytorch model.
    :return: entropy loss.
    """
    loss = 0
    for module in model.model.children():
        if isinstance(module, EntropyLinear):
            loss -= torch.sum(module.alpha * torch.log(module.alpha))
            break
    return loss


def loss_fn_kd(explainer_logits, bb_logits, labels, params, weights):
    """
    reduction="none" as to compute the KD loss for each sample
    """
    alpha = params["alpha"]
    T = params["temperature"]
    distillation_loss = torch.nn.KLDivLoss(reduction="none")(
        F.log_softmax(explainer_logits / T, dim=1), F.softmax(bb_logits / T, dim=1))
    weighted_distillation_loss = weights * torch.sum(distillation_loss, dim=1)
    weighted_prediction_loss = weights * F.cross_entropy(explainer_logits, labels, reduction="none")
    mean_distillation_loss = torch.mean(weighted_distillation_loss)
    mean_prediction_loss = torch.mean(weighted_prediction_loss)
    KD_loss = (alpha * T * T) * mean_distillation_loss + \
              (1. - alpha) * mean_prediction_loss
    return KD_loss


class KD_Residual_Loss(torch.nn.Module):
    def __init__(self, iteration, CE, KLDiv, T_KD, alpha_KD):
        super(KD_Residual_Loss, self).__init__()
        self.CE = CE
        self.KLDiv = KLDiv
        self.T_KD = T_KD
        self.alpha_KD = alpha_KD
        self.iteration = iteration

    def forward(self, student_preds, teacher_preds, target, selection_weights, prev_selection_outs=None):
        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = 1 - selection_weights

        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * (1 - selection_weights)

        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(student_preds / self.T_KD, dim=1), F.softmax(teacher_preds / self.T_KD, dim=1)),
            dim=1
        )
        distillation_risk = torch.mean(distillation_loss * weights.view(-1))
        CE_risk = torch.mean(self.CE(student_preds, target) * weights.view(-1))
        KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk

        return {
            "distillation_risk": distillation_risk,
            "CE_risk": CE_risk,
            "KD_risk": KD_risk
        }


class Distillation_Loss(torch.nn.Module):
    def __init__(self, iteration, CE, KLDiv, T_KD, alpha_KD, selection_threshold, coverage: float, lm: float = 32.0):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(Distillation_Loss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.CE = CE
        self.KLDiv = KLDiv
        self.coverage = coverage
        self.lm = lm
        self.T_KD = T_KD
        self.alpha_KD = alpha_KD
        self.iteration = iteration
        self.selection_threshold = selection_threshold

    def forward(
            self, prediction_out, target, bb_out, elens_loss, lambda_lens
    ):
        # compute emp risk (=r^)
        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(prediction_out / self.T_KD, dim=1), F.softmax(bb_out / self.T_KD, dim=1)),
            dim=1
        )
        distillation_risk = torch.mean(distillation_loss)
        CE_risk = torch.mean(self.CE(prediction_out, target))
        KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk
        entropy_risk = torch.mean(lambda_lens * elens_loss)
        emp_risk = (KD_risk + entropy_risk)

        return {
            "selective_loss": emp_risk,
            "emp_coverage": torch.tensor(1),
            "distillation_risk": distillation_risk,
            "CE_risk": CE_risk,
            "KD_risk": KD_risk,
            "entropy_risk": entropy_risk,
            "emp_risk": emp_risk,
            "cov_penalty": torch.tensor(0)
        }


class Selective_Distillation_Loss(torch.nn.Module):
    def __init__(
            self, iteration, CE, KLDiv, T_KD, alpha_KD, selection_threshold, coverage: float, dataset="cub",
            lm: float = 32.0, arch=None
    ):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(Selective_Distillation_Loss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.CE = CE
        self.KLDiv = KLDiv
        self.coverage = coverage
        self.lm = lm
        self.T_KD = T_KD
        self.alpha_KD = alpha_KD
        self.iteration = iteration
        self.selection_threshold = selection_threshold
        self.dataset = dataset
        self.arch = arch

    def forward(
            self, prediction_out, selection_out, target, bb_out, elens_loss, lambda_lens, epoch, device,
            prev_selection_outs=None
    ):
        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = selection_out
        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * selection_out

        if self.dataset == "cub" or self.dataset == "CIFAR10":
            if self.iteration > 1 and epoch >= 85:
                condition = torch.full(prev_selection_outs[0].size(), True).to(device)
                for proba in prev_selection_outs:
                    condition = condition & (proba < self.selection_threshold)
                emp_coverage = torch.sum(weights) / (torch.sum(condition) + 1e-12)
            else:
                emp_coverage = torch.mean(weights)
        elif self.dataset == "mimic_cxr":
            emp_coverage = torch.mean(weights)
        elif self.dataset == "HAM10k" or self.dataset == "SIIM-ISIC":
            if self.iteration > 1:
                condition = torch.full(prev_selection_outs[0].size(), True).to(device)
                for proba in prev_selection_outs:
                    condition = condition & (proba < self.selection_threshold)
                emp_coverage = torch.sum(weights) / (torch.sum(condition) + 1e-12)
            else:
                emp_coverage = torch.mean(weights)

        # compute emp risk (=r^)
        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(prediction_out / self.T_KD, dim=1), F.softmax(bb_out / self.T_KD, dim=1)),
            dim=1
        )

        distillation_risk = torch.mean(distillation_loss * weights.view(-1))
        CE_risk = torch.mean(self.CE(prediction_out, target) * weights.view(-1))
        KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk
        entropy_risk = torch.mean(lambda_lens * elens_loss * weights.view(-1))
        emp_risk = (KD_risk + entropy_risk) / (emp_coverage + 1e-12)

        # compute penalty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device="cuda")
        penalty = (torch.max(
            coverage - emp_coverage,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda"),
        ) ** 2)
        penalty *= self.lm

        selective_loss = emp_risk + penalty
        return {
            "selective_loss": selective_loss,
            "emp_coverage": emp_coverage,
            "distillation_risk": distillation_risk,
            "CE_risk": CE_risk,
            "KD_risk": KD_risk,
            "entropy_risk": entropy_risk,
            "emp_risk": emp_risk,
            "cov_penalty": penalty
        }
