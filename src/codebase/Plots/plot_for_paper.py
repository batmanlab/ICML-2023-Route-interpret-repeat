import numpy as np


def create_data_for_expert_label_plot(GT_tensor, pred_tensor, n_classes):
    gt_np = GT_tensor.numpy()
    pred_np = pred_tensor.argmax(dim=1).numpy()
    correctly_predicted = pred_np[pred_np == gt_np]
    correctly_predicted_per_class = []
    gt_per_class = []
    for i in range(n_classes):
        gt_per_class.append(np.count_nonzero(gt_np == i))
        correctly_predicted_per_class.append(np.count_nonzero(correctly_predicted == i))

    return gt_per_class, correctly_predicted_per_class
