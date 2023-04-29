import logging
import os
import random
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset.utils_dataset as utils_dataset
import utils
from BB.models.BB_ResNet import ResNet
from BB.models.VIT import CONFIGS, VisionTransformer
from Explainer.profiler import fit_g_cub_profiler_cnn
from Logger.logger_cubs import Logger_CUBS
from dataset.dataset_awa2 import AnimalDataset, Awa2

logger = logging.getLogger(__name__)


def train(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch)
    tb_logs_path = os.path.join(args.logs, args.dataset, "BB", f"{root}_{args.arch}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    device = utils.get_device()
    print(f"Device: {device}")

    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    dataset = AnimalDataset(args)
    train_indices, val_indices = train_test_split(
        list(range(len(dataset.target_index))), test_size=0.2, stratify=dataset.target_index
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_ds_awa2 = Awa2(train_dataset, train_transform)
    val_ds_awa2 = Awa2(val_dataset, val_transform)

    train_loader = DataLoader(train_ds_awa2, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_awa2, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    net = get_model(args, device)

    criterion = utils.get_criterion(args.dataset)
    solver = utils.get_optim(
        args.dataset,
        net,
        params={
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay
        })
    print(solver)
    schedule = utils.get_scheduler(solver, args)

    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_CUBS(1, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader, len(args.labels))
    if args.arch == "ResNet50" or args.arch == "ResNet101":
        if args.profile == "n":
            fit(
                args,
                net,
                criterion,
                solver,
                schedule,
                train_loader,
                val_loader,
                run_manager,
                run_id,
                device
            )
        else:
            fit_g_cub_profiler_cnn(
                args,
                net,
                criterion,
                solver,
                schedule,
                train_loader,
                val_loader,
                run_id,
                device
            )


def test(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Testing the network...")
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch)
    device = utils.get_device()
    print(f"Device: {device}")
    os.makedirs(output_path, exist_ok=True)

    args.root_bb = root if args.arch == "ResNet101" or args.arch == "ResNet50" else f"lr_0.03_epochs_{args.epochs}"
    args.checkpoint_bb = args.checkpoint_file
    args.smoothing_value = 0
    net = utils.get_model_explainer(args, device)
    # model_chk_pt = torch.load(os.path.join(chk_pt_path, args.checkpoint_file))
    # net.load_state_dict(model_chk_pt)
    net.eval()
    # print(net)
    print(os.path.join(chk_pt_path, args.checkpoint_file))
    test_transform = utils.get_test_transforms(args.dataset, args.img_size, args.arch)
    if args.spurious_waterbird_landbird == "y":
        _, _, test_loader = utils_dataset.get_dataloader_spurious_waterbird_landbird(
            args
        )
    else:
        test_loader = utils_dataset.get_test_dataloader(
            args.data_root,
            args.json_root,
            args.dataset,
            args.bs,
            test_transform,
            attribute_file=args.attribute_file_name
        )
    start = time.time()
    sigma_test = None
    validate(args, test_loader, net, args.dataset, args.labels, output_path, device)
    done = time.time()
    elapsed = done - start
    print("Time to test for this iteration: " + str(elapsed) + " secs")
    print("###########################################################")


def save_activations(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Saving the activations")
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path_bb = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path_bb = os.path.join(args.output, args.dataset, "BB", root, args.arch)

    device = utils.get_device()
    print(f"Device: {device}")

    bb = utils.get_model(args.arch, args.dataset, args.pretrained, len(args.labels), args.layer).to(device)
    bb.load_state_dict(torch.load(os.path.join(chk_pt_path_bb, args.checkpoint_file)))
    bb.eval()

    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    start = time.time()
    train_loader, val_loader = utils_dataset.get_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        train_transform,
        val_transform,
        train_shuffle=False
    )

    test_transform = utils.get_test_transforms(args.dataset, args.img_size)
    test_loader = utils_dataset.get_test_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        test_transform
    )
    done = time.time()
    elapsed = done - start
    print("Time to load train-test-val datasets: " + str(elapsed) + " secs")

    start = time.time()
    store_feature_maps(
        train_loader,
        args.layer,
        device,
        bb,
        args.dataset,
        output_path_bb,
        file_name=f"train_features_{args.layer}"
    )
    done = time.time()
    elapsed = done - start
    print("Time to save train activations: " + str(elapsed) + " secs")

    start = time.time()
    store_feature_maps(
        val_loader,
        args.layer,
        device,
        bb,
        args.dataset,
        output_path_bb,
        file_name=f"val_features_{args.layer}"
    )
    done = time.time()
    elapsed = done - start
    print("Time to save val activations: " + str(elapsed) + " secs")

    start = time.time()
    store_feature_maps(
        test_loader,
        args.layer,
        device,
        bb,
        args.dataset,
        output_path_bb,
        file_name=f"test_features_{args.layer}"
    )
    done = time.time()
    elapsed = done - start
    print("Time to save test activations: " + str(elapsed) + " secs")


def store_feature_maps(dataloader, layer, device, bb, dataset_name, output_path_bb, file_name):
    attr_GT = torch.FloatTensor()
    activations = []

    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for batch_id, data_tuple in enumerate(dataloader):
                image, attribute = utils.get_image_attributes(data_tuple, dataset_name)
                image = image.to(device)
                _ = bb(image).cpu().detach()
                activations.append(bb.feature_store[layer].cpu().detach().numpy())
                t.set_postfix(batch_id='{0}'.format(batch_id))
                attr_GT = torch.cat((attr_GT, attribute), dim=0)
                t.update()

    print("Activations are generated..")
    activations = np.concatenate(activations, axis=0)
    print(activations.shape)
    attr_GT = attr_GT.cpu().numpy()
    print(attr_GT.shape)

    utils.save_features(output_path_bb, f"{file_name}.h5", layer, activations)
    np.save(os.path.join(output_path_bb, f"{file_name}_attr_GT.npy"), attr_GT)


def get_bb_logits(arch, net, data, sigma_test, attribute, scale, device):
    if arch == "ResNet101" or arch == "ResNet50":
        return net(data)
    elif arch == "ViT-B_16":
        return net(data)
    elif arch == "ViT-B_16_projected":
        concept = attribute[:, 108: 110]
        y_hat_bb = net(data, concept, scale, sigma_test, device)
        return y_hat_bb


def validate(args, test_loader, net, dataset, labels, output_path, device, sigma_test=None):
    out_prob_arr_bb = []
    tensor_images = torch.FloatTensor()
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, data_tuple in enumerate(test_loader):
                data, target, attribute = data_tuple
                data, target = data.to(device), target.to(torch.long).to(device)
                attribute = attribute.to(device, dtype=torch.float)
                scale = len(test_loader.dataset) / data.size(0)
                y_hat_bb = get_bb_logits(args.arch, net, data, sigma_test, attribute, scale, device)
                out_prob_arr_bb.append(y_hat_bb)

                tensor_images = torch.cat((tensor_images, data.cpu()), dim=0)
                out_put_predict_bb = torch.cat((out_put_predict_bb, y_hat_bb), dim=0)
                out_put_GT = torch.cat((out_put_GT, target), dim=0)

                t.set_postfix(iteration=f"{batch_idx}")
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_bb_np = out_put_predict_bb.cpu().numpy()
    y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1)
    acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
    cls_report = utils.cal_classification_report(out_put_GT_np, y_hat_bb, labels)

    print(f"Accuracy of the network: {acc_bb * 100} (%)")
    print(f"tensor_images size: {tensor_images.size()}")
    print(cls_report)
    np.save(os.path.join(output_path, f"out_put_GT_prune.npy"), out_put_GT_np)
    torch.save(out_put_predict_bb.cpu(), os.path.join(output_path, f"out_put_predict_logits_bb.pt"))
    torch.save(y_hat_bb, os.path.join(output_path, f"out_put_predict_bb.pt"))
    utils.save_tensor(
        path=os.path.join(output_path, f"test_image_tensor_original.pt"), tensor_to_save=tensor_images
    )
    print(os.path.join(output_path, f"out_put_predict_bb.pt"))
    # np.save(os.path.join(output_path, f"out_put_predict_bb_prune.npy"), y_hat_bb)
    utils.dump_in_pickle(output_path=output_path, file_name="classification_report.pkl", stats_to_dump=cls_report)


def fit(
        args,
        net,
        criterion,
        solver,
        schedule,
        train_loader,
        val_loader,
        run_manager,
        run_id,
        device
):
    run_manager.begin_run(run_id)
    for epoch in range(args.epochs):
        run_manager.begin_epoch()
        net.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                solver.zero_grad()
                images, _, _, labels = data_tuple
                images, labels = images.to(device), labels.to(torch.long).to(device)
                y_hat = net(images)
                train_loss = criterion(y_hat, labels)
                train_loss.backward()
                solver.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(y_hat, labels)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        net.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    images, _, _, labels = data_tuple
                    images, labels = images.to(device), labels.to(torch.long).to(device)
                    y_hat = net(images)
                    val_loss = criterion(y_hat, labels)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(y_hat, labels)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        if schedule is not None:
            schedule.step()
        run_manager.end_epoch(net)
        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)}")

    run_manager.end_run()


def get_model(args, device):
    if args.arch == "ResNet50" or args.arch == "ResNet101":
        return ResNet(
            dataset=args.dataset, pre_trained=args.pretrained, n_class=len(args.labels),
            model_choice=args.arch, layer="layer4"
        ).to(device)
    elif args.arch == "ViT-B_16":
        _config = CONFIGS[args.arch]
        _config.split = "non-overlap"
        _config.slide_step = 12
        _img_size = args.img_size
        _smoothing_value = 0.0
        _num_classes = len(args.labels)

        model = VisionTransformer(
            _config, _img_size, zero_head=True, num_classes=_num_classes, smoothing_value=_smoothing_value
        )

        pre_trained = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/pretrained_VIT/ViT-B_16.npz"
        checkpoint = np.load(pre_trained)
        model.load_from(checkpoint)
        return model.to(device)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_model(args, model, chk_pt_path, step_id):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(chk_pt_path, f"g_best_model_step_{step_id}.pth.tar")
    """
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    """
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", chk_pt_path)
