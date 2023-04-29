import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

import utils
from BB.models.BB_Inception_V3 import get_model_isic
from Logger.logger_cubs import Logger_CUBS
from dataset.dataset_ham10k import load_ham_data


def train(args):
    device = utils.get_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}_optim_{args.optim}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch)
    tb_logs_path = os.path.join(args.logs, args.dataset, "BB", f"{root}_{args.arch}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    print(f"output_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ]
    )

    model, model_bottom, model_top = get_model_isic(args.bb_dir, args.model_name)
    train_loader, val_loader, idx_to_class = load_ham_data(args, transform, args.class_to_idx)
    criterion = torch.nn.CrossEntropyLoss()
    solver = None
    if args.optim == "SGD":
        solver = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optim == "Adam":
        solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )
    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_CUBS(
        1, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader, len(args.labels)
    )

    model = fit(
        args,
        model,
        criterion,
        solver,
        train_loader,
        val_loader,
        run_manager,
        run_id,
        device
    )

    test_and_save_model(
        model, train_loader, val_loader, idx_to_class, output_path, args.labels, device
    )


def fit(
        args,
        net,
        criterion,
        solver,
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
            for batch_id, (images, target) in enumerate(train_loader):
                solver.zero_grad()
                images = images.to(device)
                target = target.to(device)
                y_hat = net(images)
                y_hat = y_hat.logits
                train_loss = criterion(y_hat, target)
                train_loss.backward()
                solver.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(y_hat, target)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        net.eval()
        out_put_GT = torch.FloatTensor().cuda()
        out_put_predict_bb = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (images, target) in enumerate(val_loader):
                    images = images.to(device)
                    target = target.to(device)
                    y_hat = net(images)
                    val_loss = criterion(y_hat, target)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(y_hat, target)

                    out_put_predict_bb = torch.cat((out_put_predict_bb, y_hat), dim=0)
                    out_put_GT = torch.cat((out_put_GT, target), dim=0)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(net)
        proba = torch.nn.Softmax(dim=1)(out_put_predict_bb)[:, 1]
        val_auroc, val_aurpc = utils.compute_AUC(out_put_GT, pred=proba)
        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Val_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} (%) "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} (%)  "
              f"Val_AUROC: {round(val_auroc, 4)} (%)  "
              f"Val_AURPC: {round(val_aurpc, 4)} (%)  "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)}")

    run_manager.end_run()

    return net


def test_and_save_model(
        model, train_loader, val_loader, idx_to_class, output_path, labels, device
):
    model = model.eval()
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(idx_to_class)

    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    with tqdm(total=len(val_loader)) as t:
        for batch in enumerate(val_loader):
            batch_idx, (images, target) = batch
            images = images.to(device)
            target = target.to(device)
            with torch.no_grad():
                y_hat_bb = model(images)

            out_put_predict_bb = torch.cat((out_put_predict_bb, y_hat_bb), dim=0)
            out_put_GT = torch.cat((out_put_GT, target), dim=0)

            t.set_postfix(iteration=f"{batch_idx}")
            t.update()

    proba = torch.nn.Softmax(dim=1)(out_put_predict_bb)[:, 1]
    val_auroc, val_aurpc = utils.compute_AUC(out_put_GT, pred=proba)

    out_put_GT_np = out_put_GT.cpu().numpy()
    y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1)
    acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
    cls_report = utils.cal_classification_report(out_put_GT_np, y_hat_bb, labels)
    print(f"Accuracy of the network: {acc_bb * 100} (%)")
    print(f"Val AUROC of the network: {val_auroc} (0-1)")
    print(cls_report)

    np.save(os.path.join(output_path, f"out_put_GT_prune.npy"), out_put_GT_np)
    torch.save(out_put_predict_bb.cpu(), os.path.join(output_path, f"out_put_predict_logits_bb.pt"))
    torch.save(y_hat_bb, os.path.join(output_path, f"out_put_predict_bb.pt"))
    print(os.path.join(output_path, f"out_put_predict_bb.pt"))
    # np.save(os.path.join(output_path, f"out_put_predict_bb_prune.npy"), y_hat_bb)
    utils.dump_in_pickle(
        output_path=output_path, file_name="classification_report.pkl", stats_to_dump=cls_report
    )
