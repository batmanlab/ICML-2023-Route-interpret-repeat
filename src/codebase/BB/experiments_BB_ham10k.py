import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

import utils
from BB.models.BB_Inception_V3 import get_model
from dataset.dataset_ham10k import load_ham_data


def test(args):
    device = utils.get_device()
    output_path = os.path.join(args.output, args.dataset, "BB", args.arch)
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

    model, model_bottom, model_top = get_model(args.bb_dir, args.model_name)
    train_loader, val_loader, idx_to_class = load_ham_data(args, transform, args.class_to_idx)
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
    cls_report = utils.cal_classification_report(out_put_GT_np, y_hat_bb, args.labels)
    print(f"Accuracy of the network: {acc_bb * 100} (%)")
    print(f"Val AUROC of the network: {val_auroc} (0-1)")
    print(cls_report)

    np.save(os.path.join(output_path, f"out_put_GT_prune.npy"), out_put_GT_np)
    torch.save(out_put_predict_bb.cpu(), os.path.join(output_path, f"out_put_predict_logits_bb.pt"))
    torch.save(y_hat_bb, os.path.join(output_path, f"out_put_predict_bb.pt"))
    print(os.path.join(output_path, f"out_put_predict_bb.pt"))
    # np.save(os.path.join(output_path, f"out_put_predict_bb_prune.npy"), y_hat_bb)
    utils.dump_in_pickle(output_path=output_path, file_name="classification_report.pkl", stats_to_dump=cls_report)

# def load_isic(args, preprocess, n_train=2000, n_val=400):
#     df = pd.read_csv(os.path.join(ISIC_FOLDER, 'train.csv'))
#     df['path'] = df['image_name'].map(lambda name: os.path.join(ISIC_FOLDER, "train", name + '.jpg'))
#     df['y'] = df['target']
#
#     files = os.listdir(os.path.join(ISIC_FOLDER, "train"))
#     files = [os.path.join(ISIC_FOLDER, "train", f) for f in files]
#     df = df[df.path.isin(files)]
#
#     df_pos = df[df.y == 1]
#     df_neg = df[df.y == 0]
#
#     _, df_val_pos = train_test_split(df_pos, test_size=0.20, random_state=args.seed)
#     _, df_val_neg = train_test_split(df_neg, test_size=0.20, random_state=args.seed)
#     df_train_pos = df_pos[~df_pos.path.isin(df_val_pos.path)]
#     df_train_neg = df_neg[~df_neg.path.isin(df_val_neg.path)]
#
#     df_train_pos = df_train_pos.sample(n_train // 5, random_state=args.seed)
#     df_train_neg = df_train_neg.sample(4 * n_train // 5, random_state=args.seed)
#
#     df_val_pos = df_val_pos.sample(n_val // 5, random_state=args.seed)
#     df_val_neg = df_val_neg.sample(4 * n_val // 5, random_state=args.seed)
#
#     df_train = pd.concat([df_train_pos, df_train_neg])
#     df_val = pd.concat([df_val_pos, df_val_neg])
#
#     trainset = DermDataset(df_train, preprocess)
#     valset = DermDataset(df_val, preprocess)
#
#     print(f"Train, Val: {df_train.shape}, {df_val.shape}")
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
#                                                shuffle=True, num_workers=args.num_workers)
#
#     val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
#                                              shuffle=False, num_workers=args.num_workers)
#
#     class_to_idx = {"benign": 0, "malignant": 1}
#     idx_to_class = {v: k for k, v in class_to_idx.items()}
#     return train_loader, val_loader, idx_to_class
