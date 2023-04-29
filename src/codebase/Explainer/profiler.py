import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from Explainer.loss_F import entropy_loss
from Explainer.utils_explainer import get_previous_pi_vals

warnings.filterwarnings("ignore")


def fit_BB_domain_transfer_cxr_profiler(args, model, optimizer, train_loader, val_loader):
    model.cuda()
    print("******************** Profiler ********************")
    train_flops = []
    val_flops = []
    for epoch in range(1):
        # switch to train mode
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for i, data in enumerate(train_loader):
                (images, labels, image_name) = data
                disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                if args.domain_transfer == "n":
                    labels = labels[:, disease_idx]
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    labels = labels.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    features, pooled_features, logits = model(images)
                    train_loss = F.cross_entropy(logits, labels, reduction='mean')

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)

                t.set_postfix(epoch='{0}'.format(epoch))
                t.update()
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for i, data in enumerate(val_loader):
                    (images, labels, image_name) = data
                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    labels = labels[:, disease_idx]
                    if args.gpu is not None:
                        images = images.cuda(args.gpu, non_blocking=True)

                    if torch.cuda.is_available():
                        labels = labels.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        features, pooled_features, logits = model(images)
                        val_loss = F.cross_entropy(logits, labels, reduction='mean')

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    val_flops.append(fwbw_flops1)

                    t.set_postfix(epoch='{0}'.format(epoch))
                    t.update()

    total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
    print("#####" * 30)
    print(
        f"Disease: {args.disease}, Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))},  "
        f"total: {total_flops}"
    )
    print("#####" * 30)
    print()
    return (total_flops * args.epochs) / (10 ** 12)


def fit_residual_domain_transfer_cxr(
        args,
        bb_mimic,
        iteration,
        moie_target,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        kd_Loss,
        run_id,
        selection_threshold,
        device
):
    print("******************** Profiler ********************")
    train_flops = []
    val_flops = []

    for epoch in range(1):
        residual.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data in enumerate(train_loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    image = image.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                features, _, _ = bb_mimic(image)
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    out_class, out_select, out_aux = moie_target(proba_concept_x)

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                    residual_student_logits = residual(features)
                    total_train_loss = kd_Loss(
                        student_preds=residual_student_logits,
                        target=disease_label,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)

                t.set_postfix(epoch='{0}'.format(epoch + 1))
                t.update()

        residual.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    image, disease_label, proba_concept_x, attributes_gt, image_names = data
                    if torch.cuda.is_available():
                        image = image.cuda(args.gpu, non_blocking=True)
                        proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                        disease_label = disease_label.cuda(args.gpu, non_blocking=True).to(torch.long)

                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    disease_label = disease_label[:, disease_idx]
                    features, _, _ = bb_mimic(image)

                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        out_class, out_select, out_aux = moie_target(proba_concept_x)

                        pi_list = None
                        if iteration > 1:
                            pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                        residual_student_logits = residual(features)

                        total_val_loss = kd_Loss(
                            student_preds=residual_student_logits,
                            target=disease_label,
                            selection_weights=out_select,
                            prev_selection_outs=pi_list
                        )

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    val_flops.append(fwbw_flops1)
                    t.set_postfix(epoch='{0}'.format(epoch + 1))
                    t.update()

    total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
    print("#####" * 30)
    print(
        f"Disease: {args.disease}, Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))},  "
        f"total: {total_flops}"
    )
    print("#####" * 30)
    print()
    return (total_flops * args.epochs_residual) / (10 ** 12)


def fit_explainer_domain_transfer_cxr(
        args, train_loader, val_loader, moie_source, moie_target, glt_list, selective_CE_loss,
        optimizer, moie_configs, coverage, pos_cov_weight, device
):
    run_id = "MoIE_iters"
    print("******************** Profiler ********************")
    train_flops = []
    val_flops = []
    for epoch in range(1):
        moie_target.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data in enumerate(train_loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)
                out_class, _, out_aux = moie_source(proba_concept_x)
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    if args.initialize_w_mimic == "n":
                        _, out_select, _ = moie_target(proba_concept_x)
                        aux_entropy_loss_elens = entropy_loss(moie_source.aux_explainer)
                        entropy_loss_elens = entropy_loss(moie_source.explainer)

                    else:
                        out_class, out_select, out_aux = moie_target(proba_concept_x)
                        aux_entropy_loss_elens = entropy_loss(moie_target.aux_explainer)
                        entropy_loss_elens = entropy_loss(moie_target.explainer)

                    pi_list = None
                    if args.iter > 1:
                        pi_list = get_previous_pi_vals(args.iter, glt_list, proba_concept_x)

                    loss_dict = selective_CE_loss(
                        out_class, out_select, disease_label, entropy_loss_elens, moie_configs.lambda_lens,
                        prev_selection_outs=pi_list
                    )
                    train_selective_loss = loss_dict["selective_loss"]
                    emp_coverage_positive = loss_dict["emp_coverage_positive"].item()
                    emp_coverage_negative = loss_dict["emp_coverage_negative"].item()
                    cov_penalty_positive = loss_dict["cov_penalty_positive"].item()
                    cov_penalty_negative = loss_dict["cov_penalty_negative"].item()
                    train_selective_loss *= moie_configs.alpha
                    aux_KD_loss = torch.nn.CrossEntropyLoss()(out_aux, disease_label)

                    train_aux_loss = aux_KD_loss + moie_configs.lambda_lens * aux_entropy_loss_elens
                    train_aux_loss *= (1.0 - moie_configs.alpha)

                    total_train_loss = train_selective_loss + train_aux_loss
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                t.set_postfix(epoch='{0}'.format(epoch + 1))
                t.update()

        moie_target.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    image, disease_label, proba_concept_x, attributes_gt, image_names = data
                    if torch.cuda.is_available():
                        proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                        disease_label = disease_label.cuda(args.gpu, non_blocking=True).to(torch.long)

                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    disease_label = disease_label[:, disease_idx]
                    out_class, _, out_aux = moie_source(proba_concept_x)
                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        if args.initialize_w_mimic == "n":
                            _, out_select, _ = moie_target(proba_concept_x)
                            aux_entropy_loss_elens = entropy_loss(moie_source.aux_explainer)
                            entropy_loss_elens = entropy_loss(moie_source.explainer)

                        else:
                            out_class, out_select, out_aux = moie_target(proba_concept_x)
                            aux_entropy_loss_elens = entropy_loss(moie_target.aux_explainer)
                            entropy_loss_elens = entropy_loss(moie_target.explainer)

                        pi_list = None
                        if args.iter > 1:
                            pi_list = get_previous_pi_vals(args.iter, glt_list, proba_concept_x)

                        loss_dict = selective_CE_loss(
                            out_class, out_select, disease_label, entropy_loss_elens, moie_configs.lambda_lens,
                            prev_selection_outs=pi_list
                        )

                        val_selective_loss = loss_dict["selective_loss"]
                        emp_coverage_positive = loss_dict["emp_coverage_positive"].item()
                        emp_coverage_negative = loss_dict["emp_coverage_negative"].item()
                        cov_penalty_positive = loss_dict["cov_penalty_positive"].item()
                        cov_penalty_negative = loss_dict["cov_penalty_negative"].item()

                        val_selective_loss *= moie_configs.alpha
                        aux_KD_loss = torch.nn.CrossEntropyLoss()(out_aux, disease_label)

                        val_aux_loss = aux_KD_loss + moie_configs.lambda_lens * aux_entropy_loss_elens
                        val_aux_loss *= (1.0 - moie_configs.alpha)

                        if args.iter == 1:
                            idx_selected = (out_select >= 0.5).nonzero(as_tuple=True)[0]
                        else:
                            condition = torch.full(pi_list[0].size(), True).to(device)
                            for proba in pi_list:
                                condition = condition & (proba < moie_configs.selection_threshold)
                            idx_selected = (
                                    condition & (out_select >= moie_configs.selection_threshold)
                            ).nonzero(as_tuple=True)[0]

                        total_val_loss = val_selective_loss + val_aux_loss
                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    val_flops.append(fwbw_flops1)
                    t.set_postfix(epoch='{0}'.format(epoch + 1))
                    t.update()

    total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
    print("#####" * 30)
    print(
        f"Disease: {args.disease}, Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))},  "
        f"total: {total_flops}"
    )
    print("#####" * 30)
    print()
    return (total_flops * args.epochs) / (10 ** 12)


def fit_residual_ham_profiler(
        iteration,
        epochs,
        concept_bank,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        glt,
        glt_list,
        prev_residual,
        residual,
        optimizer,
        schedule,
        train_loader,
        val_loader,
        kd_Loss,
        logger,
        run_id,
        selection_threshold,
        device
):
    print("Profiler")
    train_flops = []
    val_flops = []
    for epoch in range(1):
        logger.begin_epoch()
        residual.train()

        with tqdm(total=len(train_loader)) as t:
            for batch in enumerate(train_loader):
                batch_idx, (images, target) = batch
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    train_bb_logits = None
                    if iteration == 1:
                        train_bb_logits = bb_logits
                    elif iteration > 1:
                        train_bb_logits, _ = prev_residual(phi)
                    train_concepts = compute_dist(concept_bank, phi)
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:

                    out_class, out_select, out_aux = glt(train_concepts)

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                    residual_student_logits, _ = residual(phi)
                    residual_teacher_logits = train_bb_logits - (out_select * out_class)

                    loss_dict = kd_Loss(
                        student_preds=residual_student_logits,
                        teacher_preds=residual_teacher_logits,
                        target=target,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    train_distillation_risk = loss_dict["distillation_risk"]
                    train_CE_risk = loss_dict["CE_risk"]
                    train_KD_risk = loss_dict["KD_risk"]

                    total_train_loss = train_KD_risk
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)

                t.set_postfix(epoch='{0}'.format(epoch + 1), )
                t.update()

        residual.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch in enumerate(val_loader):
                    batch_idx, (images, target) = batch
                    images = images.to(device)
                    target = target.to(device)
                    with torch.no_grad():
                        bb_logits = bb_model(images)
                        phi = bb_model_bottom(images)
                        val_bb_logits = None
                        if iteration == 1:
                            val_bb_logits = bb_logits
                        elif iteration > 1:
                            val_bb_logits, _ = prev_residual(phi)
                        val_concepts = compute_dist(concept_bank, phi)
                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        out_class, out_select, out_aux = glt(val_concepts)

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    print("Validation forward + backward flops: ", fwbw_flops1)
                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

    total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
    print("#####" * 30)
    print(
        f"Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))},  "
        f"total: {total_flops}"
    )
    print("#####" * 30)
    print()
    return (total_flops * epochs) / (10 ** 9)


def fit_g_ham_profile(
        iteration,
        concept_bank,
        epochs,
        alpha,
        temperature_KD,
        alpha_KD,
        bb_model,
        bb_model_bottom,
        bb_model_top,
        model,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        selective_KD_loss,
        logger,
        lambda_lens,
        run_id,
        selection_threshold,
        device
):
    print("Profiler")
    train_flops = []
    val_flops = []

    for epoch in range(1):
        logger.begin_epoch()
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for batch in enumerate(train_loader):
                batch_idx, (images, target) = batch
                images = images.to(device)
                target = target.to(device)
                with torch.no_grad():
                    bb_logits = bb_model(images)
                    phi = bb_model_bottom(images)
                    train_bb_logits = None
                    if iteration == 1:
                        train_bb_logits = bb_logits
                    elif iteration > 1:
                        train_bb_logits, _ = residual(phi)
                    train_concepts = compute_dist(concept_bank, phi)

                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    out_class, out_select, out_aux = model(train_concepts)

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                    entropy_loss_elens = entropy_loss(model.explainer)
                    loss_dict = selective_KD_loss(
                        out_class,
                        out_select,
                        target,
                        train_bb_logits,
                        entropy_loss_elens,
                        lambda_lens,
                        epoch,
                        device,
                        pi_list
                    )
                    train_selective_loss = loss_dict["selective_loss"]
                    train_emp_coverage = loss_dict["emp_coverage"]
                    train_distillation_risk = loss_dict["distillation_risk"]
                    train_CE_risk = loss_dict["CE_risk"]
                    train_KD_risk = loss_dict["KD_risk"]
                    train_entropy_risk = loss_dict["entropy_risk"]
                    train_emp_risk = loss_dict["emp_risk"]
                    train_cov_penalty = loss_dict["cov_penalty"]

                    train_selective_loss *= alpha
                    aux_distillation_loss = torch.nn.KLDivLoss()(
                        F.log_softmax(out_aux / temperature_KD, dim=1),
                        F.softmax(train_bb_logits / temperature_KD, dim=1)
                    )
                    aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, target)
                    aux_KD_loss = (alpha_KD * temperature_KD * temperature_KD) * aux_distillation_loss + \
                                  (1. - alpha_KD) * aux_ce_loss

                    aux_entropy_loss_elens = entropy_loss(model.aux_explainer)
                    train_aux_loss = aux_KD_loss + lambda_lens * aux_entropy_loss_elens
                    train_aux_loss *= (1.0 - alpha)

                    total_train_loss = train_selective_loss + train_aux_loss
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)

                t.set_postfix(epoch='{0}'.format(epoch + 1), )
                t.update()

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch in enumerate(val_loader):
                    batch_idx, (images, target) = batch
                    images = images.to(device)
                    target = target.to(device)
                    with torch.no_grad():
                        bb_logits = bb_model(images)
                        phi = bb_model_bottom(images)
                        val_bb_logits = None
                        if iteration == 1:
                            val_bb_logits = bb_logits
                        elif iteration > 1:
                            val_bb_logits, _ = residual(phi)
                        val_concepts = compute_dist(concept_bank, phi)

                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        out_class, out_select, out_aux = model(val_concepts)

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    print("Validation forward + backward flops: ", fwbw_flops1)
                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

    total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
    print("#####" * 30)
    print(
        f"Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))},  "
        f"total: {total_flops}"
    )
    print("#####" * 30)
    print()
    return (total_flops * epochs) / (10 ** 9)


def fit_residual_cub_profiler(
        iteration,
        epochs,
        bb,
        glt,
        glt_list,
        prev_residual,
        residual,
        optimizer,
        schedule,
        train_loader,
        val_loader,
        kd_Loss,
        layer,
        arch,
        run_id,
        selection_threshold,
        use_concepts_as_pi_input,
        device
):
    print("Running profiler")
    train_flops = []
    val_flops = []
    for epoch in range(1):
        residual.train()

        with tqdm(total=len(train_loader)) as t:
            for batch_id, (train_images, train_concepts, _, train_y, train_y_one_hot) in enumerate(train_loader):
                train_images, train_concepts, train_y, train_y_one_hot = train_images.to(device), \
                                                                         train_concepts.to(device), \
                                                                         train_y.to(torch.long).to(device), \
                                                                         train_y_one_hot.to(device)

                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)
                    train_bb_logits = bb_logits if iteration == 1 else prev_residual(feature_x)

                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    if use_concepts_as_pi_input:
                        out_class, out_select, out_aux = glt(train_concepts)
                    else:
                        out_class, out_select, out_aux = glt(train_concepts, feature_x.to(device))

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                    residual_student_logits = residual(feature_x)
                    residual_teacher_logits = train_bb_logits - out_class

                    loss_dict = kd_Loss(
                        student_preds=residual_student_logits,
                        teacher_preds=residual_teacher_logits,
                        target=train_y,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    train_distillation_risk = loss_dict["distillation_risk"]
                    train_CE_risk = loss_dict["CE_risk"]
                    train_KD_risk = loss_dict["KD_risk"]

                    total_train_loss = train_KD_risk
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)
                t.set_postfix(epoch='{0}'.format(epoch + 1), )
                t.update()

        residual.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (val_images, val_concepts, _, val_y, val_y_one_hot) in enumerate(val_loader):
                    val_images, val_concepts, val_y, val_y_one_hot = val_images.to(device), \
                                                                     val_concepts.to(device), \
                                                                     val_y.to(torch.long).to(device), \
                                                                     val_y_one_hot.to(device)
                    with torch.no_grad():
                        bb_logits, val_feature_x = get_phi_x(val_images, bb, arch, layer)
                        val_bb_logits = bb_logits if iteration == 1 else prev_residual(val_feature_x)

                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        if use_concepts_as_pi_input:
                            out_class, out_select, out_aux = glt(val_concepts)
                        else:
                            out_class, out_select, out_aux = glt(val_concepts, val_feature_x.to(device))

                        residual_student_logits = residual(val_feature_x)
                        residual_teacher_logits = val_bb_logits - out_class

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    print("Validation forward + backward flops: ", fwbw_flops1)

                    t.set_postfix(epoch='{0}'.format(epoch + 1))
                    t.update()
                    val_flops.append(fwbw_flops1)

        total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
        print("#####" * 30)
        print(
            f"Train_flops: {np.sum(np.array(train_flops))}, "
            f"Val_flops: {np.sum(np.array(val_flops))},  "
            f"total: {total_flops}"
        )
        print("#####" * 30)
        print()
        return (total_flops * epochs) / (10 ** 12)


def fit_g_cub_profiler_cnn(
        args,
        net,
        criterion,
        solver,
        schedule,
        train_loader,
        val_loader,
        run_id,
        device
):
    train_flops = []
    val_flops = []
    for epoch in range(1):
        net.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                solver.zero_grad()
                if args.dataset == "cub":
                    images, labels = utils.get_image_label(args, data_tuple, device)
                elif args.dataset == "awa2":
                    images, _, _, labels = data_tuple
                    images, labels = images.to(device), labels.to(torch.long).to(device)
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    y_hat = net(images)
                    train_loss = criterion(y_hat, labels)
                    train_loss.backward()
                    solver.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)
                t.set_postfix(epoch='{0}'.format(epoch))
                t.update()

        net.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if args.dataset == "cub":
                        images, labels = utils.get_image_label(args, data_tuple, device)
                    elif args.dataset == "awa2":
                        images, _, _, labels = data_tuple
                        images, labels = images.to(device), labels.to(torch.long).to(device)
                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        y_hat = net(images)
                        val_loss = criterion(y_hat, labels)

                    events = prof1.events()
                    fwbw_flops = sum([int(evt.flops) for evt in events])
                    val_flops.append(fwbw_flops)
                    print("val forward + backward flops: ", fwbw_flops)
                    t.set_postfix(epoch='{0}'.format(epoch))
                    t.update()

        if schedule is not None:
            schedule.step()

    total_flops = np.sum(np.array(train_flops)) + np.sum(np.array(val_flops))
    print("#####" * 30)
    print(
        f"Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))},  "
        f"total: {total_flops}"
    )
    print("#####" * 30)
    print()
    return (total_flops * epochs) / (10 ** 12)


def fit_g_awa2_profiler(
        iteration,
        dataset,
        arch,
        epochs,
        alpha,
        temperature_KD,
        alpha_KD,
        bb,
        model,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        selective_KD_loss,
        logger,
        lambda_lens,
        layer,
        run_id,
        selection_threshold,
        use_concepts_as_pi_input,
        device
):
    train_flops = []
    val_flops = []
    for epoch in range(1):
        model.train()
        out_put_class_bb = torch.FloatTensor().cuda()
        out_put_target_bb = torch.FloatTensor().cuda()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                if dataset == "awa2":
                    train_images, train_concepts, train_attrs, train_image_names, train_y, train_y_one_hot = data_tuple
                elif dataset == "mnist":
                    train_images, train_concepts, train_attrs, train_y, train_y_one_hot = data_tuple

                train_images, train_concepts, train_y, train_y_one_hot = train_images.to(device), \
                                                                         train_concepts.to(device), \
                                                                         train_y.to(torch.long).to(device), \
                                                                         train_y_one_hot.to(device)
                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)
                    train_bb_logits = bb_logits if iteration == 1 else residual(feature_x)
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    if use_concepts_as_pi_input:
                        out_class, out_select, out_aux = model(train_concepts)
                    else:
                        out_class, out_select, out_aux = model(train_concepts, feature_x.to(device))

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                    entropy_loss_elens = entropy_loss(model.explainer)
                    loss_dict = selective_KD_loss(
                        out_class,
                        out_select,
                        train_y,
                        train_bb_logits,
                        entropy_loss_elens,
                        lambda_lens,
                        epoch,
                        device,
                        pi_list
                    )
                    train_selective_loss = loss_dict["selective_loss"]
                    train_emp_coverage = loss_dict["emp_coverage"]
                    train_distillation_risk = loss_dict["distillation_risk"]
                    train_CE_risk = loss_dict["CE_risk"]
                    train_KD_risk = loss_dict["KD_risk"]
                    train_entropy_risk = loss_dict["entropy_risk"]
                    train_emp_risk = loss_dict["emp_risk"]
                    train_cov_penalty = loss_dict["cov_penalty"]

                    train_selective_loss *= alpha
                    aux_distillation_loss = torch.nn.KLDivLoss()(
                        F.log_softmax(out_aux / temperature_KD, dim=1),
                        F.softmax(train_bb_logits / temperature_KD, dim=1)
                    )
                    aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, train_y)
                    aux_KD_loss = (alpha_KD * temperature_KD * temperature_KD) * aux_distillation_loss + \
                                  (1. - alpha_KD) * aux_ce_loss

                    aux_entropy_loss_elens = entropy_loss(model.aux_explainer)
                    train_aux_loss = aux_KD_loss + lambda_lens * aux_entropy_loss_elens
                    train_aux_loss *= (1.0 - alpha)

                    total_train_loss = train_selective_loss + train_aux_loss
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)

                t.set_postfix(epoch='{0}'.format(epoch + 1), training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        out_put_class_bb = torch.FloatTensor().cuda()
        out_put_target_bb = torch.FloatTensor().cuda()
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if dataset == "awa2":
                        val_images, val_concepts, val_attrs, val_image_names, val_y, val_y_one_hot = data_tuple
                    elif dataset == "mnist":
                        val_images, val_concepts, val_attrs, val_y, val_y_one_hot = data_tuple

                    val_images, val_concepts, val_y, val_y_one_hot = val_images.to(device), \
                                                                     val_concepts.to(device), \
                                                                     val_y.to(torch.long).to(device), \
                                                                     val_y_one_hot.to(device)
                    with torch.no_grad():
                        bb_logits, feature_x = get_phi_x(val_images, bb, arch, layer)
                        val_bb_logits = bb_logits if iteration == 1 else residual(feature_x)

                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        if use_concepts_as_pi_input:
                            out_class, out_select, out_aux = model(val_concepts)
                        else:
                            out_class, out_select, out_aux = model(val_concepts, feature_x.to(device))

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    print("Validation forward + backward flops: ", fwbw_flops1)

                    t.set_postfix(epoch='{0}'.format(epoch + 1))
                    t.update()
                    val_flops.append(fwbw_flops1)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        print(
            f"Iteration: {iteration}, Train_flops: {np.sum(np.array(train_flops))}, "
            f"Val_flops: {np.sum(np.array(val_flops))}"
        )


def fit_g_cub_profiler(
        iteration,
        arch,
        epochs,
        alpha,
        temperature_KD,
        alpha_KD,
        bb,
        model,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        selective_KD_loss,
        lambda_lens,
        layer,
        run_id,
        selection_threshold,
        use_concepts_as_pi_input,
        device
):
    print("Running profiler")
    train_flops = []
    val_flops = []
    for epoch in range(1):
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (train_images, train_concepts, _, train_y, train_y_one_hot) in enumerate(train_loader):
                train_images, train_concepts, train_y, train_y_one_hot = train_images.to(device), \
                                                                         train_concepts.to(device), \
                                                                         train_y.to(torch.long).to(device), \
                                                                         train_y_one_hot.to(device)
                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)
                    train_bb_logits = bb_logits if iteration == 1 else residual(feature_x)

                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    if use_concepts_as_pi_input:
                        out_class, out_select, out_aux = model(train_concepts)
                    else:
                        out_class, out_select, out_aux = model(train_concepts, feature_x.to(device))

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                    entropy_loss_elens = entropy_loss(model.explainer)
                    loss_dict = selective_KD_loss(
                        out_class,
                        out_select,
                        train_y,
                        train_bb_logits,
                        entropy_loss_elens,
                        lambda_lens,
                        epoch,
                        device,
                        pi_list
                    )
                    train_selective_loss = loss_dict["selective_loss"]

                    train_selective_loss *= alpha
                    aux_distillation_loss = torch.nn.KLDivLoss()(
                        F.log_softmax(out_aux / temperature_KD, dim=1),
                        F.softmax(train_bb_logits / temperature_KD, dim=1)
                    )
                    aux_ce_loss = torch.nn.CrossEntropyLoss()(out_aux, train_y)
                    aux_KD_loss = (alpha_KD * temperature_KD * temperature_KD) * aux_distillation_loss + \
                                  (1. - alpha_KD) * aux_ce_loss

                    aux_entropy_loss_elens = entropy_loss(model.aux_explainer)
                    train_aux_loss = aux_KD_loss + lambda_lens * aux_entropy_loss_elens
                    train_aux_loss *= (1.0 - alpha)

                    total_train_loss = train_selective_loss + train_aux_loss
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)

                t.set_postfix(epoch='{0}'.format(epoch + 1))
                t.update()

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (val_images, val_concepts, _, val_y, val_y_one_hot) in enumerate(val_loader):
                    val_images, val_concepts, val_y, val_y_one_hot = val_images.to(device), \
                                                                     val_concepts.to(device), \
                                                                     val_y.to(torch.long).to(device), \
                                                                     val_y_one_hot.to(device)

                    with torch.no_grad():
                        bb_logits, feature_x = get_phi_x(val_images, bb, arch, layer)
                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        if use_concepts_as_pi_input:
                            out_class, out_select, out_aux = model(val_concepts)
                        else:
                            out_class, out_select, out_aux = model(val_concepts, feature_x.to(device))

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    print("Validation forward + backward flops: ", fwbw_flops1)

                    t.set_postfix(epoch='{0}'.format(epoch + 1))
                    t.update()
                    val_flops.append(fwbw_flops1)

    print(
        f"Iteration: {iteration}, Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))}, len: {len(train_flops)}"
    )


def fit_awa2_residual(
        dataset,
        iteration,
        epochs,
        bb,
        glt,
        glt_list,
        prev_residual,
        residual,
        optimizer,
        schedule,
        train_loader,
        val_loader,
        kd_Loss,
        logger,
        layer,
        arch,
        run_id,
        selection_threshold,
        use_concepts_as_pi_input,
        device
):
    train_flops = []
    val_flops = []

    for epoch in range(1):
        logger.begin_epoch()
        residual.train()

        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                if dataset == "awa2":
                    train_images, train_concepts, train_attrs, train_image_names, train_y, train_y_one_hot = data_tuple
                elif dataset == "mnist":
                    train_images, train_concepts, train_attrs, train_y, train_y_one_hot = data_tuple

                train_images, train_concepts, train_y, train_y_one_hot = train_images.to(device), \
                                                                         train_concepts.to(device), \
                                                                         train_y.to(torch.long).to(device), \
                                                                         train_y_one_hot.to(device)

                with torch.no_grad():
                    bb_logits, feature_x = get_phi_x(train_images, bb, arch, layer)

                    print(torch.sum(bb_logits.argmax(dim=1) == train_y) / train_y.size(0))

                    train_bb_logits = bb_logits if iteration == 1 else prev_residual(feature_x)

                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    if use_concepts_as_pi_input:
                        out_class, out_select, out_aux = glt(train_concepts)
                    else:
                        out_class, out_select, out_aux = glt(train_concepts, feature_x.to(device))

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, train_concepts)

                    residual_student_logits = residual(feature_x)
                    residual_teacher_logits = train_bb_logits - out_class

                    loss_dict = kd_Loss(
                        student_preds=residual_student_logits,
                        teacher_preds=residual_teacher_logits,
                        target=train_y,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    train_distillation_risk = loss_dict["distillation_risk"]
                    train_CE_risk = loss_dict["CE_risk"]
                    train_KD_risk = loss_dict["KD_risk"]

                    total_train_loss = train_KD_risk
                    optimizer.zero_grad()
                    total_train_loss.backward()
                    optimizer.step()

                events = prof.events()
                fwbw_flops = sum([int(evt.flops) for evt in events])
                train_flops.append(fwbw_flops)
                print("Training forward + backward flops: ", fwbw_flops)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                )
                t.update()

        residual.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    if dataset == "awa2":
                        val_images, val_concepts, val_attrs, val_image_names, val_y, val_y_one_hot = data_tuple
                    elif dataset == "mnist":
                        val_images, val_concepts, val_attrs, val_y, val_y_one_hot = data_tuple

                    val_images, val_concepts, val_y, val_y_one_hot = val_images.to(device), \
                                                                     val_concepts.to(device), \
                                                                     val_y.to(torch.long).to(device), \
                                                                     val_y_one_hot.to(device)
                    with torch.no_grad():
                        bb_logits, val_feature_x = get_phi_x(val_images, bb, arch, layer)
                        val_bb_logits = bb_logits if iteration == 1 else prev_residual(val_feature_x)

                    with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            with_flops=True) as prof1:
                        if use_concepts_as_pi_input:
                            out_class, out_select, out_aux = glt(val_concepts)
                        else:
                            out_class, out_select, out_aux = glt(val_concepts, val_feature_x.to(device))

                    events1 = prof1.events()
                    fwbw_flops1 = sum([int(evt.flops) for evt in events1])
                    print("Validation forward + backward flops: ", fwbw_flops1)

                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                    )
                    t.update()

        # evaluate residual for correctly selected samples (pi < 0.5)
        # should be higher
    print(
        f"Iteration: {iteration}, Train_flops: {np.sum(np.array(train_flops))}, "
        f"Val_flops: {np.sum(np.array(val_flops))}, len: {len(train_flops)}"
    )


def get_phi_x(image, bb, arch, layer):
    if arch == "ResNet50" or arch == "ResNet101" or arch == "ResNet152":
        bb_logits = bb(image)
        # feature_x = get_flattened_x(bb.feature_store[layer], flattening_type)
        feature_x = bb.feature_store[layer]
        return bb_logits, feature_x
    elif arch == "ViT-B_16":
        logits, tokens = bb(image)
        return logits, tokens[:, 0]


def compute_dist(concept_bank, phi):
    margins = (torch.matmul(concept_bank.vectors, phi.T) + concept_bank.intercepts) / concept_bank.norms
    return margins.T
