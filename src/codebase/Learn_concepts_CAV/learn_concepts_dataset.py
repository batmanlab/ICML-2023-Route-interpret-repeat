import os
import pickle

import numpy as np
import torch

from Learn_concepts_CAV.concept_loaders import get_concept_loaders
from Learn_concepts_CAV.model_zoo import get_model


# from concepts import learn_concept_bank
# from data import get_concept_loaders

def create_concept_bank(args):
    n_samples = args.n_samples

    # Bottleneck part of model
    backbone = get_model(args)
    backbone = backbone.eval()
    print("Backbone is loaded")

    path = "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/data/CUB_200_2011/class_attr_data_10"
    metadata = pickle.load(open(path+"/train.pkl", "rb"))
    n_concepts = len(metadata[0]["attribute_label"])
    print(n_concepts)
    concept_libs = {C: {} for C in args.C}
    # Get the positive and negative loaders for each concept.

    concept_loaders = get_concept_loaders(args.dataset_name, n_samples=args.n_samples,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers, seed=args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for concept_name, loaders in concept_loaders.items():
        pos_loader, neg_loader = loaders['pos'], loaders['neg']
        # Get CAV for each concept using positive/negative image split
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, args.C, device="cuda")

        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_libs[C][concept_name] = cav_info[C]
            print(concept_name, C, cav_info[C][1], cav_info[C][2])

    # Save CAV results
    for C in concept_libs.keys():
        lib_path = os.path.join(args.out_dir, f"{args.dataset_name}_{args.backbone_name}_{C}_{args.n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved to: {lib_path}")

        total_concepts = len(concept_libs[C].keys())
        print(f"File: {lib_path}, Total: {total_concepts}")
