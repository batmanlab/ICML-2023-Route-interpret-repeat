class Phi:
    def __init__(self, model, layer):
        self.grads = None
        self.model = model
        self.model_activations_store = {}

        def save_activation(layer):
            def hook(module, input, output):
                self.model_activations_store[layer] = output

            return hook

        # get the activations for 3rd Dense block
        if layer == "layer2":
            module = model.model.layer2
            module.register_forward_hook(save_activation(layer))

        elif layer == "layer3":
            module = model.model.layer3
            module.register_forward_hook(save_activation(layer))

        elif layer == "layer4":
            module = model.model.layer4
            module.register_forward_hook(save_activation(layer))
