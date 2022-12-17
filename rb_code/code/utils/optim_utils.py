from torch.optim import AdamW

# def get_optimizer_grouped_parameters_no_llrd(model, config):

#     no_decay = ['bias', "LayerNorm.bias", "LayerNorm.weight"]
#     backbone_params = model.backbone.named_parameters()

#     optimizer_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if "backbone" not in n],
#             "lr": config["decoder_lr"],
#             "weight_decay": config["weight_decay"]*1e-3,
#         },
#         {
#             "params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
#             "lr": config["encoder_lr"],
#             "weight_decay": config["weight_decay"],
#         },
#         {
#             "params": [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
#             "lr": config["encoder_lr"],
#             "weight_decay": 0.0,
#         },
#     ]

#     return optimizer_parameters


def get_optimizer_grouped_parameters_no_llrd(model, config):

    no_decay = ['bias', "LayerNorm.bias", "LayerNorm.weight"]
    backbone_params_1 = model.backbone_1.named_parameters()
    backbone_params_2 = model.backbone_2.named_parameters()
    # backbone_params_3 = model.backbone_3.named_parameters()

    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": config["decoder_lr"],
            "weight_decay": config["weight_decay"]*1e-3,
        },
        {
            "params": [p for n, p in backbone_params_1 if not any(nd in n for nd in no_decay)],
            "lr": config["encoder_lr"],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in backbone_params_1 if any(nd in n for nd in no_decay)],
            "lr": config["encoder_lr"],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in backbone_params_2 if not any(nd in n for nd in no_decay)],
            "lr": config["encoder_lr"],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in backbone_params_2 if any(nd in n for nd in no_decay)],
            "lr": config["encoder_lr"],
            "weight_decay": 0.0,
        },
        # {
        #     "params": [p for n, p in backbone_params_3 if not any(nd in n for nd in no_decay)],
        #     "lr": config["encoder_lr"],
        #     "weight_decay": config["weight_decay"],
        # },
        # {
        #     "params": [p for n, p in backbone_params_3 if any(nd in n for nd in no_decay)],
        #     "lr": config["encoder_lr"],
        #     "weight_decay": 0.0,
        # },
    ]

    return optimizer_parameters


def get_optimizer_grouped_parameters_with_llrd(model, config):
    """layerwise learning rate decay implementation
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # backbone_params = model.backbone.named_parameters()

    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": config["decoder_lr"],
            "weight_decay": config["weight_decay"]*1e-3,
        },
    ]

    # initialize lrs for backbone layers
    try:
        layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    except Exception as e:
        print("creating optimizer for SetFit...")
        layers = [model.sentence_transformer.backbone.embeddings] + list(model.sentence_transformer.backbone.encoder.layer)

    layers.reverse()
    lr = config["encoder_lr"]

    for layer in layers:
        lr *= config["llrd"]

        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["weight_decay"],
                "lr": lr,
            },

            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


def get_optimizer(model, config):
    """optimizer for model training
    """
    config = config["optimizer"]

    if config["use_llrd"]:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters_with_llrd(model, config)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters_no_llrd(model, config)

    optimizer_kwargs = {
        "betas": (config["beta1"], config["beta2"]),
        "eps": config['eps'],
        "lr": config["encoder_lr"]
    }

    if config["use_bnb"]:
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )
        return optimizer
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )

    return optimizer
