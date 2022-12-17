"""
@created by: heyao
@created at: 2022-08-25 00:48:52
"""
import torch.nn as nn
from transformers import PreTrainedModel


def init_weights(module, kaiming=False, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        if kaiming:
            nn.init.kaiming_normal_(module.weight.data)
        else:
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def differential_learning_rate(model, encoder_lr, decoder_lr, weight_decay=0.0, lr_factor=2):
    """
    TODO: find a better way to do this.
    :param model:
    :param encoder_lr:
    :param decoder_lr:
    :param weight_decay:
    :param lr_factor:
    :return:
    """
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    no_decay = [".bias", "LayerNorm.bias", "LayerNorm.weight"]
    name = "backbone"
    if hasattr(model.backbone.encoder, "layer"):
        num_layers = len(model.backbone.encoder.layer)
    elif hasattr(model.backbone.encoder, "layers"):
        num_layers = len(model.backbone.encoder.layers)
    elif hasattr(model.backbone.encoder, "blocks"):
        num_layers = len(model.backbone.encoder.blocks)
    else:
        print(model)
        raise ValueError("")
    print(f"model {model.__class__.__name__} has {num_layers} encoder layers")
    print(f"model {model.__class__.__name__} has {len(list(model.named_parameters()))} trainable parameter groups.")
    sub_layers = int(num_layers / 3)
    special_terms = ["backbone.encoder.LayerNorm.weight", "backbone.encoder.LayerNorm.bias"]
    bart_special = [
        'backbone.shared.weight', 'backbone.encoder.layernorm_embedding.weight',
        'backbone.encoder.layernorm_embedding.bias', 'backbone.encoder.embed_positions.weight',
    ]
    bart_special2 = [
        'backbone.decoder.embed_positions.weight', 'backbone.decoder.layernorm_embedding.weight',
        'backbone.decoder.layernorm_embedding.bias'
    ]
    xlarge_special = ['backbone.encoder.conv.LayerNorm.weight', 'backbone.encoder.conv.conv.weight']
    xlarge_special_bias = ['backbone.encoder.conv.conv.bias', 'backbone.encoder.conv.LayerNorm.bias']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    'embeddings' in n and not any(nd in n for nd in no_decay)],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    'embeddings' in n and any(nd in n for nd in no_decay)],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if n in special_terms],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    not any(nd in n for nd in no_decay) and
                    any(f".{i}." in n for i in range(sub_layers))],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    not any(nd in n for nd in no_decay) and
                    any(f".{i}." in n for i in range(sub_layers, sub_layers * 2))],
         'lr': encoder_lr / lr_factor, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    not any(nd in n for nd in no_decay) and
                    any(f".{i}." in n for i in range(sub_layers * 2, sub_layers * 3))],
         'lr': encoder_lr, 'weight_decay': weight_decay},

        {'params': [p for n, p in model.named_parameters() if name in n and
                    any(nd in n for nd in no_decay) and
                    any(f".{i}." in n for i in range(sub_layers))],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    any(nd in n for nd in no_decay) and
                    any(f".{i}." in n for i in range(sub_layers, sub_layers * 2))],
         'lr': encoder_lr / lr_factor, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    any(nd in n for nd in no_decay) and
                    any(f".{i}." in n for i in range(sub_layers * 2, sub_layers * 3))],
         'lr': encoder_lr, 'weight_decay': 0.0},

        {'params': [p for n, p in model.named_parameters() if name not in n],
         'lr': decoder_lr, 'weight_decay': 0.0},

        {'params': [p for n, p in model.named_parameters() if
                    any(name == n for name in xlarge_special)],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    any(name == n for name in xlarge_special_bias)],
         'lr': 0, 'weight_decay': 0.0},

        {'params': [p for n, p in model.named_parameters() if
                    any(name == n for name in bart_special)],
         'lr': encoder_lr / lr_factor / lr_factor, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    any(name == n for name in bart_special2)],
         'lr': 0, 'weight_decay': 0.0},
    ]
    ensure_all_training = len(list(model.named_parameters())) == sum(len(i['params']) for i in optimizer_parameters)
    assert ensure_all_training, "some param not training."
    return optimizer_parameters


def reinit_last_layers(model: PreTrainedModel, num_layers: int, kaiming=False):
    """Re-initialize the last-k transformer layers.
    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
    if num_layers <= 0:
        return
    if hasattr(model.encoder, "layer"):
        model.encoder.layer[-num_layers:].apply(lambda x: init_weights(x, kaiming=kaiming))
    elif hasattr(model.encoder, "layers"):
        model.encoder.layers[-num_layers:].apply(lambda x: init_weights(x, kaiming=kaiming))
    elif hasattr(model.encoder, "blocks"):
        model.encoder.blocks[-1][-num_layers:].apply(lambda x: init_weights(x, kaiming=kaiming))
    else:
        print(model)
        raise ValueError("can't re-init.")


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0, weight_decay_head=False):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    name = "backbone"
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if name in n and
                    not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if name in n and
                    any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if name not in n],
         'lr': decoder_lr, 'weight_decay': 0.0 if not weight_decay_head else weight_decay}
    ]
    return optimizer_parameters
