def create_model(model_config, model_layout, logger):
    if model_config['model_type'] == 'ViT_VaR':
        from model_zoo.feature_extrc.models import ViT_VaR
        model = ViT_VaR(model_type=model_layout['model_type'],
                        pretrained=model_layout['pretrained'],
                        interpolation_type=model_layout['interpolation_type'],
                        num_classes=model_config['num_classes'],
                        n_frames=model_config['num_frames'],
                        dim=model_layout['embedd_dim'],
                        logger=logger,
                        depth=model_layout['depth'],
                        heads=model_layout['heads'],
                        reserve_token_nums=model_config['reserve_token_nums'],
                        discard_ratio=model_config['discard_ratio'],
                        head_fusion=model_config['head_fusion'])

    elif model_config['model_type'] == 'ViT_baseline':
        from model_zoo.feature_extrc.models import ViT_baseline
        model = ViT_baseline(model_type=model_layout['model_type'],
                             pretrained=model_layout['pretrained'],
                             num_classes=model_config['num_classes'],
                             n_frames=model_config['num_frames'],
                             dim=model_layout['embedd_dim'],
                             noPE=model_layout['noPE'],
                             logger=logger,
                             discard_ratio=model_config['discard_ratio'],
                             head_fusion=model_config['head_fusion'])

    return model
