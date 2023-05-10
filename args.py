class Args():
    def __init__(self):
        self.k = 20
        self.emb_dims = 1024
        self.dropout = 0.3
        self.feature_extractor_path = 'checkpoints/model.1024.t7'
        self.load_pretrained_feature_extractor = False
        self.freeeze_feature_extractor = False
        self.attention_pooling = True
