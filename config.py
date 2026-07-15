class cfg:
    def __init__(self):
        self.model_name = "my_model"
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.num_epochs = 100
        self.data_path = "./data"
        self.save_path = "./checkpoints"

config = cfg()