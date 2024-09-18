from models.transformers.ffn import MLP, GLU
from models.transformers.vision import CLSToken, ConvolutionalPatcher, Classification, number_of_patches
from models.transformers.encoding import Learnable
from models.transformers.attention import MultiheadAttention, Encoder, RopeMultiheadAttention



from torch.nn import Module, ModuleList
from torch.nn import LayerNorm

class ViT(Module):
    def __init__(     
        self,  
        patch_shape: tuple[int, int], 
        model_dimension: int, 
        number_of_q_heads: int, 
        number_of_kv_heads: int, 
        number_of_layers: int, 
        hidden_dimension: int, 
        number_of_channels: int, 
        number_of_classes: int,
        p: float = 0.0,
        max_image_shape: tuple[int, int] = (28, 28),
    ):
        super().__init__()
        self.patcher = ConvolutionalPatcher(model_dimension, patch_shape, number_of_channels)
        self.cls_token = CLSToken(model_dimension)
        self.encoding = Learnable(model_dimension)

        self.layers = ModuleList([Encoder(
            LayerNorm(model_dimension),
            RopeMultiheadAttention(model_dimension, number_of_q_heads, number_of_kv_heads, sequence_lenght_limit=number_of_patches(max_image_shape, patch_shape)),
            LayerNorm(model_dimension),
            GLU(model_dimension, hidden_dimension, model_dimension, p=p, activation='silu')
        ) for _ in range(number_of_layers)])

        self.norm = LayerNorm(model_dimension)
        self.head = Classification(model_dimension, number_of_classes)

    def forward(self, image):
        sequence = self.patcher(image)
        sequence = self.cls_token(sequence)
        sequence = self.encoding(sequence)
        for layer in self.layers:
            sequence = layer(sequence)
        sequence = self.norm(sequence)
        return self.head(sequence)
    
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.adapters import register
from src.adapters.experiments import Experiments, Settings
from src.application.exceptions import ExperimentAlreadyExists
from src.application.commands import TrainOverEpochs, CreateExperiment
from src.application.handlers import handle_create_experiment, handle_training_over_epochs
from logging import basicConfig, INFO, getLogger

basicConfig(level=INFO)
logger = getLogger(__name__)

register('nn','ro-vit-256-4-2-1-128-1-0.0', lambda: ViT(
    patch_shape=(4, 4),
    model_dimension=32,
    number_of_q_heads=4,
    number_of_kv_heads=4,
    number_of_layers=4,
    hidden_dimension=128,
    number_of_channels=1,
    number_of_classes=10,
    p=0.0,
    max_image_shape=(28, 28)
))

register('criterion', 'cross-entropy', lambda: CrossEntropyLoss())
register('optimizer', 'adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))

from pymongo import MongoClient
mongo_client = MongoClient('mongodb://localhost:27017')
database = mongo_client['tests']
directory = 'data/tests/weights'
settings = Settings(device='cuda', workers=4, database=database, directory=directory)
experiments = Experiments(settings)

namemaspace = '119-mnist-bz=256-ro-vit'

command = CreateExperiment(name=f'{namemaspace}-ro-vit-4x4-256-4-2-1-128-1-0.0', nn='ro-vit-256-4-2-1-128-1-0.0', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256)

try:
    handle_create_experiment(command, experiments)


except ExperimentAlreadyExists as e:
    logger.info(e)


command = TrainOverEpochs(experiment=f'{namemaspace}-ro-vit-4x4-256-4-2-1-128-1-0.0', epochs=50, dataset='mnist')
handle_training_over_epochs(command, experiments)