import torch
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
import torch.nn.functional as F

from .utils import mean_pooling

class KeywordEncoderInferenceModel(pl.LightningModule):
    """
    Class for Keyword Encoder Model
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), max_len: int = 60):
        """
        Initialize the class and load the model

        Parameters:
            model_name (str): Name of the model to load
            device (torch.device): Device to use for inference
            max_len (int): Maximum length of the input text
        """
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.enc = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.enc.eval()
        self.enc.to(device)

    def forward(self, texts: list) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            texts (list): List of texts to embed

        Returns:
            torch.Tensor: Embeddings of the texts
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt').to(self.enc.device)

            model_output = self.enc(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings
    
    def calculate_similarity(self, texts: list) -> list:
        """
        Calculate the similarity between two or more texts

        Parameters:
            texts (list): List of texts to compare

        Returns:
            list: Similarities between the texts
        """
        assert len(texts) > 1, "Please provide at least two texts to compare"
        embeddings = self.forward(texts)
        similarities = torch.mm(embeddings, embeddings.t())
        similarities = similarities.cpu().numpy()
        return similarities
