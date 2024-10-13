import os
import torch
from django.core.management.base import BaseCommand, CommandParser
from transformers import BertTokenizer, BertModel

from ml_deploy.topic_cluster.model.siamese_bert import train_model, save_model, SiameseBERT
from ml_deploy.topic_cluster.data.exploration import create_df_for_siamese
from ml_deploy.topic_cluster.data.preprocess import preprocess_for_siamese

def train_and_save_siamese_bert(dataset_path, topic_mapping_path, model_save_dir):
    data_df = create_df_for_siamese(dataset_path, topic_mapping_path)
    pairs, labels = preprocess_for_siamese(data_df)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    siamese_model = SiameseBERT(bert_model)
    train_model(pairs, labels)
    save_model(tokenizer, siamese_model, model_save_dir)
    


class Command(BaseCommand):
    help = 'Train the Siamese Topic Similairy Model'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('dataset_path', type=str, help='Path to the grievance dataset file (JSON)')
        parser.add_argument('mapping_path', type=str, help='Path to the Category Mapping (.xslx)')
        parser.add_argument('save_path', type=str, help='Siamese Model save path')

    def handle(self, *args, **kwargs):
        dataset_path = kwargs['dataset_path']
        mapping_path = kwargs['mapping_path']
        model_save_path = kwargs['save_path']
        if dataset_path is not None and mapping_path is not None:
            train_and_save_siamese_bert(dataset_path, mapping_path, model_save_path)
            self.stdout.write(self.style.SUCCESS(f'Model Training Complete : {model_save_path}'))
        else:
            self.stdout.write(self.style.ERROR("No dataset or category mapping provided"))