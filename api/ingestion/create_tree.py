import os
import torch
from django.core.management.base import BaseCommand, CommandParser
from transformers import BertTokenizer, BertModel

from ml_deploy.topic_cluster.model.topic_mapper import generate_topic_mappings

class Command(BaseCommand):
    help = 'Train the Siamese Topic Similairy Model'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('mapping_path', type=str, help='Path to the Category Mapping (.xslx)')
        parser.add_argument('sheet_name', type=str, help='Name of the Excel Sheet containing data', default='Sheet2')

    def handle(self, *args, **kwargs):
        mapping_path = kwargs['mapping_path']
        sheet_name = kwargs['sheet_name']
        if mapping_path is not None and sheet_name is not None:
            generate_topic_mappings(mapping_path, sheet_name)
            self.stdout.write(self.style.SUCCESS(f'Generate Category Mappings Successful.'))
        else:
            self.stdout.write(self.style.ERROR("No category mapping provided"))