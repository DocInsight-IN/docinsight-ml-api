from typing import List, Dict
from collections import deque
import logging
import numpy as np
import pandas as pd

from ml_deploy.topic_cluster.search.tree_utils import TreeNode, create_tree, save_tree, load_tree

def find_node_by_category_code(root_node, category_code):
    if root_node.category_code == category_code:
        return root_node
        
    for child in root_node.children:
        found_node = find_node_by_category_code(child, category_code)
        if found_node:
            return found_node

def generate_embeddings(node):
    node.generate_embeddings()
    for child in node.children:
        generate_embeddings(child)

def generate_topic_mappings(mapping_file, sheet_name, save_path):
    logging.debug('Creating Category Tree...')
    tree_root = create_tree(mapping_file, sheet_name)
    logging.debug('Finished!')
    logging.debug('Creating Category Description embeddings...')
    generate_embeddings(tree_root)
    logging.debug('Finished!')
    save_tree(tree_root, save_path)
    logging.debug(f'Saved tree at: {save_path}')


class TopicMapper:
    """ Keep track of Topic tree hierarchy """

    def __init__(self, topic_root: TreeNode=None):
        self.topic_root = topic_root

    def _search_mapping(self, category_code, category_description):
        queue = deque([self.topic_root])
        while queue:
            node = queue.popleft()
            if node.category_code == category_code:
                return (node.category, node.category_emb)
            if node.category == category_description:
                return (node.category, node.category_emb)
            queue.extend(node.children)
        return (None, None)

    def get_mappings(self, category_code, category_description):
        data = self._search_mapping(category_code=category_code, category_description=category_description)
        return data

    def add_new_topic(self, parent_code, category, category_code, stage, org_code):
        parent_node = find_node_by_category_code(self.topic_root, parent_code)
        new_node = TreeNode(category, category_code, stage, org_code, parent_node)
        parent_node.add_child(new_node)

    def create_tree(self, mapping_file, sheet_name):
        self.topic_root = create_tree(excel_file=mapping_file, sheet_name=sheet_name)

    def load_tree(self, filepath):
        self.topic_root = load_tree(filepath)

    # def _depth_first_cluster(self, root_node, epsilon, min_samples):
    #     stack = [root_node]
    #     while stack:
    #         current_node = stack.pop()
                
    #         if current_node.children:
    #             embeddings = [child.category_emb for child in current_node.children]
    #             cluster_labels = self.hdbscan_cluster(embeddings, epsilon, min_samples)

    #             for child, label in zip(current_node.chidlren, cluster_labels):
    #                 child.cluster_label = label
                
    #             stack.extend(current_node.children)