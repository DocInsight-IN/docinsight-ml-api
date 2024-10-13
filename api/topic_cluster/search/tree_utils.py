import pandas as pd
import numpy as np
import logging
import pickle

from ml_deploy.apps import encoding_model

EXCEL_FILE = 'category_mapping.xlsx'
SHEET_NAME = 'Sheet2'


class TreeNode:
    def __init__(self, category, category_code, stage, org_code, parent=None):
        self.category = category
        self.category_emb = None
        self.category_code = category_code
        self.stage = stage
        self.org_code = org_code
        self.parent = parent
        self.children = []
        self.documents = []
        self.cluster_labels = None

    def add_child(self, child):
        self.children.append(child)

    def add_document(self, document):
        self.documents.append(document)

    def generate_embeddings(self, encoding_model):
        if self.category == 'TopicRoot' and self.category_code == 'Root':
            logging.debug('Root Node encountered Skipping!!')
            return
        logging.debug(f'Generating embeddings for Category : {self.category_code} ...')
        self.category_emb = np.array(encoding_model.encode(self.category))

def create_tree(excel_file=EXCEL_FILE, sheet_name=SHEET_NAME):
    tree = {}
    df = pd.read_excel(excel_file, sheet_name)

    for index, row in df.iterrows():
        subject_code = row['Code']
        parent_subject_code = row['ParentCode']

        parent_node = None
        if parent_subject_code in tree:
            parent_node = tree[parent_subject_code]

        node = TreeNode(
            category=row['Description'],
            category_code=subject_code,
            stage=row['Stage'],
            org_code=row['OrgCode']
        )

        node.generate_embeddings()

        if subject_code not in tree:
            tree[subject_code] = node

        # update the parent info for the child node
        if parent_node:
            parent_node.add_child(node)

    root_nodes = [code for code, node in tree.values() if not node.parent]
    artificial_root = TreeNode(category='TopicRoot', category_code='Root', stage='Root', org_code=None)
    for root in root_nodes:
        artificial_root.add_child(tree[root])

    return artificial_root

def assign_category_labels(row, root_node):
    categories = []
    code = row['Code']
    category_node = find_category_node(root_node, code)
    if category_node:
        categories.append(category_node.category)
        while category_node.parent:
            categories.append(category_node.parent.category)
            category_node = category_node.parent
    return categories

def find_category_node(node, code):
    if node.category_code == code:
        return node
    for child in node.children:
        found_node = find_category_node(child, code)
        if found_node:
            return found_node
    return None

def save_tree(tree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)

def load_tree(filename):
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    return tree