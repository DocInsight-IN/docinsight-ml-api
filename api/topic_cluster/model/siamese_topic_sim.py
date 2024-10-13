import torch
from sklearn.metrics.pairwise import cosine_similarity
from .siamese_bert import SiameseBERT, load_model
from .topic_mapper import TopicMapper

def highlight_words_with_attention(text, attention_weights):
    words = text.split()
    highlighted_words = []

    for word , weight in zip(words, attention_weights):
        intensity = int(weight * 255)
        # Create HTML span element with inline CSS
        highlighted_word = f'<span style="background-color: rgba(255, 0, 0, {intensity});">{word}</span>'
        highlighted_words.append(highlighted_words)
    highlighted_text = ' '.join(highlighted_words)
    return highlighted_text

class SiameseTopicSimilarity:
    def __init__(self, topic_mapper: TopicMapper, model_save_path: str) -> None:
        self.topic_mapper = topic_mapper
        self.tokenizer, self.siamese_bert = load_model(model_save_path)

    def _siamese_predict(self, texts):
        inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            outputs = self.siamese_bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        return embeddings
    
    def _topic_similarity_tree_search(self, text_embeddings, text_size):
        stack = [(self.topic_mapper.topic_root, 1.0, None)]
        best_sim = -1
        best_match_node = None
        best_match_children = []

        text_embeddings = torch.tensor(text_embeddings)
        attention_weights = torch.zeros_like(text_size)

        while stack:
            node, parent_sim, _ = stack.pop()
            node_emb = torch.tensor(node.category_emb)
            # calc cosine similarity between the node and text embeddings
            node_sim = cosine_similarity(node.category_emb, text_embeddings)
            # update the attention weights
            attention_weights = torch.max(attention_weights, node_sim.unsqueeze(0))
            combined_sim = node_sim * parent_sim
            if combined_sim > best_sim:
                best_sim = combined_sim
                best_match_node = node
                best_match_children = [child for child in node.children]

            for child in reversed(node.children):
                stack.append((child, node_sim, node))

        # normalize the attention weights
        attention_weights = attention_weights / attention_weights.sum()
        return best_sim, best_match_node, best_match_children, attention_weights
    
    def predict(self, texts):
        """
        Predicts the best matching category and also returns the best matching words that contributes to the selection of the category
        """
        all_embeddings = self._siamese_predict(texts)
        batch_results = []
        for idx, embeddings in enumerate(all_embeddings):
            best_sim, best_match_node, best_match_children, attn_weights = self._topic_similarity_tree_search(embeddings)
            num_words = len(texts[idx].split())
            # divide attention weights evenly among the words
            text_attention = attn_weights[:num_words]
            highlighted_html = highlight_words_with_attention(texts[idx], text_attention)
            batch_results.append((best_sim, best_match_node, best_match_children, highlighted_html))
        return batch_results
    