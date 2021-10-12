import torch
from transformers import pipeline as hf_pipeline

from flair.models import TARSTagger
from flair.data import Sentence


class ZeroShotNERPipeline:
    def __init__(
        self,
        model='tars-ner',
    ):
        self.model = TARSTagger.load(model)

    def __call__(self, query, labels):
        self.model.add_and_switch_to_new_task('task', labels, label_type='ner', force_switch=True)

        if not isinstance(query, list):
            inputs = [query]
        else:
            inputs = query

        inputs = [Sentence(s) for s in inputs]
        self.model.predict(inputs)
        outputs = [sentence.to_tagged_string("ner") for sentence in inputs]

        return outputs


class SemanticSearchPipeline:
    def __init__(
        self,
        model,
        context=None,
        device="cpu",
    ):
        self.model = model
        self.device = device
        self.fx = hf_pipeline("feature-extraction", model=model)
        self.context_emb = None

        # Compute embeddings for context, if given
        if context is not None:
            self.context_emb = self.compute_embeddings(context)

    def __call__(self, query, context=None, temperature=0.01, return_probs=True):
        query_emb = self.compute_embeddings(query)

        if context is not None:
            self.context_emb = self.compute_embeddings(context)
        elif self.context_emb is None:
            raise Exception("No context was given.")

        sim = torch.einsum("ij,kj->ik", query_emb, self.context_emb)

        if return_probs:
            sim_probs = torch.softmax(sim / temperature, dim=-1)
            return sim_probs
        else:
            return sim

    def compute_embeddings(self, input):
        input_emb = torch.mean(torch.tensor(self.fx(input)).to(self.device), dim=1)
        input_emb = input_emb / torch.linalg.norm(
            input_emb, ord=2, axis=-1, keepdims=True
        )
        return input_emb