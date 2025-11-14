from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GREPBiasIRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GREPBiasIRRetrieval",
        dataset={
            "path": "lyon-nlp/grep-bias-ir",
            "revision": "main",
        },
        description=(
            "GREP-BiasIR is an information retrieval dataset designed to evaluate gender bias in retrieval systems. "
            "The dataset contains queries and documents annotated for stereotypical content across multiple dimensions "
            "including Appearance, Career, Child Care, Cognitive Capabilities, Domestic Work, Physical Capabilities, "
            "and Sex & Relationship. Each document is labeled with content_gender and exp_stereotype annotations."
        ),
        reference="https://huggingface.co/datasets/lyon-nlp/grep-bias-ir",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=None,
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
    )

    def load_data(self):
        self.queries = {}
        self.relevant_docs = {}
        self.corpus = {}

        corpus = load_dataset(self.metadata.dataset["path"], name="docs", split="test")
        queries_ds = load_dataset(
            self.metadata.dataset["path"], name="queries", split="test"
        )
        qrels_ds = load_dataset(
            self.metadata.dataset["path"], name="qrels", split="test"
        )

        self.corpus["test"] = {
            str(row["d_id"]): "Title: " + row["doc_title"] + " Text: " + row["document"]
            for row in corpus
        }
        self.queries["test"] = {str(row["q_id"]): row["query"] for row in queries_ds}

        # Build relevant_docs by aggregating all relevant documents per query
        temp_relevant_docs = {}
        for row in qrels_ds:
            q_id = str(row["q_id"])
            d_id = str(row["d_id"])
            if q_id not in temp_relevant_docs:
                temp_relevant_docs[q_id] = {}
            temp_relevant_docs[q_id][d_id] = row["relevant"]
        self.relevant_docs["test"] = temp_relevant_docs

        self.bias_attributes = {
            "content_gender": {
                str(row["d_id"]): row["content_gender"] for row in corpus
            },
            "exp_stereotype": {
                str(row["d_id"]): row["exp_stereotype"] for row in corpus
            },
        }

        self.data_loaded = True
