from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HateXplainClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HateXplainClassification",
        description="HateXplain is a benchmark hate speech dataset covering multiple aspects of the issue. Each post is annotated from three perspectives: 3-class classification (hate, normal, or offensive), target community, and rationales for the labeling decision.",
        reference="https://arxiv.org/abs/2012.10289",
        dataset={
            "path": "lyon-nlp/hatexplain",
            "revision": "099763426e1e455de9a0b206768ba17224446408",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-12-31",
        ),  # Based on paper publication date (2020)
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{mathew2020hatexplain,
  author = {Binny Mathew and Punyajoy Saha and Seid Muhie Yimam and Chris Biemann and Pawan Goyal and Animesh Mukherjee},
  publisher = {AAAI conference on artificial intelligence},
  title = {HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  year = {2021},
}
""",
        prompt="Classify the given social media post into one of three categories: hate speech, normal, or offensive",
    )

    samples_per_label = 16

    # Enable fairness metrics computation by specifying the column with group information
    # This can be any categorical attribute: demographic, linguistic, domain-based, etc.
    # For HateXplain, we use 'target_community' to analyze bias across different target groups
    sensitive_attribute_column_name = "target_community"

    def dataset_transform(self):
        """Transform the dataset to the format expected by MTEB.

        The original dataset has:
        - 'post_tokens': list of tokens
        - 'annotators': list of annotators with 'label' field (0=hatespeech, 1=normal, 2=offensive)
        - 'annotators': list of annotators with 'target' field (list of target communities)

        We need to:
        - Join tokens into text
        - Get majority vote from annotators for the label
        - Extract target community information for bias analysis across groups
        - Map to text and label columns
        """
        from collections import Counter

        def process_sample(sample):
            # Join tokens into text
            text = " ".join(sample["post_tokens"])

            # Get majority vote from annotators
            labels = sample["annotators"]["label"]
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]

            # Map label to string: 0=hatespeech, 1=normal, 2=offensive
            label_map = {0: "hatespeech", 1: "normal", 2: "offensive"}
            label_str = label_map[majority_label]

            # make it binary classification (hate vs non-hate)
            # Use 1 for hate, 0 for normal (numeric labels required for AP score)
            if label_str == "hatespeech" or label_str == "offensive":
                label_int = 1
            else:
                label_int = 0

            # Extract target community (for bias analysis across different target groups)
            # Get the most common target from annotators
            targets = []
            for annotator in sample["annotators"]["target"]:
                # Each annotator may mark multiple targets
                if isinstance(annotator, list) and annotator:
                    targets.extend(annotator)
                elif annotator:
                    targets.append(annotator)

            # Use the most common target, or "none" if no targets identified
            if targets:
                target_counts = Counter(targets)
                target_community = target_counts.most_common(1)[0][0]
            else:
                target_community = "none"

            return {
                "text": text,
                "label": label_int,
                "target_community": target_community,
            }

        # Apply transformation to all splits
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                process_sample,
                remove_columns=self.dataset[split].column_names,
            )
