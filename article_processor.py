"""News article processing, categorisation and summarisation.

This module contains helper classes that take raw article records
downloaded from the Event Registry API and perform a series of
post‑processing steps:

* **Classification** – assign each article to one of a handful of
  high‑level categories (``sports``, ``lifestyle``, ``music``, or
  ``finance``).  We implement a zero‑shot classifier using
  HuggingFace's inference API with the ``facebook/bart‑large‑mnli``
  model.  The model accepts the article text and a list of candidate
  labels and returns a probability for each label.  We pick the label
  with the highest score.

* **Summarisation** – generate a concise summary of the article body.
  We use the ``facebook/bart‑large‑cnn`` model via the HuggingFace
  inference API.  The summariser is applied to each article body and
  returns a one‑paragraph summary.

* **Embedding & Clustering** – embed articles into a vector space
  using a sentence transformer (``all‑MiniLM‑L6‑v2``) and group
  together articles reporting the same story.  Articles within a
  cluster are considered duplicates or near‑duplicates.  Clusters are
  created using a simple greedy procedure based on cosine similarity
  thresholds.  More sophisticated clustering (e.g. DBSCAN) can be
  substituted here if desired.

* **Highlight extraction** – select the most important stories within
  each category by counting how many unique sources reported the
  story and checking for the presence of key phrases like ``breaking
  news``.  The top stories per category are returned as highlights.

The heuristics implemented here are intentionally simple to make the
pipeline easy to understand.  You can improve the performance by
using more advanced language models, tuning the similarity threshold
used for clustering or implementing alternative ranking strategies.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np

import requests
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import faiss


# Data class for processed articles
@dataclasses.dataclass
class ProcessedArticle:
    uri: str
    title: str
    body: str
    source: str
    date: str
    category: str
    summary: str
    embedding: np.ndarray


# Data class for highlight stories
@dataclasses.dataclass
class Highlight:
    category: str
    title: str
    summary: str
    frequency: int
    sources: List[str]
    article_uris: List[str]


class ArticleProcessor:
    """Processes, classifies and summarises articles.

    Parameters
    ----------
    hf_token: str
        Personal access token for HuggingFace Inference API.
    summary_model: str, optional
        Name of the HuggingFace model to use for summarisation.
    classification_model: str, optional
        Name of the HuggingFace model to use for zero‑shot classification.
    categories: Optional[List[str]]
        List of categories to classify into.  If not provided a default
        list of ``['sports', 'lifestyle', 'music', 'finance']`` is used.
    similarity_threshold: float
        Cosine similarity threshold for grouping articles into the same
        cluster.  A higher value yields tighter clusters.
    """

    def __init__(
        self,
        hf_token: str,
        summary_model: str = "facebook/bart-large-cnn",
        classification_model: str = "facebook/bart-large-mnli",
        categories: Optional[List[str]] = None,
        similarity_threshold: float = 0.82,
    ) -> None:
        self.hf_token = hf_token
        self.summary_model = summary_model
        self.classification_model = classification_model
        self.categories = categories or ["sports", "lifestyle", "music", "finance"]
        self.similarity_threshold = similarity_threshold
        # instantiate embedding model once; this will download and cache
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # ----------------------------- HF Helpers -----------------------------
    def _hf_post(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper to call HuggingFace Inference API.

        Parameters
        ----------
        model: str
            Repository ID of the model to call.
        payload: dict
            JSON serialisable payload.

        Returns
        -------
        dict
            JSON response parsed into a Python dictionary.
        """
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API call failed: {response.status_code} {response.text[:200]}"
            )
        return response.json()

    def classify_article(self, text: str) -> str:
        """Classify a piece of text into one of the pre‑defined categories.

        Uses the zero‑shot classification capability of
        ``facebook/bart-large-mnli`` via the HuggingFace inference
        endpoint.  The model returns a list of labels and scores; the
        highest scoring label is selected and returned.
        """
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": self.categories},
        }
        result = self._hf_post(self.classification_model, payload)
        # result is a dict with keys 'labels' and 'scores'
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        if not labels:
            return "unknown"
        # pick the label with the highest score
        max_idx = int(np.argmax(scores))
        return labels[max_idx]

    def summarise_article(self, text: str) -> str:
        """Produce a short summary of a news article.

        The summarisation model expects a single string input and
        returns a summary string.  The BART CNN model tends to
        generate multi‑sentence summaries; for our highlights we only
        need the first paragraph, so we join the output lines.
        """
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": 150,
                "min_length": 30,
                "do_sample": False,
            },
        }
        result = self._hf_post(self.summary_model, payload)
        # result may be a list of generated summaries
        if isinstance(result, list):
            summary = result[0].get("summary_text", "")
        elif isinstance(result, dict) and "summary_text" in result:
            summary = result["summary_text"]
        else:
            summary = str(result)
        return summary.strip()

    # --------------------------- Processing Steps ---------------------------
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Compute vector embeddings for a list of strings.

        This method delegates to ``HuggingFaceEmbeddings`` which
        downloads the embedding model from HuggingFace the first time it is
        used.  The result is a 2D array of shape ``(n, d)``.
        """
        return np.array(self.embedding_model.embed_documents(texts))

    def process_articles(self, articles: Iterable[Dict[str, Any]]) -> List[ProcessedArticle]:
        """Run classification, summarisation and embedding on raw articles.

        Each input article dictionary should contain at least the keys
        ``title``, ``body``, ``source`` and ``date``.  Classification
        operates on the concatenation of title and body to provide
        additional context; summarisation operates solely on the body.

        Returns
        -------
        List[ProcessedArticle]
            A list of processed article instances ready for clustering
            and storage.
        """
        processed: List[ProcessedArticle] = []
        # Precompute embeddings for efficiency; we embed the article bodies
        bodies: List[str] = [a.get("body", "") for a in articles]
        embeddings = self.embed_text(bodies)
        for idx, article in enumerate(articles):
            title = article.get("title", "")
            body = article.get("body", "")
            # Determine category using title+body for more context
            combined_text = f"{title}. {body}"
            try:
                category = self.classify_article(combined_text)
            except Exception:
                category = "unknown"
            try:
                summary = self.summarise_article(body)
            except Exception:
                # Fallback: use the first 100 characters as a naive summary
                summary = body[:200] + "..."
            processed.append(
                ProcessedArticle(
                    uri=article.get("uri", article.get("id", f"art_{idx}")),
                    title=title,
                    body=body,
                    source=article.get("source", {}).get("title", article.get("source", "")),
                    date=article.get("date", ""),
                    category=category,
                    summary=summary,
                    embedding=embeddings[idx],
                )
            )
        return processed

    def cluster_articles(self, articles: List[ProcessedArticle]) -> List[List[ProcessedArticle]]:
        """Group similar articles into clusters based on cosine similarity.

        A greedy algorithm is used: we iterate through articles in the
        order given and assign each article to an existing cluster if
        its similarity to the cluster centroid exceeds the threshold.
        Otherwise a new cluster is created.  This approach is fast
        and sufficient for small numbers of daily articles.
        """
        clusters: List[List[ProcessedArticle]] = []
        centroids: List[np.ndarray] = []
        for art in articles:
            if not clusters:
                clusters.append([art])
                centroids.append(art.embedding)
                continue
            # compute similarities to existing centroids
            sims = cosine_similarity(
                art.embedding.reshape(1, -1), np.stack(centroids)
            )[0]
            max_idx = int(np.argmax(sims))
            if sims[max_idx] >= self.similarity_threshold:
                clusters[max_idx].append(art)
                # update centroid
                members = clusters[max_idx]
                centroids[max_idx] = np.mean(
                    [m.embedding for m in members], axis=0
                )
            else:
                clusters.append([art])
                centroids.append(art.embedding)
        return clusters

    def build_highlights(self, clusters: List[List[ProcessedArticle]], top_k: int = 5) -> List[Highlight]:
        """Construct a highlight summary for each cluster and select top stories.

        We score each cluster based on the number of unique sources and
        whether the phrase ``breaking news`` appears in any article title
        or summary.  The cluster title and summary are chosen from
        the article with the longest body (as a proxy for detail).

        Parameters
        ----------
        clusters: List[List[ProcessedArticle]]
            Groups of similar articles as produced by ``cluster_articles``.
        top_k: int
            Maximum number of highlight stories to return per category.

        Returns
        -------
        List[Highlight]
            Highlight objects sorted by descending score.  The list
            contains at most ``top_k`` entries for each category.
        """
        highlights: List[Highlight] = []
        for cluster in clusters:
            # Determine unified category via majority vote
            cat_counts = collections.Counter([art.category for art in cluster])
            category = cat_counts.most_common(1)[0][0]
            # Count unique sources
            sources = list({art.source for art in cluster if art.source})
            freq = len(sources)
            # Pick representative article (longest body)
            rep = max(cluster, key=lambda a: len(a.body))
            title = rep.title
            summary = rep.summary
            # Score: frequency plus bonus if 'breaking news' appears
            bonus = any(
                "breaking" in (art.title + art.summary).lower()
                for art in cluster
            )
            score = freq + (1 if bonus else 0)
            highlights.append(
                (category, score, Highlight(
                    category=category,
                    title=title,
                    summary=summary,
                    frequency=freq,
                    sources=sources,
                    article_uris=[art.uri for art in cluster],
                ))
            )
        # sort by category then score descending
        highlights.sort(key=lambda x: (x[0], -x[1]))
        final_highlights: List[Highlight] = []
        # collect top_k per category
        cat_counts: Dict[str, int] = {}
        for cat, _, hl in highlights:
            if cat_counts.get(cat, 0) < top_k:
                final_highlights.append(hl)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        return final_highlights