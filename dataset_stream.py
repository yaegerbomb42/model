"""
dataset_stream.py
-----------------
Multi-source live dataset streamer for background distillation training.

Sources (weighted random selection each step):
  - Wikipedia random article           (always available, broad factual)
  - Hacker News top stories            (tech/current events, no auth)
  - Reddit JSON API                    (r/science, r/worldnews, r/machinelearning, r/technology)
  - NewsAPI                            (real-time headlines + body, requires NEWS_API_KEY)
  - Arxiv recent papers (last 24h)    (cutting-edge ML/science reasoning)
  - HuggingFace FineWeb-Edu stream    (high-quality educational curated text)

Usage
-----
    from dataset_stream import DatasetStreamer
    streamer = DatasetStreamer()
    for text, source_label in streamer:
        # text is a raw string, source_label is a human-readable tag
        ...
"""

import os
import random
import time
import threading
import queue
import requests
from datetime import datetime, timedelta, timezone

# ── optional heavy imports (lazy) ─────────────────────────────────────────────
_fineweb_iter = None
_fineweb_lock = threading.Lock()

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

REDDIT_SUBS   = ["science", "worldnews", "machinelearning", "technology", "philosophy",
                 "space", "economics", "askscience", "singularity", "compsci"]
HN_TOP_URL    = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL   = "https://hacker-news.firebaseio.com/v0/item/{}.json"
WIKI_URL      = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
NEWSAPI_URL   = "https://newsapi.org/v2/top-headlines"
ARXIV_SEARCH  = "https://export.arxiv.org/api/query"

# Weight table: (source_fn, relative_weight)
# Higher weight = called more often
# Prioritizing real-time news and hacker discussion for topical events
SOURCE_WEIGHTS = {
    "fineweb":    10,
    "wikipedia":  10,
    "reddit":     20,
    "hackernews": 20,
    "newsapi":    15,
    "arxiv":      10,
    "reasoning":  15,
}

# ── internal helpers ──────────────────────────────────────────────────────────

def _fetch_wikipedia() -> tuple[str, str]:
    r = requests.get(WIKI_URL, timeout=6)
    r.raise_for_status()
    data = r.json()
    title = data.get("title", "Wikipedia")
    text  = data.get("extract", "")
    return text, f"Wikipedia: {title}"


def _fetch_hackernews() -> tuple[str, str]:
    r = requests.get(HN_TOP_URL, timeout=6)
    r.raise_for_status()
    ids = r.json()[:30]
    story_id = random.choice(ids)
    r2 = requests.get(HN_ITEM_URL.format(story_id), timeout=6)
    r2.raise_for_status()
    item = r2.json()
    title = item.get("title", "")
    text  = item.get("text", "") or title  # discussion body or at least title
    # strip HTML tags crudely
    import re
    text = re.sub(r"<[^>]+>", " ", text).strip()
    return text, f"HackerNews: {title[:60]}"


def _fetch_reddit() -> tuple[str, str]:
    sub = random.choice(REDDIT_SUBS)
    sort = random.choice(["hot", "top", "new"])
    url = f"https://www.reddit.com/r/{sub}/{sort}.json?limit=25&t=day"
    headers = {"User-Agent": "DTSG-Distillation/1.0"}
    r = requests.get(url, timeout=8, headers=headers)
    r.raise_for_status()
    posts = r.json()["data"]["children"]
    if not posts:
        raise ValueError("Empty reddit response")
    post = random.choice(posts)["data"]
    selftext = post.get("selftext", "").strip()
    title    = post.get("title", "")
    text     = selftext if len(selftext) > 80 else title
    return text, f"Reddit r/{sub}: {title[:50]}"


def _fetch_newsapi() -> tuple[str, str]:
    if not NEWS_API_KEY:
        raise ValueError("No NEWS_API_KEY")
    params = {
        "apiKey":   NEWS_API_KEY,
        "language": "en",
        "pageSize": 20,
    }
    r = requests.get(NEWSAPI_URL, params=params, timeout=8)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    if not articles:
        raise ValueError("No news articles")
    art = random.choice(articles)
    title   = art.get("title", "")
    content = art.get("content", "") or art.get("description", "") or title
    return content, f"NewsAPI: {title[:60]}"


def _fetch_arxiv() -> tuple[str, str]:
    # Papers submitted in the last 48 hours
    start = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y%m%d%H%M")
    params = {
        "search_query": (
            "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML "
            "OR cat:cs.NE OR cat:quant-ph OR cat:physics.gen-ph"
        ),
        "sortBy":        "submittedDate",
        "sortOrder":     "descending",
        "max_results":   40,
    }
    r = requests.get(ARXIV_SEARCH, params=params, timeout=10)
    r.raise_for_status()
    import xml.etree.ElementTree as ET
    root  = ET.fromstring(r.text)
    ns    = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        raise ValueError("No arxiv entries")
    entry  = random.choice(entries)
    title  = entry.find("atom:title", ns).text.strip().replace("\n", " ")
    summary= entry.find("atom:summary", ns).text.strip().replace("\n", " ")
    return summary, f"Arxiv: {title[:60]}"


def _get_fineweb_chunk() -> tuple[str, str]:
    global _fineweb_iter
    with _fineweb_lock:
        if _fineweb_iter is None:
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                )
                _fineweb_iter = iter(ds)
                print("[DatasetStreamer] FineWeb-Edu stream initialized ✅")
            except Exception as e:
                print(f"[DatasetStreamer] FineWeb-Edu unavailable: {e}")
                raise
        row = next(_fineweb_iter)
    return row["text"][:800], "FineWeb-Edu (HuggingFace)"


_reasoning_iter = None
_reasoning_lock = threading.Lock()

def _get_reasoning_chunk() -> tuple[str, str]:
    global _reasoning_iter
    with _reasoning_lock:
        if _reasoning_iter is None:
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    "meta-math/MetaMathQA",
                    split="train",
                    streaming=True,
                )
                _reasoning_iter = iter(ds)
                print("[DatasetStreamer] Reasoning CoT stream initialized ✅")
            except Exception as e:
                print(f"[DatasetStreamer] Reasoning CoT unavailable: {e}")
                raise
        row = next(_reasoning_iter)
        q = row.get("instruction", "")
        a = row.get("response", "")
        text = f"Question: {q}\nThinking Process/Response: {a}"
    return text[:1000], "Reasoning-CoT (HuggingFace)"


# ── source dispatch table ─────────────────────────────────────────────────────

_SOURCES = {
    "fineweb":    _get_fineweb_chunk,
    "wikipedia":  _fetch_wikipedia,
    "reddit":     _fetch_reddit,
    "hackernews": _fetch_hackernews,
    "newsapi":    _fetch_newsapi,
    "arxiv":      _fetch_arxiv,
    "reasoning":  _get_reasoning_chunk,
}

_SOURCE_NAMES   = list(SOURCE_WEIGHTS.keys())
_SOURCE_WEIGHTS = [SOURCE_WEIGHTS[k] for k in _SOURCE_NAMES]


# ── public class ──────────────────────────────────────────────────────────────

class DatasetStreamer:
    """
    Infinite iterator over live multi-source text.

    Each call to next() returns (text: str, source_label: str).
    Failures on any single source are silently swallowed and retried
    with a different source so the training loop never stalls.

    Parameters
    ----------
    prefetch : int
        Number of samples to keep pre-fetched in a background queue.
    min_len  : int
        Minimum character length required for a sample to be used.
    """

    def __init__(self, prefetch: int = 8, min_len: int = 80):
        self.min_len  = min_len
        self._q: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=prefetch)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            if self._q.full():
                time.sleep(0.2)
                continue
            try:
                source_name = random.choices(_SOURCE_NAMES, weights=_SOURCE_WEIGHTS, k=1)[0]
                fn = _SOURCES[source_name]
                result = fn()
                if isinstance(result, tuple):
                    text, label = result
                else:
                    text, label = result, source_name
                if len(text.strip()) >= self.min_len:
                    self._q.put((text.strip(), label))
            except Exception:
                time.sleep(0.5)  # back-off on any error, then retry

    def __iter__(self):
        return self

    def __next__(self) -> tuple[str, str]:
        return self._q.get(timeout=30)

    def stop(self):
        self._stop.set()
