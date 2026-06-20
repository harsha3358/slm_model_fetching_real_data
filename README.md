# Technology Intelligence Monitor

A small intelligence dashboard that collects AI, technology, and logistics updates, removes weak or duplicate items, summarizes the useful ones, and ranks them by importance.

## Why it matters

Leaders do not need more links; they need fewer, more relevant signals. This project explores an automated way to turn research feeds and news into a prioritized briefing.

## What it does

- Fetches papers from arXiv and articles from Google News RSS
- Filters content using business and technical keywords
- Summarizes articles with TextRank
- Detects similar items with sentence embeddings
- Stores and displays ranked results in SQLite

## Technology

Python, Flask, Beautiful Soup, Sentence Transformers, scikit-learn, Sumy, and SQLite.

## Run

```bash
pip install -r requirements.txt
python intelligence_engine.py
```

Open the local Flask URL and select **Fetch Intelligence**.
