import sqlite3
import datetime
import requests
from flask import Flask, jsonify, render_template_string
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
DB = "intelligence.db"
SIM_THRESHOLD = 0.85

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TECH_KEYWORDS = [
    "transformer","llm","quantization","inference","distributed",
    "optimization","benchmark","architecture","deployment",
    "open-source","cuda","training","rag","agent",
    "supply chain","logistics","fleet","autonomous"
]

NEGATIVE_KEYWORDS = ["celebrity","opinion","lifestyle","interview"]

# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS intelligence(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            category TEXT,
            importance INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- FETCH ----------------

def fetch_arxiv():
    url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=5"
    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "xml")
    return [{
        "title": e.title.text,
        "content": e.summary.text,
        "category": "AI Research"
    } for e in soup.find_all("entry")]

def fetch_google(query, category):
    url = f"https://news.google.com/rss/search?q={query}"
    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "xml")
    return [{
        "title": item.title.text,
        "content": item.description.text,
        "category": category
    } for item in soup.find_all("item")[:5]]

# ---------------- FILTER ----------------

def technical_filter(article):
    text = (article["title"] + article["content"]).lower()
    score = 0
    for kw in TECH_KEYWORDS:
        if kw in text:
            score += 2
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            score -= 3
    return score > 2

# ---------------- SUMMARY ----------------

def summarize(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)
    return " ".join(str(s) for s in summary)

# ---------------- DUPLICATE ----------------

def is_duplicate(new_text):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT summary FROM intelligence")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return False

    stored = [r[0] for r in rows]
    new_emb = model.encode([new_text])
    stored_emb = model.encode(stored)
    sim = cosine_similarity(new_emb, stored_emb)
    return sim.max() > SIM_THRESHOLD

# ---------------- IMPORTANCE ----------------

def importance_score(article):
    text = (article["title"] + article["content"]).lower()
    tech_count = sum(1 for kw in TECH_KEYWORDS if kw in text)
    bonus = 20 if article["category"] == "Logistics" else 0
    return min(100, tech_count * 10 + bonus)

# ---------------- STORE ----------------

def store(article, summary, importance):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO intelligence(title, summary, category, importance, timestamp)
        VALUES(?,?,?,?,?)
    """, (
        article["title"],
        summary,
        article["category"],
        importance,
        datetime.datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

# ---------------- SCAN ----------------

@app.route("/scan")
def scan():
    try:
        articles = []
        articles += fetch_arxiv()
        articles += fetch_google("Indian AI startup deep tech", "Indian Tech")
        articles += fetch_google("Logistics AI automation fleet", "Logistics")

        added = 0

        for article in articles:
            if not technical_filter(article):
                continue
            summary = summarize(article["content"])
            if is_duplicate(summary):
                continue
            score = importance_score(article)
            store(article, summary, score)
            added += 1

        return jsonify({"status": "success", "added": added})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ---------------- API ----------------

@app.route("/api/intelligence")
def api_data():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT title, summary, category, importance FROM intelligence ORDER BY importance DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

# ---------------- FRONTEND ----------------

@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>Harsha Intelligence</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { background:#0f0c29; color:#00fff7; font-family:monospace; padding:20px;}
button { padding:10px 20px; background:#00fff7; border:none; cursor:pointer;}
.card { background:#111; margin:10px 0; padding:15px; border-radius:8px;}
</style>
</head>
<body>

<h1>⚡ Intelligence Dashboard</h1>
<button onclick="runScan()">Fetch Intelligence</button>
<p id="status"></p>
<div id="feed"></div>

<script>

async function runScan(){
    document.getElementById("status").innerText="Scanning...";
    const res = await fetch('/scan');
    const data = await res.json();
    if(data.status=="success"){
        document.getElementById("status").innerText="Added "+data.added+" updates.";
        loadData();
    } else {
        document.getElementById("status").innerText="Error: "+data.message;
    }
}

async function loadData(){
    const res = await fetch('/api/intelligence');
    const data = await res.json();
    const container = document.getElementById("feed");
    container.innerHTML="";
    data.forEach(item=>{
        container.innerHTML += `
        <div class="card">
            <h3>${item[0]}</h3>
            <small>${item[2]} | Score ${item[3]}</small>
            <p>${item[1]}</p>
        </div>`;
    });
}

loadData();

</script>
</body>
</html>
""")

if __name__ == "__main__":
    app.run(debug=True)