import argparse
import json
import os
import re
import sys
import time
import concurrent.futures
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import numpy as np
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss-cpu not installed correctly. Try: pip install faiss-cpu") from e

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import torch
    from threading import Thread
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

def stream_print(text: str, delay: float = 0.0) -> None:
    words = text.split()
    for i, w in enumerate(words):
        sys.stdout.write(w + (" " if i < len(words) - 1 else ""))
        sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def is_internal_link(base_netloc: str, url: str) -> bool:
    try:
        return urlparse(url).netloc == base_netloc
    except Exception:
        return False

def clean_url(u: str) -> str:
    u, _frag = urldefrag(u)
    return u.rstrip("/")

def looks_like_binary(url: str) -> bool:
    return bool(re.search(r"\.(pdf|png|jpg|jpeg|gif|svg|zip|tar|gz|mp4|mp3|webm)$", url, re.I))

@dataclass
class Page:
    url: str
    title: str
    text: str

def extract_text_from_html(url: str, html: str) -> Tuple[str, str]:
    title = ""
    text = ""

    if trafilatura is not None:
        try:
            downloaded = trafilatura.extract(html, include_comments=False, include_tables=False)
            if downloaded:
                text = downloaded
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")

    t = soup.find("title")
    if t and t.get_text(strip=True):
        title = t.get_text(strip=True)

    if not text:
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    text = normalize_whitespace(text)

    if len(text) < 200:
        text = ""

    return title, clean_junk_text(text)

def clean_junk_text(text: str) -> str:
    words = text.split()
    clean_words = [w for w in words if len(w) < 50]
    
    text = " ".join(clean_words)
    
    if len(text) > 50000:
        text = text[:50000]
        
    return text

def fetch_page_requests(session, url, timeout):
    try:
        resp = session.get(url, timeout=timeout)
        ct = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or "text/html" not in ct:
            return None
        return resp.text
    except Exception:
        return None

def fetch_page_browser(url):
    if sync_playwright is None:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="local-rag-bot/1.0")
            page = context.new_page()
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000) 
                content = page.content()
            except Exception:
                content = None
            finally:
                browser.close()
            return content
    except Exception:
        return None

def crawl_site(
    base_url: str,
    max_pages: int = 250,
    timeout: int = 10,
    rate_limit_s: float = 0.05,
    user_agent: str = "local-rag-bot/1.0",
    use_browser: bool = False,
    workers: int = None,
) -> List[Page]:
    print(f"Starting {'browser-based' if use_browser else 'fast'} parallel crawl of {base_url}...")
    base_url = clean_url(base_url)
    base = urlparse(base_url)
    base_netloc = base.netloc

    pages: List[Page] = []
    seen = {base_url}
    
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    def process_url(url):
        if looks_like_binary(url):
            return None
        
        if use_browser:
            html = fetch_page_browser(url)
        else:
            html = fetch_page_requests(session, url, timeout)

        if not html:
            return None
            
        title, text = extract_text_from_html(url, html)
        
        links = []
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            nxt = clean_url(urljoin(url, href))
            if not nxt.startswith(("http://", "https://")):
                continue
            if is_internal_link(base_netloc, nxt):
                links.append(nxt)
        
        return (url, title, text, links)

    if workers is None:
        workers = 4 if use_browser else 10

    print(f"Starting {'browser-based' if use_browser else 'fast'} parallel crawl of {base_url} with {workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_url, base_url): base_url}
        
        while futures and (max_pages <= 0 or len(pages) < max_pages):
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for fut in done:
                original_url = futures.pop(fut)
                try:
                    res = fut.result()
                    if res:
                        r_url, r_title, r_text, r_links = res
                        if r_text:
                            pages.append(Page(url=r_url, title=r_title or r_url, text=r_text))
                            if len(pages) % 5 == 0:
                                print(f"  Indexed {len(pages)} pages...", end="\r")
                        
                        for link in r_links:
                            if link not in seen:
                                seen.add(link)
                                if max_pages <= 0 or (len(pages) + len(futures) < max_pages * 2):
                                    futures[executor.submit(process_url, link)] = link
                except Exception:
                    pass

    print(f"\nCrawl complete. Found {len(pages)} pages.")
    return pages

def chunk_text(text: str, chunk_chars: int = 1500, overlap_chars: int = 250) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{2,}", s.lower())

def build_vocab(pages: List[Page], max_vocab: int = 50000) -> set:
    counts = Counter()
    for p in pages:
        counts.update(tokenize_words(p.text))
    most_common = [w for w, _ in counts.most_common(max_vocab)]
    return set(most_common)

def generate_answer(tokenizer, model, query: str, context: str):
    prompt = (
        "<|system|>\n"
        "You are a smart research assistant. Use the Context below to answer the Question. "
        "If asking for jobs/roles, list ALL distinct titles found in bullet points. "
        "If the answer is not in the context, say 'I cannot find that information'.\n"
        "</s>\n"
        "<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "</s>\n"
        "<|assistant|>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
    
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=512, 
        do_sample=False,
        repetition_penalty=1.1 
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    sys.stdout.write("\nAI Response: ")
    sys.stdout.flush()
    
    generated_text = ""
    for new_text in streamer:
        sys.stdout.write(new_text)
        sys.stdout.flush()
        generated_text += new_text
    
    sys.stdout.write("\n")
    return generated_text

def correct_query_typos(query: str, vocab: set, min_token_len: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", query)
    corrected_parts = []
    vocab_list = None

    for tok in tokens:
        if not tok.isalpha():
            corrected_parts.append(tok)
            continue

        low = tok.lower()
        if len(low) < min_token_len or low in vocab:
            corrected_parts.append(tok)
            continue

        if vocab_list is None:
            vocab_list = list(vocab)

        match = process.extractOne(low, vocab_list, scorer=fuzz.WRatio)
        if match and match[1] >= 90:
            corrected_parts.append(match[0])
        else:
            corrected_parts.append(tok)

    return "".join(corrected_parts)

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True, 
    )
    return np.array(emb, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

JOB_HINTS = re.compile(r"\b(job|jobs|career|careers|open roles|openings|positions|roles|vacanc)\b", re.I)

def detect_intent(query: str) -> str:
    q = query.strip().lower()
    if JOB_HINTS.search(q):
        return "jobs"
    return "general"

def extract_job_titles_from_text(text: str) -> List[str]:
    candidates = set()

    parts = re.split(r"[•\n\r\t]|(?<=\.)\s+|(?<=;)\s+", text)
    for p in parts:
        p = normalize_whitespace(p)
        if not p:
            continue

        if 8 <= len(p) <= 80 and re.search(r"\b(Engineer|Developer|Analyst|Trader|Researcher|Designer|Manager|Intern)\b", p):
            p = re.sub(r"\s{2,}", " ", p)
            p = p.strip(" -–—:;,.")
            if 4 <= len(p) <= 70:
                candidates.add(p)

    return sorted(candidates)[:25]

def format_sources(results: List[Dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"- {r['url']}")
    return "\n".join(lines)

def answer_from_chunks(query: str, intent: str, top_chunks: List[Dict]) -> str:
    if not top_chunks:
        return "No relevant content found on the crawled pages."

    candidate_titles = []
    if intent == "jobs":
        merged = " ".join([c["text"] for c in top_chunks[:10]])
        candidate_titles = extract_job_titles_from_text(merged)

    snippet_text = []
    for c in top_chunks[:3]:
        snippet_text.append(f"... {c['text'][:400].strip()} ...")
    
    combined_snippets = "\n\n".join(snippet_text)

    if intent == "jobs":
        if candidate_titles:
            out = "Possible roles found:\n"
            out += "\n".join([f"- {t}" for t in candidate_titles])
            out += "\n\nRelevant Context:\n" + combined_snippets
            return out
        else:
            return f"I couldn't list specific titles, but here is the relevant content:\n\n{combined_snippets}"

    return combined_snippets

def save_index(out_dir: str, index: faiss.Index, meta: List[Dict], embeddings: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def load_index(out_dir: str) -> Tuple[faiss.Index, List[Dict], np.ndarray]:
    index_path = os.path.join(out_dir, "index.faiss")
    meta_path = os.path.join(out_dir, "meta.jsonl")
    emb_path = os.path.join(out_dir, "embeddings.npy")

    if not (os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(emb_path)):
        raise FileNotFoundError("Index files not found. Run with --rebuild.")

    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta, embeddings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base site URL, e.g. https://www.janestreet.com/")
    ap.add_argument("--out", default="./site_index", help="Index output folder")
    ap.add_argument("--rebuild", action="store_true", help="Re-crawl and rebuild index")
    ap.add_argument("--max_pages", type=int, default=0, help="0 for unlimited")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rate_limit", type=float, default=0.25)
    ap.add_argument("--stream_delay", type=float, default=0.0, help="e.g. 0.01 to print word-by-word slower")
    ap.add_argument("--use_browser", action="store_true", help="Use Playwright browser to handle JS pages")
    ap.add_argument("--workers", type=int, default=None, help="Number of parallel crawler threads")
    args = ap.parse_args()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    llm_tokenizer = None
    llm_model = None
    if TRANSFORMERS_AVAILABLE:
        print("Loading Generative Decoder (TinyLlama-1.1B-Chat)...")
        try:
            llm_checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            llm_tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)
            llm_model = AutoModelForCausalLM.from_pretrained(llm_checkpoint, torch_dtype=torch.float32)
            print("Decoder loaded.")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
    else:
        print("Transformers library not found. Falling back to snippet mode.")

    if args.rebuild:
        pages = crawl_site(
            args.url, 
            max_pages=args.max_pages, 
            rate_limit_s=args.rate_limit,
            use_browser=args.use_browser,
            workers=args.workers
        )
        if not pages:
            print("No pages extracted. The site may block crawling or be mostly JS-rendered.")
            return

        meta: List[Dict] = []
        all_chunks: List[str] = []

        for p in pages:
            chunks = chunk_text(p.text)
            for i, ch in enumerate(chunks):
                meta.append({"url": p.url, "title": p.title, "chunk_id": i, "text": ch})
                all_chunks.append(ch)

        embeddings = embed_texts(model, all_chunks)
        index = build_faiss_index(embeddings)
        save_index(args.out, index, meta, embeddings)

        print(f"Indexed {len(pages)} pages, {len(all_chunks)} chunks -> {args.out}")

    index, meta, _emb = load_index(args.out)

    pages_for_vocab = [Page(url=m["url"], title=m["title"], text=m["text"]) for m in meta]
    vocab = build_vocab(pages_for_vocab, max_vocab=50000)

    print("Ready. Type a question. Commands: /exit, /help, /topk N")
    current_topk = args.topk

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q.startswith("/topk"):
            parts = q.split()
            if len(parts) == 2 and parts[1].isdigit():
                current_topk = int(parts[1])
                print(f"topk set to {current_topk}")
            else:
                print("Usage: /topk 8")
            continue
        if q == "/help":
            print("Ask about the website. Examples: 'jobs', 'contact', 'about', 'benefits'.")
            print("Commands: /exit, /topk N")
            continue

        intent = detect_intent(q)
        corrected = correct_query_typos(q, vocab)
        query_variants = [q] if corrected == q else [q, corrected]

        best = {}
        
        if intent == "jobs":
            search_k = current_topk * 10
        else:
            search_k = current_topk * 4
        
        for qv in query_variants:
            q_emb = model.encode([qv], normalize_embeddings=True).astype(np.float32)
            D, I = index.search(q_emb, k=search_k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                best[idx] = max(best.get(idx, -1e9), float(score))

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)

        final_chunks = []
        for idx, score in ranked:
            m = meta[idx]
            url = m["url"].lower()
            
            if any(x in url for x in ["closed", "expired", "archive", "ended", "past-events", "puzzle", "solution"]):
                continue
            
            if "closed" in m["title"].lower() or "puzzle" in m["title"].lower():
                continue

            final_score = score
            if intent == "jobs":
                if any(x in url for x in ["career", "job", "position", "role", "opening", "vacancy", "work-at", "join"]):
                    final_score += 0.2
                if any(x in url for x in ["2018", "2019", "2020", "2021", "2022", "privacy", "term"]):
                    final_score -= 0.1
                    
            final_chunks.append({
                "score": final_score, 
                "url": m["url"],
                "title": m["title"], 
                "text": m["text"]
            })

        final_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = final_chunks[:current_topk]

        if llm_model and llm_tokenizer:
            context_pieces = []
            for c in top_chunks[:8]:
                source_info = f"[Source: {c['url']}]"
                context_pieces.append(f"{source_info}\n{c['text']}")
            context_str = "\n\n".join(context_pieces)

            generate_answer(llm_tokenizer, llm_model, q, context_str)
        else:
            response = answer_from_chunks(q, intent, top_chunks)
            stream_print(response, delay=args.stream_delay)

        srcs = list(dict.fromkeys([c["url"] for c in top_chunks[:5]]))
        if srcs:
            print("\nSources:")
            for u in srcs:
                print(f"- {u}")


if __name__ == "__main__":
    main()
