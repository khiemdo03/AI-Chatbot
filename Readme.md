# Universal Local RAG Assistant

A privacy-first AI research tool that crawls complex websites, filters irrelevant data, and synthesizes answers using a local LLM. Designed to run offline on consumer hardware with a hybrid "Code + AI" architecture.

---

## Case Study: Engineering a Hybrid RAG System

### Situation
My initial goal was ambitious: I wanted to build a Universal Chatbot that could visit *any* company website (like Google, Microsoft, or Jane Street) and act as an expert recruiter. I wanted it to be "Pure AI"—meaning I could ask it *anything* ("List all jobs", "Who is the CEO?") and it would just figure it out using a Generative Model, running entirely offline on my laptop.

### Task
I had to build a complete pipeline: a Crawler to fetch data, a Vector Database to store it, and a Local LLM to answer questions. The real constraint was hardware. I was forced to use open-source "Small Language Models" (1.1B parameters) on a standard CPU, rather than paying for API access to massive models like GPT-4.

### Action
I spent weeks iterating on the architecture. I built a robust crawler using `Playwright` that successfully scraped 100% of the data. However, when I connected the AI, I hit a wall. 
When I asked the small model to "List all 50 Engineering jobs", it failed catastrophically. It either crashed the memory or "hallucinated" (made up jobs). It simply wasn't smart enough or big enough to handle that volume of data. 
I tried changing models (T5 to TinyLlama), I tried "Prompt Engineering" (giving it stricter rules), but the hardware limitation was absolute.

### Result
The result was **not** the "Magic AI" I originally envisioned. I realized a pure AI approach was impossible on this hardware.
I had to compromise. I pivoted to a **"Hybrid Architecture"**.
1.  **For Lists (Data Heavy):** I wrote deterministic Python code to extract and display raw job titles—effectively bypassing the AI to guarantee accuracy.
2.  **For Facts (Reasoning Heavy):** I kept the AI only for specific questions like "What are the benefits?", where it actually shone.

The final tool works perfectly, but it is a tool built on *compromise* rather than pure AI magic.

### Key Takeaway
"I learned that in the real world, hardware dictates architecture. You can't code your way out of a hardware bottleneck; sometimes you have to abandon 'Pure AI' and fall back to solid software engineering to deliver a working product."

---

## Quick Start
### Prerequisites
- Python 3.8+
- Playwright

### Installation
```bash
pip install -r requirements.txt
playwright install
```

### Usage
Run the crawler on any target website:
```bash
python rag_cli.py --url https://www.janestreet.com/ --use_browser --rebuild
```
Ask questions:
- `jobs` (Uses Hybrid Listing Mode)
- `who is the CEO?` (Uses Generative AI Mode)
