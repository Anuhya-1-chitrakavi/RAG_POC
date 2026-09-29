# ⚡ Quick Start (5 minutes)

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
⏳ First run downloads embeddings model (~400MB, happens once)

## Step 2: Add Your API Key

**Option A: Create `.env` file**
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

**Option B: Set environment variable**
```bash
# Linux/Mac
export ANTHROPIC_API_KEY='sk-ant-your-key-here'

# Windows PowerShell
$env:ANTHROPIC_API_KEY='sk-ant-your-key-here'
```

Get your key: https://console.anthropic.com/

## Step 3: Ensure PDF is in same directory
```bash
ls Chitrakavi_Lakshmi_Anuhya_-_Offer_Letter.pdf
```

## Step 4: Run!

**Interactive mode** (chat with your PDF):
```bash
python rag_pipeline.py
```

Then type questions:
```
🔍 Your query: What is the salary?
🔍 Your query: summarize
🔍 Your query: quit
```

**Quick examples**:
```bash
python example_usage.py
```

---

## 🎯 That's it!

Your RAG pipeline is ready. Try these questions:
- "What is the CTC mentioned?"
- "When is the join date?"
- "What benefits are included?"
- "Explain the probation terms"
- "summarize" (to get full summary)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| API key error | Add `ANTHROPIC_API_KEY` to `.env` or environment |
| PDF not found | Ensure PDF is in same directory |
| Slow first run | Normal - downloading embedding model |
| Bad answers | Wait for model to download, or check context |

Need help? See `README.md` for detailed guide!
