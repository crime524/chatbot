import json, torch, numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModel
import easyocr, requests, cv2
from bs4 import BeautifulSoup

# ---- Load IPC DB ----
with open("Part_of_ipc_database.json") as f:
    ipc_data = json.load(f)

# ---- Load model (LegalBERT or MiniLM if faster) ----
MODEL = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def encode(texts, batch_size=4):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        embs.append(outputs.cpu().numpy())
    return np.vstack(embs)

ipc_texts = [f"Section {s['section_number']} {s['title']} {s.get('explanation_en','')}" for s in ipc_data]
ipc_embeddings = encode(ipc_texts)

def find_best(query, top_k=3):
    q_emb = encode([query])[0]
    sims = ipc_embeddings @ q_emb / (np.linalg.norm(ipc_embeddings, axis=1) * np.linalg.norm(q_emb))
    top_ids = sims.argsort()[-top_k:][::-1]
    return [{ "section_number": ipc_data[i]["section_number"],
              "title": ipc_data[i]["title"],
              "similarity": float(sims[i]) } for i in top_ids]

# ---- OCR + URL helpers ----
reader = easyocr.Reader(["en"])

def extract_from_url(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    for s in soup(["script","style","noscript"]):
        s.extract()
    return soup.get_text(" ", strip=True)

# ---- FastAPI ----
app = FastAPI()

@app.post("/predict")
async def predict(text: str = Form(None), url: str = Form(None), file: UploadFile = None):
    extracted = ""

    if text:
        extracted = text
    elif url:
        extracted = extract_from_url(url)
    elif file:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = reader.readtext(img, detail=0)
        extracted = " ".join(result)
    else:
        return JSONResponse({"error": "Provide text, url, or file"}, status_code=400)

    results = find_best(extracted, top_k=3)
    return {"input_length": len(extracted), "results": results}
