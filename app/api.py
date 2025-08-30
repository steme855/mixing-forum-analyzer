
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mixing Forum Analyzer API")

class Query(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(q: Query):
    # TODO: hook into TF-IDF / similarity pipeline
    return {"query": q.text, "result": "not_implemented_yet"}
