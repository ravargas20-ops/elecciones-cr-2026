import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    df["texto_chunk"] = df["texto_chunk"].fillna("").astype(str)
    return df[df["texto_chunk"].str.len() > 0].reset_index(drop=True)

def build_index(df):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["texto_chunk"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # IMPORTANTE: ahora devolvemos embeddings también
    return model, index, embeddings

def retrieve(df, model, index, query, k=6):
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = index.search(q_emb, k)

    return [df.iloc[i].to_dict() for i in idxs[0]]

def format_sources(rows):
    sources = []
    for r in rows:
        p = r["pagina_inicio"]
        if r["pagina_fin"] != p:
            p = f"{p}-{r['pagina_fin']}"
        sources.append(f"- {r['partido']} — {r['doc_id']} (p. {p})")
    return "\n".join(sources)
