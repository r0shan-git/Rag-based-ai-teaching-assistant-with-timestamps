# import requests
# import os
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib

# def create_embedding(text_list):
#     r = requests.post(
#         "http://localhost:11434/api/embed",
#         json={
#             "model": "bge-m3",
#             "input": text_list
#         }
#     )

#     data = r.json()

#     if "embeddings" not in data:
#         raise ValueError(f"Ollama error: {data}")

#     return data["embeddings"]


# jsons = os.listdir("jsons")
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)

#     print(f"Creating Embeddings for {json_file}")

#     texts = [c["text"] for c in content["chunks"] if c.get("text")]
#     embeddings = create_embedding(texts)

#     for chunk, emb in zip(content["chunks"], embeddings):
#         chunk["chunk_id"] = chunk_id
#         chunk["embedding"] = emb
#         chunk_id += 1
#         my_dicts.append(chunk)

# df = pd.DataFrame.from_records(my_dicts)
# joblib.dump(df, "embeddings.joblib")



import requests
import os
import json
import pandas as pd
import joblib
import math
import time

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"


def is_valid_text(t):
    return isinstance(t, str) and t.strip()


def has_nan(vec):
    return any(math.isnan(v) for v in vec)


def embed_single(text):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "input": text.strip()},
            timeout=60
        )
        data = r.json()

        if "embeddings" not in data:
            return None

        emb = data["embeddings"][0]
        if has_nan(emb):
            return None

        return emb

    except Exception as e:
        print("❌ embed failed:", e)
        time.sleep(2)
        return None


jsons = os.listdir("jsons")
records = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"\n📄 Processing: {json_file}")

    for i, c in enumerate(content.get("chunks", []), start=1):
        text = c.get("text")
        if not is_valid_text(text):
            continue

        print(f"⏳ Chunk {i}", end="\r")

        emb = embed_single(text)
        if emb is None:
            continue

        c["chunk_id"] = chunk_id
        c["embedding"] = emb
        records.append(c)
        chunk_id += 1

        # 💾 autosave
        if chunk_id % 10 == 0:
            joblib.dump(
                pd.DataFrame.from_records(records),
                "embeddings.joblib"
            )
            print(f"\n💾 Saved till chunk {chunk_id}")

df = pd.DataFrame.from_records(records)

if len(df) > 0:
    joblib.dump(df, "embeddings.joblib")
    print(f"\n✅ embeddings.joblib saved with {len(df)} rows")
else:
    print("❌ No embeddings created")
