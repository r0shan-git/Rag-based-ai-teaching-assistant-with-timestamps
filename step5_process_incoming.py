# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# import joblib 
# import requests


# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# def inference(prompt,model):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         # "model": "deepseek-r1",
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream":False

#     })

#     response=r.json()
#     print(response)
#     return response



# df = joblib.load('embeddings.joblib')


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 5
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx] 
# # print(new_df[["title", "number", "text"]])

# prompt=f'''I am teaching web development using sigma web development course.Here are video subtitle chunks containing video title, video number,start time in seconds,end time in seconds, the text at that time :

# {new_df[["title", "number","start","end", "text"]].to_json(orient="records")}
# ---------------------------------------------

# "{incoming_query}"
# user asked this question related to the video chunks,you have to answer where and how much content is taught in which video (IN WHICH VIDEO AND AT WHAT TIMESTAMP) and guide the user to go to that particular video .If user asks unrelated question ,tell him that you can only answer question related to the course'''

# with open("prompt.txt","w") as f:
#     f.write(prompt)

# response=inference(prompt)["response"]

# print(response)


# with open("response.txt","w") as f:
#     f.write(response)

# # for index, item in new_df.iterrows():
# #     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    # print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

prompt = f"""
You are a friendly instructor for the **Sigma Web Development Course**.

You are given course video information that includes:
• Video title  
• Video number  
• Start time (in seconds)  
• End time (in seconds)  
• What is explained during that time  

📚 Course content:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

------------------------------------------------

❓ **User Question**:
"{incoming_query}"

🎯 **Answer Guidelines (FOLLOW STRICTLY)**:

1️⃣ **Understand the question clearly**
   • Identify what topic the user is asking about  

2️⃣ **Find the correct video content**
   • Match the question with the most relevant video(s)  

3️⃣ **Explain in a SIMPLE & HUMAN way**
   • Beginner-friendly language  
   • No technical jargon  

4️⃣ **MANDATORY: Mention clearly**
   • **📌 Video Title**
   • **🔢 Video Number**
   • **⏱ Timestamp (Start – End)**  

5️⃣ **Guide the user clearly**
   • Tell them **which video to open**
   • Tell them **exactly from which time to watch**

6️⃣ **FORMAT the final answer like this**
   • Use **bold text** for headings and important words  
   • Use **numbered points** for steps  
   • Use **bullet points (•)** for details  
   • Use **👉 arrows** to guide actions  

7️⃣ **STRICT RULES**
   • ❌ Do NOT mention subtitles, chunks, JSON, or datasets  
   • ❌ Do NOT guess or add extra information  
   • ❌ Do NOT answer unrelated questions  

8️⃣ **If the question is NOT related to the course**
   • Politely say:  
     **“I can only help with questions related to the Sigma Web Development course.”**

9️⃣ **Keep the answer**
   • Short  
   • Clear  
   • Easy to scan  
   • Student-friendly  
"""


# with open("prompt.txt", "w") as f:
#     f.write(prompt)
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


response = inference(prompt)["response"]
print(response)

# with open("response.txt", "w") as f:
#     f.write(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])