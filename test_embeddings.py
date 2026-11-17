import logfire
logfire.configure(service_name="embedding-smoke-test", send_to_logfire=False)

import app.services.retrieval.embeddings as emb

samples = [
    "Kubernetes Horizontal Pod Autoscaler automatically scales pods based on CPU utilization.",
    "The Metrics Server collects resource usage metrics from cluster nodes and pods.",
    "A Pattern Language describes design patterns for architecture and urban planning.",
]
vecs = emb.embed_texts(samples)
print("model_type:", emb._model_type)
print("dim:", emb.get_embedding_dim())
print("num vectors:", len(vecs), "| vector len:", len(vecs[0]))
print("first 5 values of vec 0:", [round(v, 4) for v in vecs[0][:5]])
q = emb.embed_query("How does HPA scale pods?")
print("query vector len:", len(q))
