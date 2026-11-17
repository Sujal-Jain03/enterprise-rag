from app.ingestions.chunking.splitter import chunk_text

PARA = (
    "Kubernetes Horizontal Pod Autoscaler (HPA) automatically scales the number of pods "
    "in a deployment based on observed CPU utilization or other select metrics. "
    "It does not apply to objects that cannot be scaled, such as DaemonSets. "
    "The HPA controller periodically queries the metrics API to gather resource usage data "
    "and compares it against the targets defined in the HPA object. "
)
text = "\n\n".join([PARA.strip()] * 8)  # ~3500 chars -> 3 chunks expected

chunks = chunk_text(text)
print("num chunks:", len(chunks))
for i, c in enumerate(chunks):
    print(f"chunk {i}: {len(c)} chars | head: {c[:60]!r} | tail: {c[-60:]!r}")
print("overlap present:", chunks[1][:100] in chunks[0] if len(chunks) > 1 else "n/a")
