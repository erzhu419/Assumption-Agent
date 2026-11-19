import json
memory={}
def fit(observations):
    rows=observations
    global memory
    memory={r['features']['u']: r['target'] for r in rows}
def predict(row):
    return memory.get(row['features']['u'], 0.0)
