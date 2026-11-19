memory={}
def fit(observations):
    global memory
    memory={r['features']['u']:r['target'] for r in observations}
def predict(row):
    return memory.get(row['features']['u'],0.0)
