import json
rows = json.load(open('observations.json'))
ys = [r['target'] for r in rows]
print(json.dumps({'n_rows': len(rows), 'y_min': min(ys), 'y_max': max(ys), 'non_monotone_span': max(ys) - ys[-1]}))
