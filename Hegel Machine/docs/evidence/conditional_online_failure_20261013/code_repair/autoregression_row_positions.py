# Host repair of row positions; diagnostic model family unchanged.
import json, numpy as np
from sklearn.linear_model import LinearRegression
rows=json.load(open('observations.json'))
# Test whether a low-order autoregressive response model with current controls
# predicts across whole paid trajectories. Previous responses are used only
# for this offline diagnosis, never as model inputs in the final predictor.
for p in range(1,7):
 X=[]; y=[]
 for i,r in enumerate(rows):
  h=r.get('history',[]); recent=[rows[j]['target'] for j in range(max(i-p,i-len(h)),i)]
  if len(recent)<p: recent=[0.]*(p-len(recent))+recent
  X.append([*recent,*[r['features'][k] for k in ('x','z','c')]])
  y.append(float(r['target']))
 m=LinearRegression().fit(X,y); e=np.asarray(y)-m.predict(X)
 print('AR',p,'mse',float(np.mean(e*e)),'maxabs',float(np.max(np.abs(e))),'coef',np.round(m.coef_,6).tolist(),'intercept',float(m.intercept_))
# Summarize row-level response relation of each reset input to ensuing zero controls.
for start,end in [(416,436),(436,456),(456,476),(476,496),(496,508),(508,520),(520,532),(532,544),(736,748),(748,760),(760,772),(880,892),(904,916),(952,964)]:
 print(start,[(tuple(round(rows[j]['features'][k],3) for k in ('x','z','c')),round(rows[j]['target'],6)) for j in range(start,end)])