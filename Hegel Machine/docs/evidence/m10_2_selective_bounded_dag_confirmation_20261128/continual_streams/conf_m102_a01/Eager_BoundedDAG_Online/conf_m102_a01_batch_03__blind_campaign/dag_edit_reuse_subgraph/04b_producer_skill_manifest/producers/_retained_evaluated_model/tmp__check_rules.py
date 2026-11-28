import numpy as np
Z=np.load('/tmp/cal_sink.npz'); A,V=Z['A'],Z['V']; z=np.load('/tmp/sink_eval.npz'); P,S,Y,J=[z[q] for q in 'PSYJ']; E=np.c_[z['I'],P,S,Y]; B=E[(J>0)&(J<21)]; T=E[J>=21]

def pred(a,w=0,b=0,k=0,cl=100,g=0,f=0):
 p,s=a[:,1:3].T; d=p-s
 return s+b+k*s+w*np.clip(d,-cl,cl)+g*np.abs(d)+f*np.maximum(s-.65,0)
def score(a,**kw):return abs(pred(a,**kw)-a[:,3]).mean()
def show(name,**kw):print(name,kw, 'cal/public/random/tails',np.round([score(a,**kw) for a in [A,V,B,T[:18],T[18:]]],6))
for w in [0,.1,.2,.35,.65]:
 for b in [0,.005,.01,.015,-.005]:show('blend',w=w,b=b)
for cl in [.03,.06,.12]:
 for w in [.1,.25,.5]:show('clipped',w=w,cl=cl)
for k in [.01,.025,.05,-.025]:
 for w in [0,.15]:show('scale',w=w,k=k)
for g in [-.2,-.1,.1,.2,.35]:show('asymmetric',w=.1,g=g)
for f in [.05,.1,.2,.3]:show('upper-hinge',w=.1,f=f)
for label,a in [('inner OOF',A),('official',V),('other OOF',B)]:
 p,s,y=a[:,1:4].T; d=p-s; e=y-s
 print(label,'mean/median residual',np.mean(e),np.median(e),'corr',np.round(np.corrcoef(np.c_[e,s,d,abs(d),a[:,4:]].T)[0],3))
