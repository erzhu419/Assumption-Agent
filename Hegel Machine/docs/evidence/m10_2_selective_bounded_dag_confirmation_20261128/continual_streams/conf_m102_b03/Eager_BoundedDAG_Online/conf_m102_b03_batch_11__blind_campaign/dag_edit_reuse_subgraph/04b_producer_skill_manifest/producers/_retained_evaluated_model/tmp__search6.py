exec(open('/tmp/panel_search.py').read().split('C={')[0])
print('traj sv',np.linalg.svd(t,compute_uv=False),'y quantiles',np.quantile(y,np.linspace(0,1,11)))
A={'I':lambda w:w,'a':lambda w:(w-a)/b,'A':lambda w:w*b+a}; G={k:f for k,f in ops.items() if k not in ('I','x')}; W={'z':z,'x':x,'aa':A['a'](z)}
for k,f in G.items():
 for name,w in list(W.items())[:3]:
  for j,g in A.items(): W[name+k+j]=g(f(w))
C={k:funcs[0](w) for k,w in W.items()};D={k:funcs[1](w) for k,w in W.items()};
def score(F,target):
 coef=ls(F[:52],target[:52]); return np.mean(abs(F[52:]@coef-target[52:])), np.mean(abs(F[:52]@coef-target[:52]))
r=[];q=[]
for i,c in C.items():
 for j,d in D.items():
  cross=(c[:,:,None]*d[:,None,:]).reshape(len(c),-1);base=ones(np.c_[c,d]); dot=(c*d).sum(1)
  for nm,F in (('cross',np.c_[base,cross]),('dot',np.c_[base,dot]),('plain',base),('sparsequad',np.c_[base,c*d,c*c,d*d])):
   if nm in ('cross','sparsequad'):r.append((*score(F,t),i,j,nm))
   if nm in ('dot','plain'):q.append((*score(F,y),i,j,nm))
print('trajectory',sorted(r)[:25]);print('traj train',sorted(r,key=lambda e:e[1])[:12]);print('scalar',sorted(q)[:25]);
for nm,w in {'z':z,'x':x,'Bz':z@B+h}.items():
 c,d=funcs[0](w),funcs[1](w); dot=(c*d).sum(1);d1=np.c_[c,d]; dp=ones(np.c_[d1,dot]); rr=y-dp@ls(dp,y); print(nm,'res corrs',np.corrcoef(np.c_[rr,w**3, np.tanh(w*2.15), dot**2, dot**3].T)[0]);
