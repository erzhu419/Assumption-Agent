import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/compare_maps.py').read())
from scipy.optimize import minimize_scalar
u=base['deA']; funcs={}; zz={}; fits2={}; results=[]
for name,P,key in [('B',b,'v'),('C1',c,'v1'),('C2',c,'v2')]:
 U,V=P['u'],P[key]; fits2[name]=[]
 for typ,f in [('rat',lambda t,k:t/(1+k*t*t)),('tanh',lambda t,k:np.tanh(k*t)),('sin',lambda t,k:np.sin(k*t)),('cos',lambda t,k:np.cos(k*t)-1)]:
  def fit(k):
   F=np.c_[np.ones(len(U)),U,f(U,k)]; C=np.linalg.lstsq(F,V,rcond=None)[0];return np.mean((F@C-V)**2),C
  ks=np.linspace(.1,5,45);i=np.argmin([fit(k)[0] for k in ks]);k=minimize_scalar(lambda k:fit(k)[0],bounds=(ks[max(0,i-1)],ks[min(i+1,len(ks)-1)]),method='bounded').x
  rm,C=fit(k);fits2[name].append((rm,typ,k,C,f))
 for rm,typ,k,C,f in sorted(fits2[name],key=lambda t:t[0])[:2]:
  print(name,typ,k,rm**.5,C.round(4),flush=True)
  l,sv,r=np.linalg.svd(C[1:5]); O=l@r
  funcs[name+typ]=lambda t,k=k,C=C,f=f: C[0]+t@C[1:5]+f(t,k)@C[5:]
  funcs[name+typ+'L']=lambda t,C=C: C[0]+t@C[1:5]
  funcs[name+typ+'N']=lambda t,k=k,C=C,f=f: C[0]+f(t,k)@C[5:]
  funcs[name+typ+'R']=lambda t,O=O:t@O
  funcs[name+typ+'T']=lambda t,O=O:t@O.T
funcs['I']=lambda t:t
for n1,f1 in funcs.items():
 for n2,f2 in funcs.items():
  q=f2(f1(u)); sd=np.mean(q.std(0));
  for sc in [.12,.25,.45,.7,1.0]:
   z=q*(sc/max(sd,1.e-9)); e=score(z); results.append((max(e),e.tolist(),n1,n2,sc))
for n1,f1 in funcs.items():
 for n2,f2 in funcs.items():
  q,sq=f1(u),f2(u); q=(q-q.mean(0))/q.std(0); sq=(sq-sq.mean(0))/sq.std(0)
  for wt in [-1,-.5,.25,.5,1]:
   for sc in [.15,.3,.6]:
    e=score(sc*(q+wt*sq)); results.append((max(e),e.tolist(),n1,n2,wt,sc))
print('composition top:',*sorted(results)[:16],sep='\n');print('public top:',*sorted(results,key=lambda t:max(t[1][:2]))[:10],sep='\n')
