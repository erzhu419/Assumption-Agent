exec(open('/tmp/probe.py').read().split('u,v=')[0])
from scipy.optimize import least_squares, minimize_scalar
z=np.load('/tmp/coords.npy'); q=s['panel_c']['u']; t=s['telemetry_vec']; B=ls(aff(s['panel_b']['u']),s['panel_b']['v']); w=aff(z)@B
odd={'tanh':lambda x,k:np.tanh(k*x), 'sin':lambda x,k:np.sin(k*x), 'atan':lambda x,k:np.arctan(k*x), 'rat':lambda x,k:x/(1+k*x*x), 'erf':lambda x,k:__import__('scipy').special.erf(k*x), 'asinh':lambda x,k:np.arcsinh(k*x)}
even={'cos':lambda x,k:np.cos(k*x),'rat2':lambda x,k:x*x/(1+k*x*x),'log1p':lambda x,k:np.log1p(k*x*x),'cosh':lambda x,k:np.cosh(k*x),'exp':lambda x,k:np.exp(-k*x*x),'logcosh':lambda x,k:np.log(np.cosh(k*x)), 'sqrt':lambda x,k:np.sqrt(1+k*x*x)}
cs=[]
for j,out in enumerate(['v1','v2']):
 target=s['panel_c'][out]; results=[]
 for name,f in {**odd,**even}.items():
  for mode in ('only','lin','quad'):
   def design(u,k):
    h=f(u,k); return aff(h) if mode=='only' else np.c_[aff(u),h] if mode=='lin' else np.c_[aff(u),u*u,h]
   def score(k):
    h=design(q,k);return np.mean((h@ls(h,target)-target)**2)
   grid=np.geomspace(.04,5,60); i=np.argmin([score(k) for k in grid]); opt=minimize_scalar(score,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); k=opt.x; h=design(q,k); beta=ls(h,target); results.append((opt.fun**.5,name,mode,k,design(z,k)@beta,design(w,k)@beta))
 results.sort(key=lambda a:a[0]); print(out,[r[:4] for r in results[:10]]); cs.append(results[0]);
for idx, (name,b) in enumerate([('z',z),('w',w),('c1z',cs[0][4]),('c2z',cs[1][4]),('c1w',cs[0][5]),('c2w',cs[1][5]),('sumz',cs[0][4]+cs[1][4]),('sumw',cs[0][5]+cs[1][5])]):
 i,j=np.triu_indices(4); print(name)
 for label,h in [('lin',aff(b)),('tanh',aff(np.tanh(b))),('lin+tanh',np.c_[aff(b),np.tanh(b)]),('quad',np.c_[aff(b),b[:,i]*b[:,j]]),('quad+tanh',np.c_[aff(b),b[:,i]*b[:,j],np.tanh(b)])]:
  yp=h[52:72]@ls(h[:52],Y[:52]); e=np.mean(abs(yp-Y[52:]),0); print(label,round(e[0],5),round(e[1:].mean(),5))
