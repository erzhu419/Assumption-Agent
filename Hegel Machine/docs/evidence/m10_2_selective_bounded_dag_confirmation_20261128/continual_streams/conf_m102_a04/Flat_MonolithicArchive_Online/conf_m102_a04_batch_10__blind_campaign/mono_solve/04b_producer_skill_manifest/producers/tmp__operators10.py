exec(open('/tmp/explore.py').read().split('ops=[]')[0])
Y=s['panel_c']['v1']; t=s['eval_times']; b=s['panel_b']; tel=s['telemetry_vec']; print('times',t,'telemetry',tel)
funcs={'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'rational':lambda x,k:x/(1+k*x*x),'radical':lambda x,k:x/np.sqrt(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'erf':__import__('scipy.special',fromlist=['erf']).erf,'softsign':lambda x,k:x/(1+k*abs(x))}
funcs['erf']=lambda x,k:__import__('scipy.special',fromlist=['erf']).erf(k*x)
extra={'lin':lambda x:x,'cubic':lambda x:x**3,'quintic':lambda x:x**5,'square':lambda x:x*x,'pairprod':lambda x:x*np.roll(x,-1,axis=1),'evenrat':lambda x:x*x/(1+.25*x*x)}
from itertools import combinations
fits=[]
for name,f in funcs.items():
 for nm in [(),*( (a,) for a in extra),*combinations(extra,2),('lin','cubic','quintic')]:
  for tie in (False,True):
   if tie and len(nm)!=1: continue
   def ds(x,p):
    v=f(x,p[0]); es=[extra[a](x) for a in nm]; return one(v+p[1]*es[0]) if tie else one(np.c_[v,*es]);
   def loss(p):
    d=ds(u,p);return (d@ls(d,Y)-Y).ravel()
   fit=min((least_squares(loss,[k,0] if tie else [k],bounds=([.05,-20],[8,20]) if tie else ([.05],[8]),max_nfev=100) for k in (.5,1.5,2.5)),key=lambda r:np.std(r.fun)); fits.append((np.std(fit.fun),name,nm,tie,fit.x))
for row in sorted(fits,key=lambda x:x[0])[:28]:print(row)
for name,f in funcs.items():
 for name2,h in list(funcs.items())+[('cos',lambda x,k:np.cos(k*x))]:
  if name==name2:continue
  def ds(x,p):return one(f(x,p[0])+p[2]*h(x,p[1]))
  def loss(p):
   d=ds(u,p);return (d@ls(d,Y)-Y).ravel()
  fit=min((least_squares(loss,p,bounds=([.05,.05,-30],[8,8,30]),max_nfev=100) for p in ([1.8,2.4,.2],[1.3,1.3,-.2],[2.4,1.4,.2])),key=lambda r:np.std(r.fun));
  if np.std(fit.fun)<.006:print('combo',name,name2,np.std(fit.fun),fit.x)
