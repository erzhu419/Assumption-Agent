import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X,Y=map(np.asarray,(C['true_channel_loads'],C['bridge_raw'])); K=np.max(abs(Y-np.median(Y,0)),1)<5; idx=np.flatnonzero(K); X,Y=X[K],Y[K]; pairs=[2,3,0,1]
for part in ('train','test','chamber_verification_trials'):
 if D.get(part) is not None:
  for name in ('past_channel_raw','hold_channel_raw'):
   a=np.asarray(D[part][name]).reshape(-1,4);print(part,name,len(a),np.isfinite(a).all(),a.min(0).round(3),a.max(0).round(3))
main={'sinlin':lambda u,s:[u,np.sin(s*u)],'sinsat':lambda u,s:[np.tanh(s*u),np.sin(2.5*u),u],'tanhlin':lambda u,s:[u,np.tanh(s*u)],'rational':lambda u,s:[u,u/(1+(s*u)**2)],'sinpoly':lambda u,s:[u,u**3,np.sin(s*u)],'sinlinear2':lambda u,s:[u,np.sin(s*u),np.sin(2*s*u)],'sinplus':lambda u,s:[u,np.sin(s*u),1-np.cos(s*u)]}
cr={'tanh2':lambda v,t:[np.tanh(t*v)**2],'rat2':lambda v,t:[v*v/(1+(t*v)**2)],'cos':lambda v,t:[1-np.cos(t*v)],'exp':lambda v,t:[-np.expm1(-(t*v)**2)],'log':lambda v,t:[np.log1p((t*v)**2)],'root':lambda v,t:[1-1/np.sqrt(1+(t*v)**2)],'mixed':lambda v,t:[np.tanh(t*v)**2,1-np.cos(t*v)]}
for j,k in enumerate(pairs):
 u,v,y=X[:,j],X[:,k],Y[:,j]; print('CHANNEL',j); rankings=[]
 for a,f in main.items():
  for b,g in cr.items():
   for odd in (0,1):
    def calc(p,m):
     s,t=p; A=np.column_stack([np.ones(len(u)),*f(u,s),v,*([np.tanh(t*v)] if odd else []),*g(v,t)]); q=np.linalg.lstsq(A[m],y[m],rcond=None)[0]; return A@q-y,q
    m=idx>=0
    best=min((least_squares(lambda p:calc(p,m)[0],seed,bounds=([.12,.12],[4.5,4.5]),max_nfev=90) for seed in ([2.5,1.0],[1.,1.5])),key=lambda r:r.cost)
    e,q=calc(best.x,m); rankings.append((np.std(e),a,b,odd,best.x,q,calc))
 for r,a,b,odd,p,q,_ in sorted(rankings,key=lambda z:z[0])[:8]:
  f,g=main[a],cr[b]; ee=np.zeros(len(u));
  for mask in [idx%5==i for i in range(5)]:
   def check(p):
    s,t=p; A=np.column_stack([np.ones(len(u)),*f(u,s),v,*([np.tanh(t*v)] if odd else []),*g(v,t)]); return A@np.linalg.lstsq(A[~mask],y[~mask],rcond=None)[0]-y
   p2=least_squares(lambda p:check(p)[~mask],[2.5 if a.startswith('sin') else 1.,1.],bounds=([.12,.12],[4.5,4.5]),max_nfev=150).x; ee[mask]=check(p2)[mask]
  print(a,b,odd,'fit/oof/max',np.round([r,np.std(ee),max(abs(ee))],7),'p',p.round(5),'q',q.round(5))
