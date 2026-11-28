from probe import *
from scipy.linalg import orthogonal_procrustes
p=s['panel_c']; u=p['u']; L=poly(u,1)
ops={}
for key in ('v1','v2'):
 Y=p[key]; results=[]
 for name,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('atan',np.arctan),('rational',lambda q:q/(1+q*q)),('square',np.square)]:
  for mode in ['ridge','coordinate','pre','direct']:
   preds=[]; errs=[]
   for ix in range(4):
    yy=Y[:,ix]; w=np.linalg.lstsq(L,yy,rcond=None)[0]; starts=[w,2*w,*[np.r_[b,np.eye(4)[ix]] for b in [0,.6,1.4]],*[np.r_[0,e] for e in np.eye(4)]]
    best=None
    for st in starts:
     def design(x,q):
      l=poly(x,1)
      if mode=='ridge': return np.c_[l,f(l@q)]
      if mode=='direct': return np.c_[l[:,0],f(l@q)]
      if mode=='coordinate': return np.c_[l,f(x*q[1]+q[0])]
      return np.c_[l[:,0],f(x*q[1]+q[0])]
     q0=st[:2] if mode in ('coordinate','pre') else st
     def fun(q):
      D=design(u,q);return D@np.linalg.lstsq(D,yy,rcond=1e-10)[0]-yy
     r=least_squares(fun,q0,max_nfev=90,bounds=(-5,5)); loss=np.mean(r.fun**2)
     if best is None or loss<best[0]: best=loss,r.x
    q=best[1]; D=design(u,q); w=np.linalg.lstsq(D,yy,rcond=1e-10)[0]; preds.append((q,w)); errs.append(np.sqrt(best[0]))
   results.append((np.mean(errs),name,mode,preds))
 print(key,[(round(e,6),f,m) for e,f,m,_ in sorted(results)[:10]])
 for e,name,mode,params in sorted(results)[:3]:
  print('panel coefficients',key,name,mode,[(q.round(3),w.round(3)) for q,w in params])
  f={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'atan':np.arctan,'rational':lambda q:q/(1+q*q),'square':np.square}[name]
  def predict(x,mode=mode,params=params,f=f):
   l=poly(x,1); cols=[]
   for q,w in params:
    D={'ridge':lambda:np.c_[l,f(l@q)],'direct':lambda:np.c_[l[:,0],f(l@q)],'coordinate':lambda:np.c_[l,f(x*q[1]+q[0])],'pre':lambda:np.c_[l[:,0],f(x*q[1]+q[0])]}[mode]()
    cols.append(D@w)
   return np.column_stack(cols)
  ops[key+name+mode]=predict

def screen(ops):
 zs={'raw':x,'a':za,'bi':zb,'abi':zba,'bia':zab,'b':x@B[1:]+B[0],'ab':za@B[1:]+B[0]}
 Y=np.c_[y,T]; results=[]
 for zname,z in zs.items():
  d=poly(x,2); Q,_=np.linalg.qr(d)
  for pname,op in ops.items():
   v=op(z)
   for name,F in [('c',v),('c*q',np.column_stack([v[:,i]*z[:,j] for i in range(4) for j in range(4)])),('c2',poly(v,2)[:,1:]),('both',np.c_[v,*[v[:,i]*z[:,j] for i in range(4) for j in range(4)]])]:
    G=F-Q@(Q.T@F); G=G/np.maximum(np.linalg.norm(G,axis=0),1e-5); D=np.c_[d,G]; es=[]
    for jj in np.array_split(np.arange(len(x)),4):
     ii=np.setdiff1d(np.arange(len(x)),jj); reg=np.r_[np.full(d.shape[1],1.e-9),np.full(G.shape[1],0.1)]; w=np.linalg.solve(D[ii].T@D[ii]+np.diag(reg),D[ii].T@Y[ii]); es.append(np.mean(abs(D[jj]@w-Y[jj]),axis=0)[[0,-1]])
    es=np.array(es);results.append((es.mean(0),es.max(0),zname,pname,name))
 for k in range(2): print('best panel features',k,sorted(results,key=lambda r:r[0][k])[:8])
screen(ops)
