import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement
spec=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];t=s['train']['traj'];o,g=m._calibration(s['panel_a']);w=(x-o)/g;u,v=(s['panel_b'][k] for k in ('u','v'))
def operator(f,r,linear=True):
 def d(a):return np.c_[np.ones(len(a)),a,f(a*r)] if linear else np.c_[np.ones(len(a)),f(a*r)]
 a=d(u);c=np.linalg.lstsq(a,v,rcond=None)[0];e=v-a@c;q=np.linalg.qr(a)[0];return (e/(1-(q*q).sum(1))[:,None]),c,d(w)@c
ops={}
for f,starts in [(np.sin,[1.6]),(np.tanh,[2.5]),(lambda a:a/(1+a*a),[.7,2]),(lambda a:a*a*a/(1+a*a),[.7,2]),(lambda a:np.arctan(a),[2.5])]:
 for linear in (True,False):
  def loss(r):return np.mean(operator(f,r,linear)[0]**2)
  fit=minimize_scalar(loss,bounds=(.12,5),method='bounded');e,c,z0=operator(f,fit.x,linear);name=f.__name__+str(linear)+str(len(ops));ops[name]=(z0,fit.fun)
  print(name,round(fit.x,4),round(fit.fun**.5,6),c.round(4).T if linear and f in (np.sin,np.tanh) else '')
z=m._operator(s['panel_b'])(w); all_i=np.arange(len(x));folds=[all_i[52:]]+[np.random.default_rng(k).permutation(len(x))[-20:] for k in range(4)]+[np.argsort(sign*x[:,k])[-16:] for k in range(4) for sign in (1,-1)]
print('y range',y.min(),y.max(),'z range',z.min(0),z.max(0))
def ridge(a,b,p):return m._ridge(a,b,p)
def test(z,op):
 a=m._affine(z);q=m._quadratic(z); n=z.shape[1];p=np.column_stack([z[:,i]*z[:,j]*z[:,k] for i,j,k in combinations_with_replacement(range(n),3)]);a3=np.c_[q,p]; out=[]
 for mode in ('lin','quad','cub','tanhlinear','tanhquad','tanhplusquad','quadrank1','rational'):
  for reg in ([.001,.01,.04] if mode in ('quad','cub') else [.003]):
   errs=[]
   for j in folds:
    i=np.setdiff1d(all_i,j)
    if mode in ('lin','quad','cub'):
     d={'lin':a,'quad':q,'cub':a3}[mode];yp=d[j]@ridge(d[i],y[i],reg)
    else:
     b=ridge(a[i],y[i],reg); c=ridge(q[i],y[i],reg);pred=None
     if mode.startswith('tanh'):
      d=q if mode=='tanhquad' else a; lin0=c if mode=='tanhquad' else b
      if mode=='tanhplusquad':
       def pred(h,j):return np.tanh(a[j]@h[:5])*h[5]+q[j]@h[6:]
       init=np.r_[b,1.,np.zeros(q.shape[1])]
      else:
       def pred(h,j):return np.tanh(d[j]@h[:-2])*h[-2]+h[-1]
       init=np.r_[lin0,1.,0.]
     elif mode=='quadrank1':
      def pred(h,j):return a[j]@h[:5]+h[-1]*(z[j]@h[5:9])**2
      init=np.r_[b,np.ones(4)*.2,.3]
     else:
      def pred(h,j):return (q[j]@h[:-1])/(1+h[-1]**2*(z[j]@b[1:])**2)
      init=np.r_[c,.3]
     def res(h):return np.r_[pred(h,i)-y[i],np.sqrt(reg)*h]
     h=least_squares(res,init,max_nfev=100).x;yp=pred(h,j)
    errs.append(np.mean(abs(yp-y[j])))
   e=np.array(errs);out.append((op,mode,reg,*e[:1],e[1:5].mean(),e[5:].max(),e[10],e[5]))
 return out
out=[]
for name,(zz,loss) in ops.items():
 if loss**.5<.02:out+=test(zz,name)
for j in (4,5):
 print('best by',j)
 for row in sorted(out,key=lambda r:r[j])[:14]:print(row[:3],np.round(row[3:],5))
