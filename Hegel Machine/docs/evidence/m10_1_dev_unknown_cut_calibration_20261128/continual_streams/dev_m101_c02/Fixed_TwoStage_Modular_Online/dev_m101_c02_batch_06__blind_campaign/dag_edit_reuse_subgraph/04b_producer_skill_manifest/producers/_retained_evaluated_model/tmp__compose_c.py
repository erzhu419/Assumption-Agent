import itertools,pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_00.py'); a=m['fit_a'](s['panel_a']); t=m['inv_a'](s['train']['x'],a); p,b=m['_fit_b'](s['panel_b']); B=lambda x:m['_basis_b'](x,p)@b; C=s['panel_c']; u=C['u']; Y=s['train']['traj']; rs=[]; affine=lambda x:np.c_[np.ones(len(x)),x]; pairs=list(itertools.combinations_with_replacement(range(4),2)); Q=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cuts=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]; sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node'];
def report(tag,z):
 d=affine(z); vals=[np.mean(abs(d[j]@np.linalg.lstsq(d[i],Y[i],rcond=None)[0]-Y[j])) for i,j in cuts]; rs.append((max(vals),tag,vals,z))
def inv(f,x,u,v):
 initial=affine(x)@np.linalg.lstsq(affine(v),u,rcond=None)[0]; return np.array([least_squares(lambda w:f(w[None])[0]-xx,w0,max_nfev=65).x for w0,xx in zip(initial,x)])
trans=[]
for src in ('u','v1','v2'):
 for dst in ('u','v1','v2'):
  if src==dst:continue
  x,v=C[src],C[dst]
  for kind,feat in [('q',lambda x:np.c_[affine(x),Q(x)]),('s',lambda x:np.c_[affine(x),x-.4*np.sin(2.8*x)])]:
   w=np.linalg.lstsq(feat(x),v,rcond=None)[0]; f=lambda z,feat=feat,w=w:feat(z)@w; err=np.sqrt(np.mean((f(x)-v)**2)); print(src,dst,kind,round(err,5));
   if src=='u' and dst=='v2' and kind=='q': print(np.round(w,4))
   trans.append((src+dst+kind,f));
   if err<.015:trans.append(('inv'+src+dst+kind,lambda z,f=f,x=x,v=v:inv(f,z,x,v)))
for name,f in trans:
 for tag,z in [(name,f(t)),(name+' after B',f(B(t))),('B after '+name,B(f(t)))]:report(tag,z)
 for name2,g in trans:
  if name.startswith('u') and name2.startswith(('v1','v2')):report(name2+' after '+name+' after B',g(f(B(t))))
report('original',B(t))
for val,tag,vals,z in sorted(rs,key=lambda w:w[0])[:12]:
 out=sink({'__input__':{'train':{k:s['train'][k][:52] for k in ('y','traj')}},'node_00':{'z_tr':z[:52],'z_te':z[52:]}}); print(tag,vals,np.mean(abs(out['y_hat']-s['train']['y'][52:])))
