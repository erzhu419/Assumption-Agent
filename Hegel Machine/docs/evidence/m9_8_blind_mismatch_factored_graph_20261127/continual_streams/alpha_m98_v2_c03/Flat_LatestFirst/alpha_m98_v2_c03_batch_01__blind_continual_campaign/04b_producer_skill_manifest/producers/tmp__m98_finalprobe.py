exec(open('/tmp/m98_gate_cv.py').read().split('rank=sorted')[0])
from itertools import combinations
for g in (0.,.25,.5,.75,1.):
 names=['cubic','rational winner','rational alternate','ratio norm 1.0 2','ratio norm 0.0 3']; w=np.array([g,(1-g)*.25,(1-g)*.25,(1-g)*.35,(1-g)*.15]); es=[sum(a*err(name,v) for a,name in zip(w,names)) for _,v in F]; print('ENSEMBLE',g,stats(es))
for name in ('cubic','rational winner','rational alternate','ratio norm 1.0 2','ratio norm 0.0 3'):
 B,o,p=lib[name]; c=fit(B,y-o,np.ones(n,bool),p); print('COEF',name,c)
# Test whether a simple load-history contrast, as opposed to a dense quadratic, explains the drift gain.
load=np.c_[hold,past]; dv={f'p{j}':load[:,j] for j in range(load.shape[1])}; dv.update({'mean':past.mean(1),'std':past.std(1),'range':np.ptp(past,axis=1),'roughness':np.sqrt(np.mean(np.diff(past,axis=1)**2,axis=1))});
for i,j in combinations(range(load.shape[1]),2):
 for s in (-1.,-.5,.5,1.): dv[f'p{i}{s:+}p{j}']=load[:,i]+s*load[:,j]
for name,q in dv.items():
 for z,zz in [('sq',q*q),('abs',abs(q)),('cube',q**3),('centered_sq',(q-np.mean(q))**2)]: add('load '+name+' '+z,r*zz,np.c_[r],b[2]+r)
loadout=[]
for name in lib:
 if name.startswith('load '):
  e=[err(name,v) for _,v in F]; z=stats(e); loadout.append((max(z[:2]),tuple(z),name))
print('LOAD',sorted(loadout)[:6]); print('FALLBACK',stats([err('linear',v) for _,v in F]));
Xtest=xp[len(tr['lab_id']):]; u=Xtest[:,0]+.5*Xtest[:,1]; v=Xtest[:,1]; z=Xtest[:,3]; rt=Xtest@b[3:]; Btest=[np.c_[np.ones(len(rt)),Xtest,u**3,v*v*z],np.c_[np.ones(len(rt)),Xtest,(v+Xtest[:,2])**3/(1+(v-z)**2)],np.c_[np.ones(len(rt)),Xtest,(Xtest[:,0]+2*v)**3/(1+(v-z)**2)],(rt*np.maximum(abs(Xtest)-1,0).__pow__(2).sum(1))[:,None],(rt*(abs(Xtest)**3).sum(1))[:,None]]
for name,Bt in zip(names,Btest):
 B,o,p=lib[name]; c=fit(B,y-o,np.ones(n,bool),p); a=Bt@c+(b[2]+rt if name.startswith('ratio') else 0.); print('DEPLOY',name,np.round(np.quantile(a,[0,.05,.5,.95,1]),4))
