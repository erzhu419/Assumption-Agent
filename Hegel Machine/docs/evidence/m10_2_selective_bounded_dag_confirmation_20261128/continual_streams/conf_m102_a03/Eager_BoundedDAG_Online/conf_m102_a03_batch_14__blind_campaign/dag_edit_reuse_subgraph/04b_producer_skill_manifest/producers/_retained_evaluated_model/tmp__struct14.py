exec(open('/tmp/read14.py').read().split('for f in (1.25')[0])
from itertools import combinations
P=s['panel_b']; U=P['u']; Y=P['v'];
for name,fn in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('linear+cubic',lambda x:x+x**3),('sat',lambda x:x/(1+x*x))]:
 for sh in (0,1):
  def design(f):return aff(np.c_[U,fn(f*U)] if sh else fn(f*U))
  def err(f):
   D=design(f); return np.mean((Y-D@fit(D,Y))**2)
  r=minimize_scalar(err,bounds=(.3,4),method='bounded'); print('PANEL',name,sh,round(r.x,6),np.sqrt(r.fun))
for f in [1.25,2.15,2.8,2.9,3.0]:
 D=aff(np.sin(f*U));print('pure sin',f,np.sqrt(np.mean((Y-D@fit(D,Y))**2)))
T=s['train']['traj']; tar=np.c_[s['train']['y'],T[:,-1]]; times=s['eval_times'];
for h in range(2):
 key=('y','amp')[h];allres=[]
 for pn,xx in [('u',u),('affz',en['z'])]+[(f'sin{f}',aff(np.sin(f*u))@fit(aff(np.sin(f*U)),Y)) for f in (1.25,2.15,2.8,2.9,3.0)]+[('sinaff',aff(np.c_[u,np.sin(2.9*u)])@fit(aff(np.c_[U,np.sin(2.9*U)]),Y))]:
  a=aff(xx);rr=tar[:,h]-a@fit(a,tar[:,h]); fs=[]; names=[]
  def add(n,x):names.append(n);fs.append(x)
  for i,j in combinations(range(4),2):add(f'{i}*{j}',xx[:,i]*xx[:,j])
  for i,j,k in combinations(range(4),3):add(f'{i}*{j}*{k}',xx[:,i]*xx[:,j]*xx[:,k])
  for pair in [(0,2,1,3),(0,1,2,3),(0,3,1,2)]:
   i,j,k,l=pair
   for sign in [-1,1]:add(f'pair{pair}:{sign}',xx[:,i]*xx[:,j]+sign*xx[:,k]*xx[:,l])
  for i in range(4):
   q=xx[:,i];add(f'{i}sq',q*q);add(f'{i}cube',q**3)
   for f in (.22,1,1.25,2.15,3):
    add(f'tanh{i}:{f}',np.tanh(f*q));add(f'sin{i}:{f}',np.sin(f*q));add(f'cos{i}:{f}',np.cos(f*q));add(f'rat{i}:{f}',q/(1+f*q*q))
  for j,(n,x) in enumerate(zip(names,fs)):
   d=np.c_[a,x];c=fit(d[:52],tar[:52,h]);allres.append((np.mean(abs(tar[52:,h]-d[52:]@c)),pn,n,np.round(c,3)))
  d=a.copy();remain=list(range(len(fs)))
  for _ in range(7):
   ix=min(remain,key=lambda j:np.mean((tar[:52,h]-np.c_[d[:52],fs[j][:52]]@fit(np.c_[d[:52],fs[j][:52]],tar[:52,h]))**2));remain.remove(ix);d=np.c_[d,fs[ix]]
   c=fit(d[:52],tar[:52,h]);allres.append((np.mean(abs(tar[52:,h]-d[52:]@c)),pn,','.join(names[i] for i in range(len(fs)) if i not in remain),np.round(c,3)))
 print(key,sorted(allres,key=lambda a:a[0])[:12])
