exec(open('/tmp/panels.py').read().split("for p in ['panel_b'")[0])
def score(A,Y):
 q,r=np.linalg.qr(A); e=Y-q@(q.T@Y); h=1-(q*q).sum(1); return np.mean(abs(e/h[:,None]),axis=0)
for p in ['panel_b','panel_c']:
 u=s[p]['u']; d=library(u); names=list(d); X=np.array(list(d.values())).T
 for k in range(4):
  x=u[:,k]
  for w in np.arange(0.5,5.01,.25):
   for name,z in [('sin',np.sin(w*x)),('cos',np.cos(w*x)),('tanh',np.tanh(w*x))]:
    if f'{name}{w:g}x{k}' not in d: d[f'{name}{w:g}x{k}']=z
  d[f'x{k}^3']=x**3
 names=list(d);X=np.array(list(d.values())).T
 for v in sorted(set(s[p])-{'u'}):
  Y=s[p][v]; print(p,v,'affine',score(X[:,:5],Y),'quad',score(X[:,:15],Y))
  for j in range(4):
   y=Y[:,j:j+1]; beam=[(1.,())]
   for order in range(1,4):
    fits={}
    for _,ids in beam:
     for i in range(5,len(names)):
      new=tuple(sorted((*ids,i)))
      if i in ids or new in fits: continue
      a=X[:,[*range(5),*new]]
      if np.linalg.cond(a)>250: continue
      fits[new]=score(a,y)[0]
    beam=sorted((e,k) for k,e in fits.items())[:7]
    print(j,order,[(round(e,5),[names[i] for i in k]) for e,k in beam[:3]])
a=s['panel_a'];good=np.max(abs(a['v']-a['u']),axis=1)<1;u=a['u'][good];v=a['v'][good];g=((u-u.mean(0))*(v-v.mean(0))).sum(0)/((u-u.mean(0))**2).sum(0);o=v.mean(0)-g*u.mean(0);x=(s['train']['x']-o)/g;Y=np.column_stack([s['train']['y'],s['train']['traj']]); L=np.array(list(library(x).values())).T
for i in [5,9,15]: print('direct',i,score(L[:,:i],Y))
