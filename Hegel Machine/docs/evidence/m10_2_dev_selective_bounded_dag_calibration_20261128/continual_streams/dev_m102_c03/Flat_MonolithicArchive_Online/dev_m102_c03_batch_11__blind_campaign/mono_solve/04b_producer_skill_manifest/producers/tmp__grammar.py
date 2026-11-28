exec(open('/tmp/identify.py').read().split("p=s['panel_a']")[0]); from itertools import combinations
func.update(soft=lambda x:x/(1+abs(x)), square=lambda x:x*x,cube=lambda x:x**3)
p=s['panel_a'];u=p['u'];v=p['v'];a=aff(u);good=np.linalg.norm(v-a@ls(a,v),axis=1)<3;u=u[good];v=v[good];a=aff(u); x=u; w=np.roll(x,2,axis=1)
for name,fn in func.items():
 for base in [[],['xw'],['xw','x3'],['xw','x2'],['xw','x2','w2']]:
  add={'xw':x*w,'x3':x**3,'x2':x*x,'w2':w*w};
  def res(q,ret=False):
   err=[]; bs=[]
   for j in range(4):
    f=np.c_[np.ones(len(x)),x[:,j],w[:,j],*(add[k][:,j] for k in base),fn(q*w[:,j])]; b=ls(f,v[:,j]);bs.append(b);err.append(f@b-v[:,j]);
   return np.array(bs).T if ret else np.std(err)
  r=minimize_scalar(res,bounds=(.12,4),method='bounded');
  if min(r.fun,res(2.15))<.012: print('A',name,base,r.x,r.fun,'at 2.15',res(2.15),'coef',res(r.x,True).round(3).tolist())
p=s['panel_c'];u=p['u'];
for key in ('v1','v2'):
 y=p[key]; opts=[]
 for i,j in combinations(func,2):
  for mode in [0,1,2]:
   def design(r):return np.c_[np.ones(len(u)),*( [u] if mode else []),*( [u**3] if mode==2 else []),func[i](r[0]*u),func[j](r[1]*u)]
   def res(r):
    a=design(r);return (a@ls(a,y)-y).ravel()
   for start in ([1.,1.],[2.15,2.15]):
    r=least_squares(res,start,bounds=(.1,5),max_nfev=80,gtol=1e-10); opts.append((np.std(r.fun),i,j,mode,r.x.tolist()))
 print(key,sorted(opts)[:20]);
