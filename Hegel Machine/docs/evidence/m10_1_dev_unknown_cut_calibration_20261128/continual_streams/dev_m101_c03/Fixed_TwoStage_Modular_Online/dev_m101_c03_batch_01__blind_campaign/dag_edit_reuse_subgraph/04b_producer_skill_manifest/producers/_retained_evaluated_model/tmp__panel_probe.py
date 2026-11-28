import pickle,itertools as it,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u=s['panel_c']['u']; n,d=u.shape
names=['1']+[f'u{i}' for i in range(d)]; cols=[np.ones(n),*u.T]; groups={'aff':list(range(5))}
def add(g,name,x):
 groups.setdefault(g,[]).append(len(cols)); names.append(name); cols.append(x)
for deg in (2,3):
 for ix in it.combinations_with_replacement(range(d),deg): add('poly'+str(deg),str(ix),u[:,ix].prod(1))
for i in range(d):
 a=u[:,i]
 for w in (.5,1,1.5,2,2.5,3,3.14,4,5,6):
  for f in (np.sin,np.cos,np.tanh): add(f.__name__+str(w),f'{f.__name__}({w}u{i})',f(w*a))
 for w in (.5,1,2): add('rational',f'u{i}/(1+{w}u{i}²)',a/(1+w*a*a))
 add('abs',f'abs(u{i})',abs(a))
 for j in range(i+1,d):
  b=u[:,j]
  for k in (.5,1,1.5,2):
   for sign in (-1,1):
    t=a+sign*k*b
    for f in (np.sin,np.cos,np.tanh):
     for w in (1,2,3,4): add('pair',f'{f.__name__}({w}*(u{i}{sign*k:+}u{j}))',f(w*t))
  for w in (.5,1,2):
   add('rational',f'u{i}/(1+{w}u{j}²)',a/(1+w*b*b)); add('rational',f'u{j}/(1+{w}u{i}²)',b/(1+w*a*a)); add('rational',f'u{i}*u{j}/(1+{w}*(u{i}²+u{j}²))',a*b/(1+w*(a*a+b*b)))
  for w in (1,2,3,4):
   for f in (np.sin,np.cos,np.tanh): add('product',f'{f.__name__}({w}*u{i}u{j})',f(w*a*b))
X=np.column_stack(cols)
for key in ('v1','v2'):
 y=s['panel_c'][key]; print(key)
 stats=[]
 for group,ix in groups.items():
  if len(ix)>17: continue
  m=X[:,groups['aff']+(ix if group!='aff' else [])]; q=np.linalg.qr(m)[0]; r=y-q@(q.T@y); loo=r/(1-(q*q).sum(1))[:,None]; stats.append((round(np.sqrt(np.mean(loo**2)),5),group,round(np.sqrt(np.mean(r*r)),5)))
 print('LOO RMS, group, fit RMS',sorted(stats)[:12]);
 for j in range(4):
  selected=groups['aff'].copy(); target=y[:,j]; print('col',j)
  for k in range(5):
   q=np.linalg.qr(X[:,selected])[0]; r=target-q@(q.T@target); z=X-q@(q.T@X); zn=np.linalg.norm(z,axis=0); p=np.abs(z.T@r)/np.maximum(zn,1e-9); top=p.argsort()[-4:][::-1]; print(round(np.std(r),5),[(names[i],round(p[i],4)) for i in top]); selected.append(top[0])
