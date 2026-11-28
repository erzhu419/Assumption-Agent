import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
D=lambda x:np.c_[np.ones(len(x)),x]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
adj=lambda x:x*np.roll(x,-1,axis=1)
cs=lambda x:np.cos(2.4*x)
A=s['panel_a'];u,v=A['u'],A['v'];g=np.max(abs(v-u),1)<1
coef=np.array([ls(D(u[g,j:j+1]),v[g,j]) for j in range(4)])
x=s['train']['x'];z=(x-coef[:,0])/coef[:,1]
B=s['panel_b'];C=s['panel_c']; b=ls(D(B['u']),B['v']); p=ls(D(adj(C['u'])),C['v1']);q=ls(D(cs(C['u'])),C['v2']);
ops={'B':lambda x,b=b:D(x)@b,'P':lambda x,p=p:D(adj(x))@p,'Q':lambda x,q=q:D(cs(x))@q}
print('B',b,'P',p,'Q',q,sep='\n')
t=s['train']['traj']; y=s['train']['y']; Y=np.c_[y,t]; n=52
nodes={'Z':z,'X':x}
for depth in range(1,4):
 for key in it.permutations('BPQ',depth):
  nm=''.join(key);nodes[nm]=ops[key[-1]](nodes[nm[:-1] or 'Z'])

def metric(a):
 h=ls(a[:n],Y[:n]); tr=np.mean(abs(Y[:n]-a[:n]@h),axis=0);cv=np.mean(abs(Y[n:]-a[n:]@h),axis=0)
 return np.round([cv[0],cv[1:].mean(),tr[0],tr[1:].mean()],5)
R=[]
for k,l in it.combinations_with_replacement(nodes,2):
 a,v=nodes[k],nodes[l];dot=np.sum(a*v,1)
 for kind,feat in [('add',D(np.c_[a,v])),('dot',D(np.c_[a,v,dot])),('pair',D(np.c_[a,v,(a[:,:,None]*v[:,None,:]).reshape(len(a),16)])),('extra',D(np.c_[a,v,dot,a[:,0]*v[:,1],v[:,0]*a[:,1]]))]:
  R.append((k,l,kind,metric(feat)))
for j in range(4):
 print('ORDER',j);print(*sorted(R,key=lambda a:a[-1][j])[:13],sep='\n')
print('time singular raw',np.linalg.svd(t,compute_uv=False))
for cols in [('Z','B','P','Q'),('Z','B','P','Q','BP','BQ'),('Z','B','P','Q','BP','BQ','PQ','QP')]:
 print(cols,metric(D(np.column_stack([nodes[key] for key in cols]))))
