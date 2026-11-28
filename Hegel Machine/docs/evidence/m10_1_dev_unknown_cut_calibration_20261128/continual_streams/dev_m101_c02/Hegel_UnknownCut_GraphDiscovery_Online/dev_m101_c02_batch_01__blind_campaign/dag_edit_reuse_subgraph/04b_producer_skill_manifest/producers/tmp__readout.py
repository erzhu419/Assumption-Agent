exec(open('/tmp/refine.py').read().split("a=s['panel_a']")[0])
a=s['panel_a'];m=np.max(abs(a['v']-a['u']),axis=1)<1;u=a['u'][m];v=a['v'][m];g=((u-u.mean(0))*(v-v.mean(0))).sum(0)/((u-u.mean(0))**2).sum(0);o=v.mean(0)-g*u.mean(0);x=(s['train']['x']-o)/g; y=s['train']['y']; T=s['train']['traj'];Y=np.column_stack([y,T]);
def mapped(p,v,k,w):
 u=s[p]['u']; a,c,*_=fit(u,s[p][v],k,w);return mat(x,k,a)@c
b=mapped('panel_b','v','s',[2.8]); c=mapped('panel_c','v1','h',[2.2]); d=mapped('panel_c','v2','c',[1.27]);
from itertools import combinations_with_replacement
ones=np.ones((len(x),1)); H=np.column_stack([ones,b,c]); q=[b[:,i]*b[:,j] for i,j in combinations_with_replacement(range(4),2)];
def blocked(A,Y):
 es=[]
 for ids in np.array_split(np.arange(len(A)),6):
  m=np.ones(len(A),bool);m[ids]=False; coef=np.linalg.lstsq(A[m],Y[m],rcond=None)[0];es.append(np.abs(A[ids]@coef-Y[ids]).mean(0))
 e=np.array(es);return np.round(e.mean(0),5),np.round(e.reshape(len(e),-1).mean(1),5),round(e.max(),5)
print('y coefficients [intercept,b,c,upper triangle bb]',np.linalg.lstsq(np.column_stack([H,*q]),y,rcond=None)[0]);
for label,A in [('B',np.column_stack([ones,b])),('BC',H),('B+BB',np.column_stack([ones,b,*q])),('BC+BB',np.column_stack([H,*q])),('BCD+BB',np.column_stack([H,d,*q]))]: print('y',label,blocked(A,y))
for label,A in [('BC',H),('BC+B²',np.column_stack([H,b*b])),('BC+allBB',np.column_stack([H,*q])),('BC+C²',np.column_stack([H,c*c])),('BC+cosX',np.column_stack([H,np.cos(1.27*x)])),('BC+tanhB',np.column_stack([H,np.tanh(b)])),('BC+tanhC',np.column_stack([H,np.tanh(c)])),('BC+BD',np.column_stack([H,b*d])),('BCD',np.column_stack([H,d]))]: print('traj',label,blocked(A,T))
print('trajectory right singular vectors',np.linalg.svd(T,full_matrices=False)[2][:2]);print('eval times',s['eval_times'])
