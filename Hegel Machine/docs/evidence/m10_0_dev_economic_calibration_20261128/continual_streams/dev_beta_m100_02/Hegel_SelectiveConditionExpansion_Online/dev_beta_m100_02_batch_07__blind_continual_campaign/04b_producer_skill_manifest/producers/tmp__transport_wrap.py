from transport_pulses import *

def wall(key, pair):
    k=s['kappa_'+key]; v=s['vel_'+key]; h=s['L'+key]/(len(k)-1)
    A=axis(key,tuple(named(r=int(t=='R')) for t in pair)).tolil()
    for i,j,p,t in [(0,1,len(k)-1,pair[0]),(len(k)-1,len(k)-2,0,pair[1])]:
        if t in ('R','N'): continue
        sg=1 if i==0 else -1
        near=(k[i]+k[j])/(2*h*h)+max(-sg*v[i],0)/h
        far=(k[i]+k[p])/(2*h*h)+max(sg*v[i],0)/h
        if t=='C': near=(k[i]+k[j])/(2*h*h); far=(k[i]+k[p])/(2*h*h)
        if t=='K': near=k[i]/h**2; far=k[i]/h**2
        if t=='H': near=k[i]/h**2; far*=.5
        if t=='V': near=k[i]/h**2
        if t=='B': near=k[i]/h**2; far=0
        A[i,j]=-near; A[i,p]=-far; A[i,i]=near+far+(s['robin_'+key]/h if t=='B' else 0)
    return A.tocsc()

C={key:{p:wall(key,p) for p in itertools.product('RNWCKHV',repeat=2)} for key in 'xy'}
results=[]
for x,ax in C['x'].items():
 for y,ay in C['y'].items():
    e,f=errs(predict(ax,ay)); results.append((x+y,e,f))
for omit in (0,1,2,None):
 idx=[i for i in range(3) if i!=omit]
 print('omit',omit,[(g,np.round(e,8),np.round(f,8)) for g,e,f in sorted(results,key=lambda t:np.mean(t[1][idx]**2))[:4]])
for g,e,f in results:
 if g in [tuple(t) for t in ('WWNN','NNWW','WWWW','WWRR','RRWW')]: print('symmetric topology',g,e,f)
print('endpoint data',[(key,s['kappa_'+key][[0,-1]],s['vel_'+key][[0,-1]]) for key in 'xy'])
