from pathlib import Path
p=Path('/holdout_task/node_01.py')
s=p.read_text()
start=s.index('    cq = _ls(')
end=s.index('    for z in ',start)
s=s[:start]+'''    # Profile the panel transfer using reference rows only.
    def ref(z, mode):
        if mode == 0:
            return np.c_[np.ones(len(z)), _cross(z)]
        if mode == 1:
            return np.c_[np.ones(len(z)), z, z*z, _cross(z)]
        w = np.tanh((mode-1)*0.5*z)
        return np.c_[np.ones(len(z)), w, w*w, _cross(w)]
    errors=[]
    for mode in range(7):
        h=ref(u,mode); Q, d, _=np.linalg.svd(h,full_matrices=False)
        Q=Q[:,d>1e-9*d[0]]
        errors.append(np.mean(((p['v1']-Q@(Q.T@p['v1']))/np.maximum(1-(Q*Q).sum(1),0.05)[:,None])**2))
    mode=int(np.argmin(errors))
    cq = _ls(ref(u, mode), p['v1'])
    cc = _ls(np.c_[np.ones(len(u)), np.cos(2.4*u)], p['v2'])
'''+s[end:]
s=s.replace('p1 = np.c_[np.ones(len(z)), q]@cq','p1 = ref(z, mode)@cq')
start=s.index('\n\ndef _paths')
s=s[:start]+'''
        bb=z@b[1:]+b[0]
        for g in (p1,p2,bb,np.c_[z,p1,p2],np.c_[bb,p1,p2]):
            gp=_cross(g)
            for k in (0.8,1.6,2.4,3.5):
                yield np.c_[z, g, np.sin(k*g),np.cos(k*g), np.tanh(k*g),gp,np.sin(k*gp),np.cos(k*gp)]
        yield np.c_[z,p1,p2,bb,_cross(np.c_[z,p1,p2,bb])]
'''+s[start:]
s=s.replace('    vals=[]\n    for k in range(h.shape[1]):','''    # Quadratic and local kernels on the same fold-standardized coordinates.
    dn=(l*l).mean(1); dv=(v*v).mean(1); scale=max(l.shape[1],1)
    kd=np.maximum(dn[:,None]+dn[None,:]-2*g/scale,0)
    kv=np.maximum(dv[:,None]+dn[None,:]-2*vv/scale,0)
    for gamma in (0.25,1.):
        K=np.exp(-gamma*kd); V=np.exp(-gamma*kv)
        out.append(V@np.linalg.solve(K+0.06*np.eye(len(l)),h)+m)
    vals=[]
    for k in range(h.shape[1]):''')
p.write_text(s)
