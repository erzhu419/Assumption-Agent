import runpy
import numpy as np
n = runpy.run_path('/tmp/test_robin_eq01.py')
m = n['m']
def study(kind):
    s = n['generate'](821 + kind, (28,28,28), kind)
    s.update(zip(('Lx','Ly','Lz'), ((.55,1.15,2.2),(2.2,.55,1.15))[kind]))
    if kind:
        x,y,z = np.indices(s['kappa_base'].shape)
        s['kappa_mod2'] = np.where((y+z//3)%11<5,17.,.002)*s['kappa_base']
    order = ([0,6,2,3,4,8,1,7,5],[5,0,8,2,7,3,6,4,1])[kind]
    s['thetas'] = s['thetas'][order]
    return s
if __name__ == '__main__':
    s = study(1)
    Ax = [n['oracle_operator'](s,t)[0].tocsr() for t in np.sort(s['thetas'])]
    old = m._iterate
    i = 0
    def inspect(A,b,M,g):
        global i
        j=i; i+=1
        u = old(A,b,M,g)
        B = Ax[j//3]; norm = np.linalg.norm(b)
        a = np.linalg.norm(b-A@u)/norm; r = np.linalg.norm(b-B@u)/norm
        if max(a,r)>1e-8:
            f=m._factor(A); Al=A.astype(np.longdouble); bl=b.astype(np.longdouble)
            v=u.copy()
            for _ in range(2):
                v += f.solve(np.asarray(bl-Al@v,dtype=float))
            w=m._factor(B).solve(b)
            print(f'theta={np.sort(s["thetas"])[j//3]} mode={j%3}: orig internal/oracle={a:.3g}/{r:.3g}; refine internal/oracle={np.linalg.norm(b-A@v)/norm:.3g}/{np.linalg.norm(b-B@v)/norm:.3g}; oracle direct internal/oracle={np.linalg.norm(b-A@w)/norm:.3g}/{np.linalg.norm(b-B@w)/norm:.3g}; matrix difference action={np.linalg.norm((A-B)@u)/norm:.3g}',flush=True)
        return u
    m._iterate = inspect
    m.solve_robin_continuation_tomography(s)
