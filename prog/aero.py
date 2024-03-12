import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")
from numba import njit

#%% geometry

@njit
def rect(i): # 1 <= i <= n+m
    if i <= n:
        return array_wing[i-1].reshape((4,3))
    return array_tail[i-1-n].reshape((4,3))


@njit
def tau1(i):
    r = rect(i)
    A, B, C, D = r[0], r[1], r[2], r[3]
    t = (B + C - A - D)/2
    return t/np.linalg.norm(t)


@njit
def normal(i):
    r = rect(i)
    A, B, C, D = r[0], r[1], r[2], r[3]
    t1 = tau1(i)
    t2 = (C + D - A - B)/2
    n = np.cross(t1, t2)
    return n/np.linalg.norm(n)


@njit
def tau2(i):
    t2 = np.cross(normal(i), tau1(i))
    return t2/np.linalg.norm(t2)


@njit
def s(i):
    r = rect(i)
    d1 = r[2] - r[0]
    d2 = r[3] - r[1]
    return 0.5*np.linalg.norm(np.cross(d1, d2))


@njit
def x(i):
    r = rect(i)
    A, B, C, D = r[0], r[1], r[2], r[3]
    return (A + B + C + D)/4


@njit
def j_sep_plus(i): # 1 <= i <= m
    if m == 20: # wing_10_20
        return 10*(21 - i) 
    if m == 10: # wing_20_10
        return 20*(11 - i)
    if m == 40: # wing_20_40
        return 20*(41 - i)

@njit
def j_sep_minus(i):
    if m == 20: # wing_10_20
        return 10*(20 + i)
    if m == 10: # wing_20_10
        return 20*(10 + i)
    if m == 40: # wing_20_40
        return 20*(40 + i)


file_wing = ['wing_10_20.dat', 
             'wing_20_10.dat', 
             'wing_20_40.dat']
file_tail = ['wing_10_20_tail.txt', 
             'wing_20_10_tail.txt', 
             'wing_20_40_tail.txt']
k = 2

array_wing = np.genfromtxt(file_wing[k])
array_tail = np.genfromtxt(file_tail[k])
n = len(array_wing)
m = len(array_tail)


#%% right side

alpha = np.pi/36  # 5 градусов
beta = 0
w_inf = np.array([np.cos(alpha)*np.cos(beta),
                  np.sin(alpha)*np.cos(beta),
                  np.sin(beta)])
f = np.zeros(m+n+1)
for i in range(1, n+1):
    f[i-1] = -np.dot(w_inf, normal(i))


#%% matrix A

@njit
def Lifanov(a, b, x):
    if np.linalg.norm(a-b) < 1e-8:
        return np.zeros(3)
    if  np.abs(np.linalg.norm(b-a)**2 * np.linalg.norm(x-a)**2 - \
        ((b-a)@(x-a))**2 ) < 1e-8:
        return np.zeros(3)
    return 1/(4*np.pi) * np.cross(b-a, x-a) / (
    np.linalg.norm(b-a)**2 * np.linalg.norm(x-a)**2 - \
    ((b-a)@(x-a))**2 ) * (
    (b-a)@(x-b)/np.linalg.norm(x-b) - \
    (b-a)@(x-a)/np.linalg.norm(x-a))
            

@njit
def W(j, x):
    r = rect(j)
    A, B, C, D = r[0], r[1], r[2], r[3]
    return Lifanov(A, B, x) + \
           Lifanov(B, C, x) + \
           Lifanov(C, D, x) + \
           Lifanov(D, A, x)
           
           
A = np.zeros((n, n+m))
for i in range(1, n+1):
    for j in range(1, n+m+1):
        A[i-1, j-1] = W(j, x(i)) @ normal(i)


#%% solve

A_hat = np.zeros((n+m+1, n+m+1))
A_hat[:n, :n+m] = A
A_hat[:n, -1] = np.ones(n)
B = np.zeros((m,n))
for i in range(1, m+1):
    B[i-1, j_sep_plus(i)-1] = 1
    B[i-1, j_sep_minus(i)-1] = -1
A_hat[n:n+m, :n] = B
A_hat[n:n+m, n:n+m] = -np.eye(m)
for i in range(1, n+1):
    A_hat[-1, i-1] = s(i)
        
g = np.linalg.solve(A_hat, f)


#%% calc

def w(x):
    return np.sum([g[j-1]*W(j, x) for j in range(1, n+m+1)], axis=0)


def w_tot(x):
    return w_inf + w(x)


def C_p(x):    
    return 1 - 4*np.linalg.norm(w_tot(x))**2


F_p = -1/5*np.sum([normal(i)*C_p(x(i))*s(i) for i in range(1, n+1)], axis=0)


#%% pressure

p = [str(C_p(x(i)))+'\n' for i in range(1, n+1)]
with open('pressure.txt', 'w') as f:
    f.write(str(n)+'\n')
    f.writelines(p)


#%% plot


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
# points_wing = np.concatenate((array_wing[:,  :3],
#                               array_wing[:, 3:6],
#                               array_wing[:, 6:9],
#                               array_wing[:, 9:12]))
# points_wing = np.unique(points_wing, axis=0)
# ax.scatter(points_wing[:, 0], points_wing[:, 1], points_wing[:, 2])

colloc = np.array([x(i) for i in range(n)])
C_p_colloc = np.array([C_p(x) for x in colloc])
graph = ax.scatter(colloc[:, 0], colloc[:, 1], colloc[:, 2], c=C_p_colloc)
fig.colorbar(graph)

# for e in array_wing:
#     ax.plot([e[0], e[3]], [e[1], e[4]], zs=[e[2],e[5]], c='blue')
#     ax.plot([e[3], e[6]], [e[4], e[7]], zs=[e[5],e[8]], c='blue')
#     ax.plot([e[6], e[9]], [e[7], e[10]], zs=[e[8],e[11]], c='blue')
#     ax.plot([e[9], e[0]], [e[10], e[1]], zs=[e[11],e[2]], c='blue')


# ax.set_xlim(0, 1)
# ax.set_ylim(-0.5, 0.5)

# X = np.zeros((n, 3))
# for i in range(n):
#     X[i] = x(i)
# ax.scatter(*X.T, c=(f>0))