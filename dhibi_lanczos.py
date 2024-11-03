from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as la
import copy as cp

from colour import Color
'''
To get the different plots from my report you can uncomment/comment the docstrings type comment below the "# i - plot" comments
where i = 1 or 2
'''


# initialisation
e1 = -13.6
e2 = -3.4
N = 10
Nit = 100
d = [e1,e2]*N
guess = np.random.randint(5,10,2*N)
t = np.array([3*i/100 for i in range(0,Nit,10)])
t1 = np.copy(t)
t2 = np.copy(t)*1.5

def hamiltonians(e1,e2,t1,t2,N):
    H = np.zeros((t1.size,2*N,2*N))
    diagonal = np.diag([e1,e2]*N)
    for i in range(t1.size):
        diagonal_above = np.diag([-t1[i], -t2[i]]*(N-1),2)
        diagonal_below = np.diag([-t1[i], -t2[i]]*(N-1),-2)

        periodic_boundary_above = np.diag([-t1[i], -t2[i]], 2*(N-1))
        periodic_boundary_below = np.diag([-t1[i], -t2[i]], -2*(N-1))

        H[i,:,:] = diagonal + diagonal_above + diagonal_below + periodic_boundary_above + periodic_boundary_below
    return H

# diagonalization methods
def exact_diagonalization(H):
    eigenvalues = la.eigvals(H)
    return eigenvalues

def lanczos(H,guess):
    Lv=np.zeros(H.shape, dtype=complex) # creates matrix for Lanczos vectors
    Hk=np.zeros(H.shape, dtype=complex) # creates matrix for the Hamiltonian in Krylov subspace

    eigenvalues = []
    Lv[0]=guess/la.norm(guess) # creates the first Lanczos vector as the normalized guess vector

    # performs the first iteration step of the Lanczos algorithm
    w=np.dot(H,Lv[0]) 
    a=np.dot(np.conj(w),Lv[0])
    w=w-a*Lv[0]
    Hk[0,0]=a
    #Performs the iterative steps of the Lanczos algorithm
    for j in range(1,len(guess)):
        b=np.sqrt((np.dot(np.conj(w),w)))
        Lv[j]=w/b
         
        w=np.dot(H,Lv[j])
        a=np.dot(np.conj(w),Lv[j])
        w=w-a*Lv[j]-b*Lv[j-1]
        # recovery of orthogonality
        for i in range(j):
            q = np.dot(np.conj(Lv[i]),w)
            w = (w - q*Lv[i])/(1-q**2)
        
        #Creates tridiagonal matrix Hk using a and b values
        Hk[j,j]=a.real
        Hk[j-1,j]=b.real
        Hk[j,j-1]=b.real

        block = Hk[0:j,0:j]
        eigval_j = la.eig(block)[0]
        eigenvalues.append(np.sort(eigval_j))
    return (eigenvalues)


H = hamiltonians(e1,e2,t1,t2,N)

eigenvalues = np.sort(exact_diagonalization(H),axis=1)

# plots initialization
fig, ax = plt.subplots()

# 1 - plot for the convergence of Lanczos method
'''
X = np.arange(1,eigenvalues.shape[1]+1) #(1,2*N+1)
# define font sizes
SIZE_DEFAULT = 16
SIZE_LARGE = 18
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

# bords du plot
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(min(X), max(X))

# set ticks
ax.set_xticks(X)
ax.set_yticks([e1,e2])
ax.tick_params(labelsize=SIZE_DEFAULT)

# set labels
ax.set_xlabel("Iteration", fontsize=SIZE_LARGE)
ax.set_ylabel("Energy (eV)",fontsize=SIZE_LARGE)


red = Color("red")
colors = list(red.range_to(Color("black"),eigenvalues.shape[1]))


# exact eigenvalues
# we consider only one matrix with a hopping parameter close to the transition
exact_eigenvalues = eigenvalues[6,:]
#print(f'exact_eigenvalues.shape = {exact_eigenvalues.shape}') # '(200,)
Y0 = np.ones_like(X)*exact_eigenvalues[0]
ax.plot(X, Y0, linestyle='--', color=colors[0].hex, label='exact eigenvalues')

for i in range(1,exact_eigenvalues.size):
    ei = exact_eigenvalues[i]
    Yi = np.ones_like(X)*ei

    ax.plot(X, Yi, linestyle='--', color=colors[i].hex)


# eigenvalues with lanczos method
H_to_diag = H[6,:,:]
lanczos_eigenvalues = lanczos(H_to_diag,guess)

for j in range(2*N):
    Yj = []
    for i in range(j,len(lanczos_eigenvalues)):
        Yj.append(lanczos_eigenvalues[i][j])
    
    ax.plot(np.arange(j+1,len(lanczos_eigenvalues)+1), Yj, marker='+', color=colors[j].hex)



# plotting eigenvalues for one atom
Y1 = np.ones_like(X)*e1
Y2 = np.ones_like(X)*e2
ax.plot(X, Y2, linestyle='--', color='#d60000', linewidth=2)
ax.plot(X, Y1, linestyle='--', color='#d60000', linewidth=2)
ax.text(
    X[-1] * 1.01,
    e1,
    "H ground state",
    color='#d60000',
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)
ax.text(
    X[-1] * 1.01,
    e2,
    "H first excitation",
    color='#d60000',
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)
#plt.savefig("lanczos.png", dpi=300)


plt.show()
'''

# 2 - plot for the transition with numpy :
'''
red = Color("red")
colors = list(red.range_to(Color("black"),eigenvalues.shape[1]))

# define font sizes
SIZE_DEFAULT = 16
SIZE_LARGE = 18
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

# bords du plot
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(min(X), max(X))

# set ticks
ax.set_xticks(t)
ax.set_yticks([e1,e2])
ax.tick_params(labelsize=SIZE_DEFAULT)

# set labels
ax.set_xlabel("Hopping parameter t", fontsize=SIZE_LARGE)
ax.set_ylabel("Energy (eV)",fontsize=SIZE_LARGE)

# plot the eigenvalues
for i in range(eigenvalues.shape[1]):
    ax.plot(t,eigenvalues[:,i], color=colors[-1].hex,linestyle='-', marker='.', markersize=10 )

# finding the metal-insulator transition
bandgap = np.min(eigenvalues[:,N+1:],axis=1)-np.max(eigenvalues[:,0:N], axis=1)
k_min = np.argmin(np.abs(bandgap)) # index of the transition

# plotting the transition
ax.vlines(t[k_min],linestyle='--', color='#d60000', linewidth=3,ymin=eigenvalues[-1,0]*1.1,ymax=eigenvalues[-1,-1]*1.3)
ax.text(
    t[k_min],
    np.max(eigenvalues)*1.5,
    "Metal-insulator transition",
    color='#d60000',
    fontweight="bold",
    horizontalalignment="center",
    verticalalignment="center",
)

# eigenvalues for one atom
Y1 = np.ones_like(t)*e1
Y2 = np.ones_like(t)*e2
ax.plot(t, Y2, linestyle='--', color='#d60000', linewidth=4)
ax.plot(t, Y1, linestyle='--', color='#d60000', linewidth=4)
ax.text(
    t[-1] * 1.01,
    e1,
    "H ground state",
    color='#d60000',
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)
ax.text(
    t[-1] * 1.01,
    e2,
    "H first excitation",
    color='#d60000',
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)
#plt.savefig("lanczos.png", dpi=300)
plt.show()
'''