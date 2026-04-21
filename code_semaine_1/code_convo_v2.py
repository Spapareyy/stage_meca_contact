import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

#paramètres
N=200
L=1
nu=0.3
E=69e9 #alu
p0=10e5 #pression atmosphérique ?
pi=np.pi

lambda_l=L/2   #plus grande longueur d'onde possible 
lambda_s=L/(N/3) #plus petite longueur d'onde possible 
q_s=2*pi/lambda_s  #plus grand nombre d'onde
q_l=2*pi/lambda_l  #plus petit nombre d'onde 


#calcul du déplacement vertical du solide déformable
x=np.linspace(0,L,N,endpoint=False) #permet de supprimer le dernier point pour que FFT et analytique soit 'identiques'
x1,x2= np.meshgrid(x,x)  #grille 2D

ptest=p0*np.cos((2*pi*x1)/L)  #pression test pour comparer avec l'analytique

freq=2*pi*np.fft.fftfreq(N,L/N)  # fftfreq donne cycle/longueur donc multiplication par 2pi pour avoir des radian/longueur

q1,q2=np.meshgrid(freq,freq)
qnorm=np.sqrt((q1**2)+(q2**2))

qnorm[0,0]=1 # car division par 0 impossible

B_t=(2*(1-(nu**2)))/(E*qnorm)

B_t[0,0]=0 # on remet juste la valeur à 0 

p_t=np.fft.fft2(ptest) 

u_t=B_t*p_t
u_num=np.fft.ifft2(u_t)

u_an=((p0*(1-(nu**2))*L)/(pi*E))*(np.cos((2*pi*x1)/L))




plt.figure(1)

plt.plot(x,u_num[N//2,:],label='numérique')
plt.plot(x,u_an[N//2,:],label='analytique')

plt.xlabel("position sur la surface du matériau en mètres")
plt.ylabel("déplacement u3 en mètres")
plt.title("déplacement vertical du solide déformable selon x1 en x2 = L/2 ")

plt.legend()
plt.grid()


diff=np.abs(u_num-u_an)

err_abs_max= np.max(diff)
err_rel=(err_abs_max/np.max(np.abs(u_an)))*100
print("\nPour le déplacement vertical du solide déformable : ")

print("erreur absolue maximale :",err_abs_max,"mètres")

print("erreur relative : ",err_rel,"%")

# si on augmente N on affine le maillage ce qui augmente la précision et diminue l'errreur




#%%


## minimisation de l'énergie

#corps rigide 
h=1e-6* np.cos(2*pi*x1/L) - 0.2e-6  # ordre de grandeur des rugosités
#h=np.cos(2*pi*x1/L)* np.cos(2*pi*x2/L)

#fonction à minimiser 
def contact_energie(p_liste):
    
    p_grille=p_liste.reshape((N,N)) #on transforme la pression initiale en grille
    
    p_t=np.fft.fft2(p_grille) #transformée de fourier
    u_t=B_t*p_t #multiplication
    
    u_num=np.real(np.fft.ifft2(u_t)) #on repasse en spatial
    
    energie=np.real(np.sum(0.5*p_grille*u_num-p_grille*h)).item() #.item donne un scalaire
    grad=np.real((u_num-h).flatten())
    #le flatten sert a transformer un tableau a plusieurs dimensions en vecteur à une seule ligne
    return energie, grad 




p_init = np.zeros(N * N) #pression nulle 
bornes = [(0, None) for i in range(N * N)] #la pression est au moins égale à 0 et pas de maximum ( condition de non pénétrabilité)


res = minimize(contact_energie, p_init, method='L-BFGS-B', jac=True, bounds=bornes, options={'gtol': 1e-12, 'ftol': 1e-12})
#le jac =true permet de converger plus vite 

#récupération des résultats
p_opt = res.x.reshape((N, N)) #on prend les résultats en grille 2D
u3_opt = np.real(np.fft.ifft2(B_t * np.fft.fft2(p_opt)))


# Comparaison avec l'analytique

amplitude_h=1e-6
E_star = E / (1 - nu**2) #module d'élasticité réduit
p_star = (np.pi * E_star * amplitude_h) / L #pression de contact complet
p_moy = np.mean(p_opt) 

#calcul de la demi-largeur de contact (a)
a_theo = (L / np.pi) * np.arcsin(np.sqrt(p_moy / p_star)) if p_moy < p_star else L / 2 # si pmoy est supérieur ou égal a pstar le contact est complet donc largeur maximale
a_num = (np.count_nonzero(p_opt[0, :] > 1e-5) * (L / N)) / 2  # des que la pression est non nulle on compte le point et on multiplie par la distance entre deux points


#calcul des profils de pression p(x)

p_analytique = (2 * p_moy / np.sin(np.pi * a_theo / L)**2) * np.abs(np.cos(np.pi * x / L)) * np.sqrt(np.maximum(0, np.sin(np.pi * a_theo / L)**2 - np.sin(np.pi * x / L)**2))
p_numerique = p_opt[0, :]




#tracé des pressions analytique et numérique

plt.figure(2)
plt.plot(x, p_numerique, 'o', label='numérique')
plt.plot(x, p_analytique, '-', label='analytique', color='red')

plt.xlabel('position x en mètres')
plt.ylabel('pression en Pascals')
plt.title('Comparaison de la pression analytique et numérique')
plt.legend()


erreur_pression = np.linalg.norm(p_numerique - p_analytique) / np.linalg.norm(p_analytique) * 100  #on utilise linalg pour eviter la division par 0

print("\nPour les calculs de pression")
print("erreur relative pression : ",erreur_pression," %")

# calcul de l'erreur relative sur la largeur
erreur_largeur = abs(a_num - a_theo) / a_theo * 100

print("demi-largeur de contact analytique : ",a_theo)
print("demi-largeur de contact numérique  : ",a_num)
print("erreur relative largeur        : ",erreur_largeur ,"%")



#tracé du solide rigide et du solide déformable avec la pression numérique
plt.figure(3)
fig, ax1 = plt.subplots()

ax1.plot(x, h[0, :], 'k--', label='solide rigide')
ax1.plot(x, u3_opt[0, :], 'b-', label='solide déformable')
ax1.set_ylabel("déplacement u3 en mètres")
ax1.set_xlabel("position x en mètres")
ax2 = ax1.twinx()
ax2.fill_between(x, 0, p_opt[0, :], color='green', label='pression')
ax2.set_ylabel("pression en pascals",color='green')


h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper right')
plt.title(r"solide déformable vs corps rigide avec médiane corps rigide=-0.5$\mu$m")
plt.show()




#topographie des zones de contact 

plt.figure(4, figsize=(12, 5))
plt.suptitle("Comparaison de la topographie entre le résultat analytique et numérique")

#numérique
plt.subplot(1, 2, 1)
plt.imshow(p_opt, cmap='viridis')
plt.colorbar(label='Pression en pascals')
plt.title('Numérique (L-BFGS-B)')
plt.xlabel('x en mètres')
plt.ylabel('y en mètres')

#analytique
plt.subplot(1, 2, 2)
p_analytique_2d = np.tile(p_analytique, (N, 1))
plt.imshow(p_analytique_2d, extent=[0, L, 0, L], origin='lower', cmap='viridis')
plt.colorbar(label='Pression en pascals')
plt.title('Analytique ')
plt.xlabel('x en mètres')
plt.ylabel('y en mètres')

plt.tight_layout() 
plt.show()





#%%
# création d'une rugosité aléatoire



np.random.seed(4) # pour avoir tjrs le même aléatoire
H=0.9 # nombre de Hurst 
alpha=(2*(H+1))

phi = np.random.uniform(-pi, pi, (N, N))

#filtre de q avant la puissance
ampli = np.zeros_like(qnorm)
masque = (qnorm >= q_l) & (qnorm <= q_s)
ampli[masque] = np.sqrt(qnorm[masque]**(-alpha))

ampli[0,0]=0 #pour que l'amplitude moyenne soit centrée en 0
h_rug_f=ampli*np.exp(1j*phi)

h_rug_s=np.fft.ifft2(h_rug_f)


#premiere ligne de la surface rugueuse
plt.figure(5)

plt.plot(x,h_rug_s[0,:])
plt.xlabel("position x en mètre")
plt.ylabel("hauteur de rugosité en micromètre")
plt.title(" Rugosité générée aléatoirement en y=0 pour H=0.3")




#%%
# C(q) en fonction du nombre d'onde analytique et numérique



#psd numerique
surf_reel_freq=np.fft.fft2(h_rug_s.real)   #on prend la partie reelle de la reponse frequentielle
C_num=np.abs(surf_reel_freq)**2 # on la met au carré car le PSD est égal à l'amplitude au carré


#psd en loglog

#on utilise qnorm (non shifté) pour le calcul
q_1d = qnorm.flatten() 
# on ne garde que les points dans la borne [q_l, q_s]
mask_1d = (q_1d >= q_l) & (q_1d <= q_s)
q_final = q_1d[mask_1d]
C_analytique = q_final**(-alpha) 

#pour le PSD en echelle loglog on transforme en 1D
     
C_num_1d = C_num.flatten()[mask_1d]  
C_ana_1d = C_analytique.flatten()




plt.figure(7,figsize=(8, 6))
pente=(np.log10(C_num_1d[-1])-np.log10(C_num_1d[0]))/(np.log10(q_final[-1])-np.log10(q_final[0]))
#tracé du psd numérique en loglog
plt.loglog(q_final, C_num_1d, 'o', alpha=0.1,label=f'numérique (pente {pente:.2f})')
#tracé du psd analytique en loglog
plt.loglog(q_final, C_ana_1d, 'r-', label=f'analytique (pente -{alpha:.2f})')

plt.xlabel('Nombre d\'onde q [rad/m]')
plt.ylabel('PSD [m⁴]')
plt.title('Comparaison du PSD : analytique vs numérique')
plt.legend()
plt.grid()
plt.show()


#psd spectral




plt.figure(8,figsize=(8, 6))
#on utilise log pour mieux voir le détail du PSD
plt.figure(8,figsize=(8, 6))
# L'extent correct pour un spectre centré (de -q_s à +q_s environ)
q_max = np.max(qnorm)
im = plt.imshow(np.log10(np.fft.fftshift(C_num)), 
                extent=[-q_max, q_max, -q_max, q_max], 
                cmap='viridis')
plt.colorbar(im, label='log10(PSD)')
plt.title(f'PSD en Donut (alpha={alpha:.2f})')
plt.xlabel("Fréquence qx [rad/m]")
plt.ylabel("Fréquence qy [rad/m]")
plt.show()





#%%
# on essaie de prendre un corps rigide avec des rugosités cette fois



med=-0.8e-6 # mediane du corps rigide
h=h_rug_s.real+med

#fonction à minimiser 
def contact_energie(p_liste):
    
    p_grille=p_liste.reshape((N,N)) #on transforme la pression initiale en grille
    
    p_t=np.fft.fft2(p_grille) #transformée de fourier
    u_t=B_t*p_t #multiplication
    
    u_num=np.real(np.fft.ifft2(u_t)) #on repasse en spatial
    
    energie=np.real(np.sum(0.5*p_grille*u_num-p_grille*h)).item() #.item donne un scalaire
    grad=np.real((u_num-h).flatten())
    #le flatten sert a transformer un tableau a plusieurs dimensions en vecteur à une seule ligne
    return energie, grad 



p_init = np.zeros(N * N) #pression nulle 
bornes = [(0, None) for i in range(N * N)] #la pression est au moins égale à 0 et pas de maximum ( condition de non pénétrabilité)


res = minimize(contact_energie, p_init, method='L-BFGS-B', jac=True, bounds=bornes, options={'gtol': 1e-12, 'ftol': 1e-12})
#le jac =true permet de converger plus vite 

#récupération des résultats
p_opt = res.x.reshape((N, N)) #on prend les résultats en grille 2D
u3_opt = np.real(np.fft.ifft2(B_t * np.fft.fft2(p_opt)))

#on calcule la surface d'un element
dS = (L / N)**2

#on fait la somme des pressions * surface
W_calcule = np.sum(p_opt) * dS

print(f"\nForce totale générée par ce contact : {W_calcule:.2f} Newtons")

#tracé du solide rigide et du solide déformable avec la pression numérique
plt.figure(9)
fig, ax1 = plt.subplots()

ax1.plot(x, h[0, :], 'k--', label='solide rigide')
ax1.plot(x, u3_opt[0, :], 'b-', label='solide déformable')
ax1.set_ylabel("déplacement u3 en mètres")
ax1.set_xlabel("position x en mètres")
ax2 = ax1.twinx()
ax2.fill_between(x, 0, p_opt[0, :], color='green', label='pression',alpha=0.3)
ax2.set_ylabel("pression en pascals",color='green')


h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper right')
plt.title(rf"solide déformable vs corps rigide avec médiane corps rigide={med*1e6:.1f}$\mu$m")
plt.show()


plt.figure(6, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title(f"map 2d de la rugosité générée aléatoirement pour H = {H}",y=1.05)
plt.imshow(h_rug_s.real, extent=[0, L, 0, L], origin='lower', cmap='viridis')
plt.colorbar(label='Hauteur [µm]')
plt.xlabel('position x en mètre')
plt.ylabel('position y en mètre')

plt.subplot(1, 2, 2)
plt.title(f"map 2d de la pression pour H = {H}",y=1.05)
plt.imshow(p_opt, extent=[0, L, 0, L], origin='lower', cmap='viridis')
plt.colorbar(label='pression en MPa')
plt.xlabel('position x en mètre')
plt.ylabel('position y en mètre')

plt.show()

#%% 
#cette fois on impose la force W : nouvelle contrainte



#on prend un N petit pour la méthode avec contraintes linéaires car elle est longue
N=20

#calcul du déplacement vertical du solide déformable
x=np.linspace(0,L,N,endpoint=False) #permet de supprimer le dernier point pour que FFT et analytique soit 'identiques'
x1,x2= np.meshgrid(x,x)  #grille 2D

freq=2*pi*np.fft.fftfreq(N,L/N)  # fftfreq donne cycle/longueur donc multiplication par 2pi pour avoir des radian/longueur

q1,q2=np.meshgrid(freq,freq)
qnorm=np.sqrt((q1**2)+(q2**2))
qnorm[0,0]=1 # car division par 0 impossible

B_t=(2*(1-(nu**2)))/(E*qnorm)

B_t[0,0]=0 # on remet juste la valeur à 0 



# création d'une rugosité aléatoire
np.random.seed(4) # pour avoir tjrs le même aléatoire
H=0.9 # nombre de Hurst 
alpha=(2*(H+1))

phi = np.random.uniform(-pi, pi, (N, N))

# APPLICATION DU MASQUE AVANT LA PUISSANCE
ampli = np.zeros_like(qnorm)
masque = (qnorm >= q_l) & (qnorm <= q_s)
ampli[masque] = np.sqrt(qnorm[masque]**(-alpha))

ampli[0,0]=0 #pour que l'amplitude moyenne soit centrée en 0
h_rug_f=ampli*np.exp(1j*phi)

h_rug_s=np.fft.ifft2(h_rug_f)






med=0 # mediane du corps rigide
h=h_rug_s.real+med
h=(h/np.std(h))*1e-6

W=6e4 # force imposée

dS = (L / N)**2 #surface d'un element

#matrice A (1 ligne, N*N colonnes) remplie de dS
A = np.full((1, N**2), dS)

#la contrainte : A * p doit être exactement égal à W
# lb = lower bound (limite basse), ub = upper bound (limite haute)
contrainte_W = LinearConstraint(A, lb=W, ub=W)

#fonction à minimiser 
def contact_energie(p_liste):
    
    p_grille=p_liste.reshape((N,N)) #on transforme la pression initiale en grille
    
    p_t=np.fft.fft2(p_grille) #transformée de fourier
    u_t=B_t*p_t #multiplication
    
    u_num=np.real(np.fft.ifft2(u_t)) #on repasse en spatial
    
    energie=np.real(np.sum(0.5*p_grille*u_num-p_grille*h)).item() #.item donne un scalaire
    grad=np.real((u_num-h).flatten())
    #le flatten sert a transformer un tableau a plusieurs dimensions en vecteur à une seule ligne
    return energie*1e9, grad *1e9 #coef


p_init = np.zeros(N * N) #pression nulle 
p_moyen = W / (L**2)
p_init = np.full(N * N, p_moyen) #pression non nulle pour faciliter le calcul au solveur 
bornes = [(0, None) for i in range(N * N)] #la pression est au moins égale à 0 et pas de maximum ( condition de non pénétrabilité)


res = minimize(contact_energie, p_init, method='SLSQP', jac=True, bounds=bornes,constraints=[contrainte_W])
#le jac =true permet de converger plus vite 

#récupération des résultats
p_opt = res.x.reshape((N, N)) #on prend les résultats en grille 2D
u3_opt = np.real(np.fft.ifft2(B_t * np.fft.fft2(p_opt)))


#%%
#tracé du solide rigide et du solide déformable avec la pression numérique
plt.figure(9)
fig, ax1 = plt.subplots()

ax1.plot(x, h[0, :]*1e6, 'k--', label='solide rigide',zorder=11)
ax1.plot(x, u3_opt[0, :]*1e6, 'b-', label='solide déformable',zorder=10)
ax1.set_ylabel("déplacement u3 en micromètres")
ax1.set_xlabel("position x en mètres")
ax2 = ax1.twinx()
ax2.fill_between(x, 0, p_opt[0, :], color='green', label='pression',zorder=1,alpha=0.3)
ax2.set_ylabel("pression en pascals",color='green')


h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper right')
plt.title(rf"solide déformable vs corps rigide avec médiane corps rigide={med*1e6:.1f}$\mu$m")
plt.show()

#%%