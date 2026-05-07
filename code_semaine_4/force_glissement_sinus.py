import tamaas as tm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

N=200
if len(sys.argv) > 3:
    load = float(sys.argv[1])
    suff_load = sys.argv[1]
    hurst = float(sys.argv[2])
    suff_hurst = sys.argv[2]
    pas = int(sys.argv[3])
    suff_pas = sys.argv[3]
else: #si execution via spyder
    load = 1e-5 
    hurst = 0.7
    pas = 100    #changer valeur pour décaler de x pas
    suff_load = str(load)
    suff_hurst = str(hurst)
    suff_pas = str(pas)

os.makedirs("resultats", exist_ok=True)


L =1.

#surface sinusoidale
x_tmp = np.linspace(0, L, N, endpoint=False)
y_tmp = np.linspace(0, L, N, endpoint=False)
xx, yy = np.meshgrid(x_tmp, y_tmp, indexing='ij')

h0=1e-6 #ampltiude des bosses
# Surface simple 2D (4 bosses spatiales)
surface = h0 * np.sin(2 * np.pi * 4 * xx / L) * np.sin(2 * np.pi * 4 * yy / L)

x = np.linspace(0, L, N, endpoint=False)
#calcul du psd
C_q_2D = tm.Statistics2D.computePowerSpectrum(surface)

#vecteurs d'ondes
freqs_x = np.fft.fftfreq(N, d=L/N) * 2 * np.pi  #fréquences spatiales pour l'axe x
freqs_y = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi  # rfftfreq crée la dimension N//2 + 1,  fréquences spatiale pour l'axe y
qx, qy = np.meshgrid(freqs_x, freqs_y,indexing='ij')  #grille 2D des fréquences spatiales
q_norm = np.sqrt(qx**2 + qy**2) #fréquence spatiale absolue 


model = tm.Model(tm.model_type.basic_2d, [L, L], [N, N])
model.E= 1.
nu=0.5
model.nu = nu
V_cible=0.1 #pour avoir la meme vitesse peu importe la valeur de N
pas_temps=(L/N)/V_cible

G_i = np.array([3.0])   # si on a k=0.1 , et Einf=1 on a dE=9 et E=3*G avec nu=0.5 donc G=dE/3=3
tau_i = np.array([0.1]) # taurelax= k*tau_fluage avec k=0.1 et tau_fluage =1 , taurelax=0.1

solver = tm.MaxwellViscoelastic(model, surface, 1e-9,
                                time_step=pas_temps,
                                shear_moduli=G_i,
                                characteristic_times=tau_i)

#solveur
dx=L/N
dS = dx * dx
historique_ft = []  #pour enregistrer l'evolution de ft et la plot qu'a la fin
historique_A_reel = [] #aire de contact reelle
temps = []

#on calcule la pente initiale (gradient selon l'axe x)
#on ne s'interesse qu'a la pente selon x
pente_y, pente_x = np.gradient(surface, dx)

#boucle
#on boucle exactement 'pas + 1' fois pour s'arrêter sur le pas demandé
for i in range(pas + 1):
    solver.solve(load)
    
    #on décale la surface seulement si on n'est pas au dernier pas
    # (pour que l'image finale corresponde bien à l'état après la résolution)
    ft = np.sum(model.traction * pente_x) * dS
    A_reel = np.sum(model.traction > 0) * dS
    historique_ft.append(ft)
    historique_A_reel.append(A_reel)
    temps.append(i * pas_temps) # pas * time_step
    if i < pas:
        surface[:] = np.roll(surface, shift=-1, axis=1)
        pente_x[:] = np.roll(pente_x, shift=-1, axis=1) #on décale la pente avec la surface
#%%
##### tracé des surfaces  #####
fig_def, ax1 = plt.subplots(figsize=(10, 5))
plt.axvline(x=(pas/N )%1,ymin=0,ymax=1)
#ces 4 lignes servent a obtenir l'endroit avec la pression la plus élevée 
y_max = np.argmax(np.max(model.traction, axis=1)) #on prend l'indice de la pression la plus élevée parmi l'ensemble des pressions maximales de chaque ligne
#ymax=128 #ici on choisit n'importe quel endroit si on ne veut pas la pression maximale
h_cut = surface[y_max, :] #on prend la ligne de la surface rugueuse qui correspond à cette pression
u_cut = model.displacement[y_max, :] #on prend le deplacement de la surface deformee qui correspond a cette pression
p_cut = model.traction[y_max, :] #on prend le profil de pression de la ligne qui correspond a cette pression

#h_cut : profil de la surface rigide sur la ligne choisie
#u_cut : déplacement vertical de la surface déformée 

#on cherche le point de pression max sur la ligne du ymax
# et on force le solide déformable à toucher la surface exactement à cet endroit.

offset = np.max(h_cut - u_cut) #prend l'ecart le plus grand entre surf rigide et surf deformable

u_plot = u_cut+ offset

ax1.plot(x, h_cut , 'k', label='Solide rigide')
ax1.plot(x, u_plot, 'b-', label='Solide déformable')

ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)",title=f"Profil de contact (y={y_max}, Pas numéro {pas})")
ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)", title=f"Profil de contact (y={y_max}, Pas numéro {pas})")

#tracé de la pression
ax2 = ax1.twinx()
ax2.fill_between(x, 0, p_cut, color='green', alpha=0.3, label='Pression')
ax2.set_ylabel("Pression", color='green')
ax1.grid()
fig_def.legend(loc='upper right')
fig_def.savefig(f"resultats/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}.png")

if len(sys.argv) > 3:
    plt.close(fig_def)
else:
    plt.show()
 

#calcul du coef de frottement
fn = load * L * L
mu_final = ft / fn

chemin_txt = f"resultats/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}.txt"
with open(chemin_txt, "w") as f:
    
    f.write(f"Ft = {ft}\nmu = {mu_final}\nAire_reelle_finale = {A_reel}")



####### méthode persson #######
V = (L / N) / pas_temps   #vitesse de glissement (distance d'un pas / temps d'un pas)
omega = qx * V             #fréquence d'excitation vue par le solide déformable (rad/s)
Surface_totale = L * L

#calcul de la force de frottement théorique
F_analytique_t = []

for i, t in enumerate(temps):
    reponse_totale = 0
    for j in range(len(G_i)):
        g, tau = G_i[j], tau_i[j]
        #g correspond a E (inf)- E(0)
        terme_stationnaire = (g * 1j * omega * tau) / (1 + 1j * omega * tau) #dapres equation 37
        
        #cette formule inclut la mémoire de la position initiale (g* oscillation)
        oscillation = np.exp(-t/tau) * np.exp(-1j * omega * t)
        reponse_t = terme_stationnaire * (1 - oscillation) + (g * oscillation) #dapres equation 39, g* oscillation est la relaxation de la contrainte initiale
        
        reponse_totale += reponse_t  #on additionne les réponses de chaque branche
    
    E_perte_t = 2 * np.imag(reponse_totale) / (1 - nu) #on garde uniquement la partie imaginaire 
    contribution = qx*q_norm * C_q_2D * E_perte_t  #q*cos(phi)= qnorm*(qx/qnorm)= qnorm*qx
    integrale_q = np.sum(contribution)*2  #*2 car tamaas calcul le PSD pour la moitié des fréquences
    
    #on récupère l'aire réelle mesurée par Tamaas à cet instant précis
    A_reel_t = historique_A_reel[i]
    
    #on calcule le pourcentage de contact, entre 0 et 1
    P_q_t = A_reel_t / Surface_totale  #comme p_q_t dépend de q et de t on a décidé de pas l'inclure dans l'equation de f_t
    
    #on applique la formule de l'équation 18 (avec le 1/2 et cos(0)=1)
    f_t = 0.5* integrale_q * Surface_totale* P_q_t
    F_analytique_t.append(f_t)

F_analytique_t = np.array(F_analytique_t)




##### méthode carbone putignano #####
k = 0.1
tau = 1.0  
vit = (L / N) / pas_temps  

#la rugosité recule sur l'axe y dans le np.roll (axis=1, shift=-1), donc la vitesse relative est -vit
omega_car = -qy * vit  

#terme devant le G
M_qv = k + ((1 - k)/(1 - 1j *omega_car *tau))

solver_stat = tm.PolonskyKeerRey(model, surface, 1e-7)
solver_stat.solve(load)

#on modifie le G 
Green = model.operators['westergaard_neumann']['influence'][:]
model.operators['westergaard_neumann']['influence'][:] = Green * M_qv

#résolution avec le nouveau G
solver_stat.solve(load)

#calcul de la force asymptotique
ft_asymptote = np.sum(model.traction * pente_x) * dS
print(f"Force asymptotique (Carbone-Putignano) : {ft_asymptote:.4e}")

#calcul de l'erreur relative entre la fin de la simulation et l'asymptote
erreur_relative = abs(historique_ft[-1] - ft_asymptote) / ft_asymptote * 100
force_totale_n = load * L**2 #force normale réelle appliquée


#tracé de fx et mu
fig_fx, ax_fx = plt.subplots(figsize=(8, 5))
ax_fx.plot(temps, historique_ft, 'r-', lw=1.5, label="Simulation Tamaas")
ax_fx.plot(temps, F_analytique_t, 'k--', lw=1.5, label="Théorie Persson")

#ajout de l'asymptote sur le graphique
ax_fx.axhline(y=ft_asymptote, color='b', linestyle='-.', label=f"Asymptote : {ft_asymptote:.2e}")

#ajout des infos de force appliquée et de l'erreur
texte_info = (f"Force appliquée (Load) : {force_totale_n:.2e} \n" f"Erreur Relative en régime permanent: {erreur_relative:.2f} %")

# On place la boîte de texte en haut à gauche (axes coords)
ax_fx.text(0.02, 0.95, texte_info, transform=ax_fx.transAxes, fontsize=10,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


ax_fx.set(xlabel="Temps", ylabel="Force de frottement Fx",title=f"Évolution du frottement (Pas = {pas}, N = {N})")
ax_fx.grid()
ax_fx.legend(loc='lower right')

#ajout de mu sur le deuxième axe
ax_mu = ax_fx.twinx()
ymin, ymax = ax_fx.get_ylim()
ax_mu.set_ylim(ymin / fn, ymax / fn)
ax_mu.set_ylabel("Coefficient de frottement $\mu$", color='red')

fig_fx.savefig(f"resultats/courbe_fx_total_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}.png")
plt.show()