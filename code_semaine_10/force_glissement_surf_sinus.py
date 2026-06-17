import tamaas as tm
import numpy as np

import matplotlib.pyplot as plt
import sys
import os

N=200
if len(sys.argv) > 5:
    load = float(sys.argv[1])
    suff_load = sys.argv[1]
    v_cible = float(sys.argv[2])
    hurst = float(sys.argv[3])
    suff_hurst = sys.argv[3]
    pas = int(sys.argv[4])
    suff_pas = sys.argv[4]
    temps_attente = int(sys.argv[5])
    suff_temps_attente = sys.argv[5]
    suff_v_cible = f"{v_cible:.2f}"
    if len(sys.argv) > 6:
        div_tau = float(sys.argv[6])
        suff_div_tau = sys.argv[6]  #on garde le texte brut pour le nom du fichier
    else:
        div_tau = 30.0
        suff_div_tau = "30"
else: #si execution via spyder
    import datetime
    temps_attente = 0
    load = 1
    hurst = 0.7
    v_cible= 0.16 #pour avoir la meme vitesse peu importe la valeur de N
    div_tau = 50.0
    pas = int(10*div_tau)    #changer valeur pour décaler de x pas
    suff_div_tau = str(div_tau)
    suff_load = str(load)
    suff_hurst = str(hurst)
    suff_pas = str(pas)
    suff_v_cible = f"{v_cible:.2f}"
    suff_temps_attente = str(temps_attente)
    timestamp = datetime.datetime.now().strftime("%Hh%Mm%Ss")
    suff_load = f"{load}_spyder_{timestamp}"


nom_doss="full_contact_sin_differentes_vitesse_gradexact"
os.makedirs(nom_doss, exist_ok=True)


L =1.

#surface sinusoidale
x_tmp = np.linspace(0, L, N, endpoint=False)
y_tmp = np.linspace(0, L, N, endpoint=False)
xx, yy = np.meshgrid(x_tmp, y_tmp, indexing='ij')

h0=1 #ampltiude des bosses
# Surface simple 2D (4 bosses spatiales)
surface = h0 * np.sin(2 * np.pi * 4 * xx / L) * np.sin(2 * np.pi * 4 * yy / L)
load=tm.Statistics2D.computeFullContactPressure(surface)
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
load*=model.E_star*10/L 
#on multiplie la force normale par la vraie raideur du materiau pour avoir les bonnes dimensions et on divise par L pour les bonnes dimensions
# car load est en metres , model E star en Pascals et L en metres

G_i = np.array([3.0])   # si on a k=0.1 , et Einf=1 on a dE=9 et E=3*G avec nu=0.5 donc G=dE/3=3
tau_i = np.array([0.1]) # taurelax= k*tau_fluage avec k=0.1 et tau_fluage =1 , taurelax=0.1
pas_temps = tau_i[0] / div_tau
if len(sys.argv) > 6:
    pas = int(200 * div_tau)


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


h_fft_init = np.fft.rfft2(surface)

#l'axe de défilement dans la boucle est l'axe y (axis=1), on utilise donc qy
pente_spectrale_init = 1j * qy * h_fft_init 

#retour dans l'espace réel pour obtenir la grille des pentes
pente_x = np.fft.irfft2(pente_spectrale_init, s=(N, N))



for i in range(temps_attente):
    solver.solve(load)
    
#on calcule la distance exacte parcourue en un pas de temps 
#(- car on reculait sur l'axe y avec shift=-1)
dy_step = -v_cible * pas_temps

#on précalcule le déphasage de Fourier (le théorème du retard) une seule fois
phase_shift = np.exp(-1j * qy * dy_step)
#boucle
#on boucle exactement 'pas + 1' fois pour s'arrêter sur le pas demandé
for i in range(pas + 1):
    solver.solve(load)
    
    #on décale la surface seulement si on n'est pas au dernier pas
    # (pour que l'image finale corresponde bien à l'état après la résolution)
    ft = np.sum((model.traction) * pente_x) * dS
    A_reel = np.sum(model.traction > 0) * dS
    historique_ft.append(ft)
    historique_A_reel.append(A_reel)
    temps.append(i * pas_temps) # pas * time_step
    if i < pas:
        # Décalage spectral de la surface
        surf_fft = np.fft.rfft2(surface)
        surface[:] = np.fft.irfft2(surf_fft * phase_shift, s=(N, N))
        
        # Décalage spectral de la pente
        pente_fft = np.fft.rfft2(pente_x)
        pente_x[:] = np.fft.irfft2(pente_fft * phase_shift, s=(N, N))
#%%
##### tracé des surfaces  #####
fig_def, ax1 = plt.subplots(figsize=(10, 5))
plt.axvline(x=(-pas/N )%1,ymin=0,ymax=1)

y_max = np.argmax(np.max(model.traction, axis=1)) #on prend l'indice de la pression la plus élevée parmi l'ensemble des pressions maximales de chaque ligne

#y_max=128 #ici on choisit n'importe quel endroit si on ne veut pas la pression maximale
#h_cut : profil de la surface rigide sur la ligne choisie
#u_cut : déplacement vertical de la surface déformée 

u_tot_2d = model.displacement


u_cut_total = u_tot_2d[y_max, :] #on prend le deplacement de la surface deformee qui correspond a cette pression
h_cut = surface[y_max, :] #on prend la ligne de la surface rugueuse qui correspond à cette pression
p_cut = model.traction[y_max, :].copy() #on prend le profil de pression de la ligne qui correspond a cette pression
offset = np.max(h_cut - u_cut_total) #prend l'ecart le plus grand entre surf rigide et surf deformable


u_plot = u_cut_total





ax1.plot(x, h_cut , 'k', label='Solide rigide')
ax1.plot(x, u_plot, 'b-', label='Solide déformable')

ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)",title=f"Profil de contact (y={y_max}, Pas numéro {pas})")
ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)", title=f"Profil de contact (y={y_max}, Pas numéro {pas})")

#tracé de la pression
ax2 = ax1.twinx()
#ax2.fill_between(x, 0, p_cut, color='green', alpha=0.3, label='Pression')
                 
ax2.plot(x,p_cut.real, color='green', alpha=0.3, label='Pression')
ax2.set_ylabel("Pression", color='green')
ax1.grid()
fig_def.legend(loc='upper right')
fig_def.savefig(f"{nom_doss}/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}.png")


if len(sys.argv) > 3:
    plt.close(fig_def)
else:
    plt.show()
 

#calcul du coef de frottement
fn = load * L * L
mu_final = ft / fn

chemin_txt = f"{nom_doss}/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}.txt"
with open(chemin_txt, "w") as f:
    
    f.write(f"Ft = {ft}\nmu = {mu_final}\nAire_reelle_finale = {A_reel}")


#%%
####### méthode persson #######
V = v_cible   #vitesse de glissement (distance d'un pas / temps d'un pas)
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
    contribution = qx*q_norm * C_q_2D * E_perte_t  #q*cos(phi)= qnorm*(qx/qnorm)= qx
    integrale_q = np.sum(contribution)*2  #*2 car tamaas calcul le PSD pour la moitié des fréquences
    
    #on récupère l'aire réelle mesurée par Tamaas à cet instant précis
    A_reel_t = historique_A_reel[i]
    
    #on calcule le pourcentage de contact, entre 0 et 1
    P_q_t = A_reel_t / Surface_totale  #comme p_q_t dépend de q et de t on a décidé de pas l'inclure dans l'equation de f_t
    
    #on applique la formule de l'équation 18 (avec le 1/2 et cos(0)=1)
    f_t = 0.5* integrale_q * Surface_totale* P_q_t
    F_analytique_t.append(f_t)

F_analytique_t = np.array(F_analytique_t)


#%%

##### méthode carbone putignano #####
k = 0.1
tau = 1.0  
vit = v_cible 

#la rugosité recule sur l'axe y dans le np.roll (axis=1, shift=-1), donc la vitesse relative est -vit
omega_car = -qy * vit  

#terme devant le G
M_qv = k + ((1 - k)/(1 - 1j *omega_car *tau))

solver_stat = tm.PolonskyKeerRey(model, surface, 1e-9)
solver_stat.solve(load)

#on modifie le G 
Green = model.operators['westergaard_neumann']['influence'][:].copy()
model.operators['westergaard_neumann']['influence'][:] = Green * M_qv

#résolution avec le nouveau G
solver_stat.solve(load)

#calcul de la force asymptotique
ft_carbone = np.sum((model.traction) * pente_x) * dS
print(f"Force asymptotique (Carbone-Putignano) : {ft_carbone:.4e}")


#%%

#### calcul analytique pour contact complet ####
#on utilise la matrice de green
G_complexe = Green * M_qv
G_complexe[0, 0] = 1.0 #pour éviter la division par zéro en q=0

h_fft = np.fft.rfft2(surface) #tf de la surface

p_fft = h_fft / G_complexe # calcul de la pression analytique
p_fft[0, 0] = 0.0 #on annule la pression moyenne

pente_spectrale_fft = 1j * qy * h_fft

#retour dans l'espace réel
p_analytique = np.fft.irfft2(p_fft, s=(N, N))
pente_analytique = np.fft.irfft2(pente_spectrale_fft, s=(N, N))

ft_parseval = np.sum((load+p_analytique) * pente_analytique) * dS
print(f"Force  de frottement analytique  : {ft_parseval:.4e}")





#calcul de l'erreur relative entre tamaas et carbone-putignano
erreur_relative = abs(historique_ft[-1] - ft_carbone) / ft_carbone * 100
force_normale = load * L**2 #force normale réelle appliquée

#calcul de l'erreur relative entre tamaas et carbone-putignano
err_tp=abs(historique_ft[-1]-ft_parseval)/ft_parseval *100

#calcul de l'erreur relative entre parseval et carbone-putignano
err_cp=abs(ft_carbone-ft_parseval)/ft_parseval *100

print("erreur tamaas/parseval : ",err_tp,"erreur carbone_parseval : ",err_cp)

ratio_ft_fn=historique_ft[-1]/force_normale

#tracé de fx et mu
fig_fx, ax_fx = plt.subplots(figsize=(8, 5))
ax_fx.plot(temps, historique_ft, 'r-', lw=1.5, label="Simulation Tamaas")
ax_fx.plot(temps, F_analytique_t, 'k--', lw=1.5, label="Théorie Persson")

#ajout de l'asymptote sur le graphique
ax_fx.axhline(y=ft_carbone, color='b', linestyle='-.', label="Carbone-Putignano")
ax_fx.axhline(y=ft_parseval, color='g', linestyle=':', label="Parseval")

#ajout des infos de force normale et de l'erreur
texte_info = (f"Force normale (Load) : {force_normale:.2e} \n" f"Erreur Relative entre tamaas et carbone: {erreur_relative:.2f} % \n" f"Ft en régime permanent (tamaas) : {historique_ft[-1]:.2e}. \n" f"Erreur Tamaas/Parseval : {err_tp:.2e} % \n" f"Ft (Parseval) : {ft_parseval:.2e} \n"f"Ft(Carbone) : {ft_carbone:.2e}")

#on place la boîte de texte en haut à gauche (axes coords)
ax_fx.text(0.4, 0.55, texte_info, transform=ax_fx.transAxes, fontsize=10,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax_fx.set(xlabel="Temps", ylabel="Force de frottement Ft",title=f"Frottement (surface sinusoïdale) (Pas = {pas}, N = {N}, vit= {v_cible}, pas_temps = {pas_temps:.4f}, phase pré-charg = {temps_attente})")
ax_fx.grid()
ax_fx.legend(loc='lower right')

#ajout de mu sur le deuxième axe
ax_mu = ax_fx.twinx()
ymin, ymax = ax_fx.get_ylim()
ax_mu.set_ylim(ymin / fn, ymax / fn)
ax_mu.set_ylabel("Coefficient de frottement $\mu$", color='red')

fig_fx.savefig(f"{nom_doss}/courbe_fx_total_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}.png")

# Sauvegarde du pic de frottement pour le graphique de fluage
pic_frottement = np.max(historique_ft)
chemin_pic = f"{nom_doss}/pic_frottement_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}.txt"
with open(chemin_pic, "w") as f:
    f.write(f"{temps_attente}\t{pic_frottement}\n")

dossier_erreur = "resultat_erreur_tau_diff_sinus"
os.makedirs(dossier_erreur, exist_ok=True)
chemin_erreur = f"{dossier_erreur}/erreur_div_{suff_div_tau}.txt"
with open(chemin_erreur, "w") as f:
    f.write(f"{div_tau}\t{err_tp}\n")
    
if "snakemake" in sys.modules or len(sys.argv) > 5:
    # snakemake
    plt.close('all')
else:
    #spyder
    plt.show()