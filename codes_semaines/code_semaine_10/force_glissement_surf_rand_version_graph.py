import tamaas as tm
import numpy as np
tm.initialize(8)
import matplotlib.pyplot as plt
import sys
import os

N=256
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
    load = 28 #valeur contact complet: environ 28
    hurst = 0.7
    v_cible= 0.07 #pour avoir la meme vitesse peu importe la valeur de N
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
    
if len(sys.argv) > 7:
    nom_doss = sys.argv[7]
else:
    nom_doss = "full_contact_rand_differents_temps"
os.makedirs(nom_doss, exist_ok=True)


L =1.
spectrum = tm.Isopowerlaw2D()
spectrum.q0 = 12
spectrum.q1 = 12
spectrum.q2 = 60
spectrum.hurst = hurst
generator = tm.SurfaceGeneratorFilter2D([N, N])
generator.spectrum = spectrum
generator.random_seed = 4
surface = generator.buildSurface() / spectrum.rmsSlopes()
h0=1 #ampltiude des bosses
surface *= h0 
#load=tm.Statistics2D.computeFullContactPressure(surface)
x = np.linspace(0, L, N, endpoint=False)
xx, yy = np.meshgrid(x, x, indexing='ij')
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
#load*=model.E_star*10/L

#on multiplie la force normale par la vraie raideur du materiau pour avoir les bonnes dimensions et on divise par L pour les bonnes dimensions
# car load est en metres , model E star en Pascals et L en metres


G_i = np.array([3.0])   # si on a k=0.1 , et Einf=1 on a dE=9 et E=3*G avec nu=0.5 donc G=dE/3=3
tau_i = np.array([0.1]) # taurelax= k*tau_fluage avec k=0.1 et tau_fluage =1 , taurelax=0.1
pas_temps = tau_i[0] / div_tau
if "erreur" in nom_doss:
    pas = int(pas * div_tau)
    
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
#pente_y, pente_x = np.gradient(surface, dx)


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

#on précalcule le déphasage de Fourier une seule fois
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
        #on calcule le déphasage total depuis la position initiale (t=0)
        #on utilise (i + 1) car c'est le décalage pour le prochain pas à résoudre
        dephasage_total = np.exp(-1j * qy * (i + 1) * dy_step)
    
        #décalage spectral en partant toujours du spectre de référence
        surface[:] = np.fft.irfft2(h_fft_init * dephasage_total, s=(N, N))
        pente_x[:] = np.fft.irfft2(pente_spectrale_init * dephasage_total, s=(N, N))
#%%
##### tracé des surfaces  #####
fig_def, ax1 = plt.subplots(figsize=(10, 5))
plt.axvline(x=(-pas/N )%1,ymin=0,ymax=1)
#ces 4 lignes servent a obtenir l'endroit avec la pression la plus élevée 
y_max = np.argmax(np.max(model.traction, axis=1)) #on prend l'indice de la pression la plus élevée parmi l'ensemble des pressions maximales de chaque ligne
u_tot_2d = model.displacement.copy()

h_cut = surface[y_max, :]  #on prend la ligne de la surface rugueuse qui correspond à cette pression
p_cut = model.traction[y_max, :]  #on prend le profil de pression de la ligne qui correspond a cette pression
u_cut = u_tot_2d[y_max, :]  #on prend le deplacement de la surface deformee qui correspond a cette pression
offset = np.max(h_cut - u_cut)


u_plot = u_cut


ax1.plot(x, h_cut , 'k', label='Solide rigide')
ax1.plot(x, u_plot, 'b-', label='Solide déformable')

ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)",title=f"Profil de contact (y={y_max}, Pas numéro {pas})")
ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)", title=f"Profil de contact (y={y_max}, Pas numéro {pas})")

#tracé de la pression
ax2 = ax1.twinx()
#ax2.fill_between(x, 0, p_cut,
                 
ax2.plot(x,p_cut, color='green', alpha=0.3, label='Pression')
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
    
    f.write(f"Ft = {ft}\nmu = {mu_final}\nAire_reelle_initiale = {historique_A_reel[0]}\nAire_reelle_finale = {A_reel}")
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

ft_parseval = np.sum(p_analytique * pente_analytique) * dS
print(f"Force  de frottement analytique  : {ft_parseval:.4e}")

#%%
#calcul de l'erreur relative entre tamaas et carbone-putignano
erreur_relative = abs(historique_ft[-1] - ft_carbone) / ft_carbone * 100
force_normale = load * L**2 #force normale réelle appliquée

#calcul de l'erreur relative entre tamaas et ft_analytique_ discrétisée
err_tp=abs(historique_ft[-1]-ft_parseval)/ft_parseval *100

#calcul de l'erreur relative entre ft_analytique discrétisée et carbone-putignano
err_cp=abs(ft_carbone-ft_parseval)/ft_parseval *100

print("erreur tamaas/parseval : ",err_tp,"erreur carbone_parseval : ",err_cp)

ratio_ft_fn=historique_ft[-1]/force_normale

#tracé du graph
fig_fx, ax_fx = plt.subplots(figsize=(8, 5))

ax_fx.plot(temps, historique_ft, 'r-', lw=1.5, label="Simulation numérique")
ax_fx.axhline(y=ft_parseval, color='g', linestyle=':', label="Analytique (régime permanent)")

texte_info = (f"Force normale : {force_normale:.2e} N\n" 
              f"Erreur numérique / analytique : {err_tp:.2f} %")

ax_fx.text(0.4, 0.55, texte_info, transform=ax_fx.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Noms des axes en gras
ax_fx.set_xlabel("Temps (s)", fontweight='bold')
ax_fx.set_ylabel("Force de frottement Ft (N)", fontweight='bold')

ax_fx.grid()
ax_fx.legend(loc='lower right')

ax_mu = ax_fx.twinx()
ymin, ymax = ax_fx.get_ylim()
ax_mu.set_ylim(ymin / fn, ymax / fn)
ax_mu.set_ylabel("$\mu$ = $F_t/F_N$", color='red', fontweight='bold')

#chiffres des axes en gras
for label in ax_fx.get_xticklabels() + ax_fx.get_yticklabels():
    label.set_fontweight('bold')
for label in ax_mu.get_yticklabels():
    label.set_fontweight('bold')

fig_fx.savefig(f"{nom_doss}/courbe_fx_total_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}.png")

#%%

#tracé force de frottement en fonction de la vitesse de glissement

#on crée un tableau de 60 vitesses
vitesses_theoriques = np.logspace(-3, 2, 60)
ft_theoriques = []

#on utilise la variable 'Green' qui contient la matrice pure (sauvegardée avant Carbone)
Green_original = Green.copy()

#boucle sur toutes les vitesses pour calculer le régime permanent théorique
for v_test in vitesses_theoriques:
    
    #fréquence d'excitation pour cette vitesse spécifique
    omega_test = -qy * v_test  
    
    #module complexe (avec k=0.1 et tau=1.0)
    M_qv_test = k + ((1 - k) / (1 - 1j * omega_test * tau))
    
    #matrice de Green modifiée pour cette vitesse UNIQUEMENT
    G_complexe_test = Green_original * M_qv_test
    G_complexe_test[0, 0] = 1.0 # Évite la division par zéro
    
    #calcul de la pression analytique dans l'espace de Fourier
    p_fft_test = h_fft / G_complexe_test
    p_fft_test[0, 0] = 0.0 # Annule la pression moyenne
    
    #retour dans l'espace réel
    p_analytique_test = np.fft.irfft2(p_fft_test, s=(N, N))
    
    #force de frottement asymptotique (Parseval)
    ft_test = np.sum(p_analytique_test * pente_analytique) * dS
    ft_theoriques.append(ft_test)

ft_theoriques = np.array(ft_theoriques)

#on récupère la valeur numérique finale de Tamaas
ft_numerique_final = historique_ft[-1]

#tracé de la courbe
fig_cloche, ax_cloche = plt.subplots(figsize=(8, 5))
ax_cloche.plot(vitesses_theoriques, ft_theoriques, 'b-', lw=2, label="Courbe analytique")

#on place le point rouge en utilisant la vraie valeur numérique Tamaas
ax_cloche.plot([v_cible], [ft_numerique_final], 'ro', markersize=8, 
               label=f"Numérique (v={v_cible} m/s, Ft={ft_numerique_final:.3f} N)")

ax_cloche.set_xscale('log')
ax_cloche.set_xlabel("Vitesse de glissement V (m/s)")
ax_cloche.set_ylabel("Force de frottement Ft (N)")
#ax_cloche.set_title("Évolution théorique du frottement en fonction de la vitesse (aléatoire)")
ax_cloche.grid(True, which="both")
ax_cloche.legend(loc='center right', fontsize='small')

#sauvegarde de l'image
fig_cloche.savefig(f"{nom_doss}/courbe_theorique_cloche_V_{suff_v_cible}.png")

if len(sys.argv) > 3:
    plt.close(fig_cloche)
else:
    plt.show()


#%%
#tracé des deux illustrations

# On définit un ratio de 1:1 pour les largeurs des deux subplots
fig_surf, (ax_2d, ax_1d) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

#graph 1
surf_plot = ax_2d.pcolormesh(xx, yy, surface, cmap='viridis', shading='auto')
ax_2d.set_xlabel("Position x (m)")
ax_2d.set_ylabel("Position y (m)")
ax_2d.set_aspect('equal')
fig_surf.colorbar(surf_plot, ax=ax_2d, label="Hauteur (m)")

indice_coupe_x = int(N * 0.55)
x_coupe = x[indice_coupe_x]
ax_2d.axvline(x=x_coupe, color='red', linestyle='--', linewidth=2, label="Ligne de coupe (axe y)")
ax_2d.legend(loc="upper right")

#graph 2 (profil)
profil_1d_y = surface[indice_coupe_x, :]  #coupe du solide rigide
u_1d_y = u_tot_2d[indice_coupe_x, :]      #coupe du solide déformable 

#tracé de la surface du solide rigide (en noir)
ax_1d.plot(x, profil_1d_y, 'k-', linewidth=1.5, label='solide rigide')

#tracé de la surface du solide déformable (en bleu)
ax_1d.plot(x, u_1d_y, 'b-', linewidth=1.5, label="solide déformable à l'état final")

#tracé de l'état initial (z=0)
ax_1d.axhline(0, color='gray', linestyle='--', label="solide déformable à l'état initial (z=0)")


#limites pour l'axe y
limite_basse = -0.02  
limite_haute = 0.03   
ax_1d.fill_between(x, u_1d_y, limite_haute, color='blue', alpha=0.15)
ax_1d.set_ylim(limite_basse, limite_haute)

#limites pour l'axe x
limite_gauche = 0.0  
limite_droite = 1.0  
ax_1d.set_xlim(limite_gauche, limite_droite)

ax_1d.set_xlabel("Position y (m) : axe du glissement")
ax_1d.set_ylabel("Hauteur (m)")
ax_1d.grid(True, linestyle='--', alpha=0.7)

# Ajout de la légende
ax_1d.legend(loc='lower left', fontsize=9)



#flèche de glissement
ax_1d.annotate('', xy=(0.55, 0.8), xytext=(0.85, 0.8),
               xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(facecolor='blue', edgecolor='blue', width=1.5, headwidth=6))
ax_1d.text(0.7, 0.85, 'Glissement du solide rigide', transform=ax_1d.transAxes,
           ha='center', color="blue", va='bottom', fontsize=10, fontweight='bold')

#flèche de la Force Normale
ax_1d.annotate('', xy=(0.25, 0.68), xytext=(0.25, 0.88),
               xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(facecolor='red', edgecolor='red', width=1.5, headwidth=6))
ax_1d.text(0.25, 0.9, r'Force Normale $F_N$', transform=ax_1d.transAxes,
           ha='center', color="red", va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

#sauvegarde
nom_image_surf = f"{nom_doss}/illustration_surf_rand_combinee.png"
fig_surf.savefig(nom_image_surf, bbox_inches='tight', dpi=300)
print(f"Image sauvegardée sous : {nom_image_surf}")

if len(sys.argv) > 3:
    plt.close(fig_surf)
else:
    plt.show()
    
