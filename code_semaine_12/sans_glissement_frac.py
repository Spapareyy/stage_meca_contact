import tamaas as tm
tm.initialize(8)
import numpy as np

import matplotlib.pyplot as plt
import sys
import os
import tamaas.utils as tmu
N=512
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
        div_temps = float(sys.argv[6])
        suff_div_temps = sys.argv[6]  #on garde le texte brut pour le nom du fichier
    else:
        div_temps = 5.0
        suff_div_temps = "5.0"
else: #si execution via spyder
    import datetime
    temps_attente = 0
    load = 1 #valeur contact complet: 60
    hurst = 0.7
    v_cible= 4 #pour avoir la meme vitesse peu importe la valeur de N
    div_temps = 1.0
    pas = int(200)    #changer valeur pour décaler de x pas
    suff_div_temps = str(div_temps)
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
    nom_doss = "full_contact_sin_tests_spyder"
    
if len(sys.argv) > 8:
    alpha = float(sys.argv[8])
    suff_alpha = sys.argv[8]
else:
    alpha = 0.5  #Spyder
    suff_alpha = "0.5"
os.makedirs(nom_doss, exist_ok=True)


L =1.

#surface sinusoidale
x_tmp = np.linspace(0, L, N, endpoint=False)
y_tmp = np.linspace(0, L, N, endpoint=False)
xx, yy = np.meshgrid(x_tmp, y_tmp, indexing='ij')

#surface simple 2D (4 bosses spatiales)
surface = np.sin(2 * np.pi * 4 * xx / L) * np.sin(2 * np.pi * 4 * yy / L)

#calcul numérique de la pente RMS réelle de la surface sinusoïdale
k = 2 * np.pi * 4 / L
rms_slope = k / np.sqrt(2)

h0 = 1 #c'est la pente RMS visée 

#on normalise la surface par sa propre pente, puis on applique h0
surface = (surface / rms_slope) * h0

#load=tm.Statistics2D.computeFullContactPressure(surface)
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
#load*=model.E_star*100/L 
#on multiplie la force normale par la vraie raideur du materiau pour avoir les bonnes dimensions et on divise par L pour les bonnes dimensions
# car load est en metres , model E star en Pascals et L en metres

#G_i = np.array([3.0])   # si on a k=0.1 , et Einf=1 on a dE=9 et E=3*G avec nu=0.5 donc G=dE/3=3
#tau_i = np.array([0.1]) # taurelax= k*tau_fluage avec k=0.1 et tau_fluage =1 , taurelax=0.1


##### matériau fractionnaire  #####
k = 0.1  #ratio entre la rigidité à long terme (E_inf) et instantanée (E_0)


#nombre de branches discrètes souhaitées
N_branches = 30 

#tau et g a utiliser avec le solveur
tau_i, G_i = tmu.fractional_zener(alpha, k, N_branches)

#print des valeurs générées
print(f"tau_i ({len(tau_i)} branches) : {tau_i}")
print(f"G_i : {G_i}")


if "erreur" in nom_doss:
    pas = int(pas *div_temps)

dx=L/N

#on adapte le pas de temps en fonction du temps final souhaité en faisant varier div_temps plus haut

#configuration des pas géométriques
n_steps = 100  # 100 pas entre le début et le temps d'attente 
t_start = 1e-6
t_end = 1000.0

#on calcule le nombre de décades
nb_decades = np.log10(t_end) - np.log10(t_start)

#résolution constante (25 points par décade)
n_steps = int(25 * nb_decades)

temps_points = np.geomspace(t_start, t_end, num=n_steps + 1)
#on ajoute +1 car on a besoin de N+1 points pour faire N différences (intervalles)


dS = dx * dx
historique_A_reel = []
historique_temps = []

#boucle
for i in range(n_steps):
    #calcul du pas de temps spécifique à chaque itération
    dt_dynamique = temps_points[i+1] - temps_points[i]
    
    #nouveau solveur associé au pas de temps de chaque itération
    solver = tm.MaxwellViscoelastic(
        model, 
        surface, 
        1e-9,
        time_step=dt_dynamique,
        shear_moduli=G_i,
        characteristic_times=tau_i
    )
    
    #
    state_vars = solver.state_fields
    
    #si ce n'est pas le tout premier pas, on rend la mémoire au solveur
    if i > 0:
        for field in state_vars:
            state_vars[field][:] = model[field]
            
    #solveur pour un pas
    solver.solve(load)
    
    #on garde la mémoire de l'étape précédente pour la nouvelle étape
    for field in state_vars:
        model[field] = state_vars[field]
        
    #enregistrement des résultats pour les graphs
    A_reel = np.sum(model.traction > 0) * dS
    historique_A_reel.append(A_reel)
    historique_temps.append(temps_points[i+1])

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
fig_def.savefig(f"{nom_doss}/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}_dt_{suff_div_temps}.png")


if len(sys.argv) > 3:
    plt.close(fig_def)
else:
    plt.show()
 

chemin_txt = f"{nom_doss}/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}_dt_{suff_div_temps}.txt"
with open(chemin_txt, "w") as f:
    
    f.write(f"Aire_reelle_initiale = {historique_A_reel[0]}\nAire_reelle_finale = {historique_A_reel[0]}\n")

#%%
fig_def.legend(loc='upper right')
    
fig_def.savefig(f"{nom_doss}/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}_alpha_{suff_alpha}_dt_{suff_div_temps}.png")
if len(sys.argv) > 3:
    plt.close(fig_def)
else:
    plt.show()

chemin_txt = f"{nom_doss}/deformee_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}_alpha_{suff_alpha}_dt_{suff_div_temps}.txt"
with open(chemin_txt, "w") as f:
    f.write(f"Aire_reelle_initiale = {historique_A_reel[0]}\nAire_reelle_finale = {historique_A_reel[-1]}\n")

#%% Tracé de l'aire
plt.figure(figsize=(8, 5))
plt.semilogx(historique_temps, historique_A_reel, 'b-', label=f"$\\nu=0.5$ (Alpha={alpha})")
plt.xlabel("Temps")
plt.ylabel("Aire de contact réelle $A(t)$")
plt.title(f"Évolution de l'aire selon le temps (Load={load})")
plt.grid()
plt.legend()

plt.savefig(f"{nom_doss}/fluage_aire_vs_temps_load_{load}_alpha_{suff_alpha}.png", bbox_inches='tight')

chemin_historique = f"{nom_doss}/historique_fluage_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}_V_{suff_v_cible}_ta_{suff_temps_attente}_alpha_{suff_alpha}_dt_{suff_div_temps}.txt"
np.savetxt(chemin_historique, np.column_stack([historique_temps, historique_A_reel]), header="Temps Aire_reelle")

if "snakemake" in sys.modules or len(sys.argv) > 5:
     # snakemake
    plt.close('all')
else:
    #spyder
    plt.show()