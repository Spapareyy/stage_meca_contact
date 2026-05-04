import tamaas as tm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
N=250
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
    pas = 600    #changer valeur pour décaler de x pas
    suff_load = str(load)
    suff_hurst = str(hurst)
    suff_pas = str(pas)

os.makedirs("resultats", exist_ok=True)


L =1.
spectrum = tm.Isopowerlaw2D()
spectrum.q0 = 20
spectrum.q1 = 20
spectrum.q2 = 80
spectrum.hurst = hurst
generator = tm.SurfaceGeneratorFilter2D([N, N])
generator.spectrum = spectrum
generator.random_seed = 4
surface = generator.buildSurface()
h0=1e-6 #ampltiude des bosses
surface *= h0 / tm.Statistics2D.computeRMSHeights(surface)

x = np.linspace(0, L, N, endpoint=False)
profil =h0 *np.cos(2*np.pi*x/L)  #profil 1D du cos
surface = np.tile(profil, (N, 1))   #conversion en 2D
surface -= np.mean(surface)       #on centre autour de 0

model = tm.Model(tm.model_type.basic_2d, [L, L], [N, N])
model.E= 1.
nu=0.5
model.nu = nu
pas_temps=0.01

G_i = np.array([0.12, 0.08, 0.05])
tau_i = np.array([0.1, 1.0, 10.0])
solver = tm.MaxwellViscoelastic(model, surface, 1e-10,
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

pente_rms = np.sqrt(np.mean(pente_x**2 + pente_y**2))
variance_pentes = pente_rms**2

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
#tracé
fig_def, ax1 = plt.subplots(figsize=(10, 5))
plt.axvline(x=(pas/N )%1,ymin=0,ymax=1)
#ces 4 lignes servent a obtenir l'endroit avec la pression la plus élevée 
y_max = np.argmax(np.max(model.traction, axis=1)) #on prend la ligne ou la pression est maximale
y_max=128  #ici on choisit n'importe quel endroit si on ne veut pas la pression maximale
h_cut = surface[y_max, :] #on prend la ligne de la surface rugueuse qui correspond à cette pression
u_cut = model.displacement[y_max, :] #on prend le deplacement de la surface deformee qui correspond a cette pression
p_cut = model.traction[y_max, :] #on prend le profil de pression de la ligne qui correspond a cette pression

#h_cut : profil de la surface rigide sur la ligne choisie
#u_cut : déplacement vertical de la surface déformée 



#on cherche le point de pression max sur la ligne du ymax
# et on force le solide déformable à toucher la surface exactement à cet endroit.
x_contact = np.argmax(p_cut) # on trouve cette fois le i,j qui correspond a la pression max
offset = h_cut[x_contact] - u_cut[x_contact]
u_plot = (u_cut + offset) 


#tracé des surfaces
ax1.plot(x, h_cut , 'k', label='Solide rigide', lw=1.5)
ax1.plot(x, u_plot, 'b-', label='Solide déformable', lw=1.5)
ax1.set(xlabel="Position x (m)", ylabel="Hauteur (µm)",title=f"Profil de contact (y={y_max}, Pas numéro {pas})")

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



#partie analytique (persson)

q = 2 * np.pi / L         #vecteur d'onde des bosses
V = (L / N) / pas_temps   #vitesse de glissement (distance d'un pas / temps d'un pas)
omega = q * V             #fréquence d'excitation vue par le solide déformable (rad/s)


#calcul du module de perte (G'') (dissipation viscoélastique)
#formule mathématique d'un modèle de maxwell à plusieurs branches
G_second = np.sum(G_i * (omega * tau_i) / (1 + (omega * tau_i)**2))

#E* = 2*G / (1 - nu) pour la conversion de module
E_perte = 2 * G_second / (1 - nu)
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
    
    #on récupère l'aire réelle mesurée par Tamaas à cet instant précis
    A_reel_t = historique_A_reel[i]
    
    #on calcule le pourcentage de contact, entre 0 et 1
    P_q_t = A_reel_t / Surface_totale
    
    #on applique la formule de l'équation 19 (avec le 1/2 et cos(0)=1)
    f_t = 0.5 * P_q_t * E_perte_t * Surface_totale *(q * h0)**2  #q**2*C(q)  
    F_analytique_t.append(f_t)

F_analytique_t = np.array(F_analytique_t)





#tracé de fx et mu
fig_fx, ax_fx = plt.subplots(figsize=(8, 4))
ax_fx.plot(temps, historique_ft, 'r-', lw=1.5, label="Simulation Tamaas")

ax_fx.plot(temps, F_analytique_t, 'k--', lw=1.5, label="Théorie Persson (Dynamique)")

ax_fx.set(xlabel="Temps (s)", ylabel="Force de frottement Fx (N)",title=f"Évolution du frottement (Nombre de pas totaux = {pas})")
ax_fx.grid()
ax_fx.legend()

#on ajoute mu sur la même courbe car c'est juste un coef de fx
ax_mu = ax_fx.twinx()
ymin, ymax = ax_fx.get_ylim()
ax_mu.set_ylim(ymin / fn, ymax / fn)
ax_mu.set_ylabel("Coefficient de frottement $\mu$", color='red')

#enregistrement du fx(t) final avec le mu
fig_fx.savefig(f"resultats/courbe_fx_total_step_{suff_pas}_load_{suff_load}_H_{suff_hurst}.png")
if len(sys.argv) > 3:
    plt.close(fig_fx)
else: 
    plt.show()