import tamaas as tm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
N=150
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
spectrum = tm.Isopowerlaw2D()
spectrum.q0 = 20
spectrum.q1 = 20
spectrum.q2 = 80
spectrum.hurst = hurst
generator = tm.SurfaceGeneratorFilter2D([N, N])
generator.spectrum = spectrum
generator.random_seed = 4
surface = generator.buildSurface()
surface *= 1e-6 / tm.Statistics2D.computeRMSHeights(surface)

model = tm.Model(tm.model_type.basic_2d, [L, L], [N, N])
model.E= 1.
model.nu = 0.5
pas_temps=0.1
solver = tm.MaxwellViscoelastic(model, surface, 1e-10,
                                time_step=pas_temps,
                                shear_moduli=[0.12, 0.08, 0.05],
                                characteristic_times=[0.1, 1.0, 10.0])

#solveur
dx=L/N
dS = dx * dx
historique_ft = []  #pour enregistrer l'evolution de ft et la plot qu'a la fin
temps = []

#on calcule la pente initiale (gradient selon l'axe x)
#on ne s'interesse qu'a la pente selon x
_, pente_x = np.gradient(surface, dx)

#boucle
#on boucle exactement 'pas + 1' fois pour s'arrêter sur le pas demandé
for i in range(pas + 1):
    solver.solve(load)
    
    #on décale la surface seulement si on n'est pas au dernier pas
    # (pour que l'image finale corresponde bien à l'état après la résolution)
    ft = np.sum(model.traction * pente_x) * dS
    historique_ft.append(ft)
    temps.append(i * pas_temps) # pas * time_step
    if i < pas:
        surface[:] = np.roll(surface, shift=-1, axis=1)
        pente_x[:] = np.roll(pente_x, shift=-1, axis=1) #on déplace la pente avec la surface
#%%
#tracé
fig_def, ax1 = plt.subplots(figsize=(10, 5))
x = np.linspace(0, L, N, endpoint=False)
plt.axvline(x=pas/N,ymin=0,ymax=1)
#ces 4 lignes servent a obtenir l'endroit avec la pression la plus élevée 
y_max = np.argmax(np.max(model.traction, axis=1)) #on prend l'endroit i,j ou la pression est maximale
y_max=128  #ici on choisit n'importe quel endroit si on ne veut pas la pression maximale
h_cut = surface[y_max, :] #on prend la ligne de la surface rugueuse qui correspond à cette pression
u_cut = model.displacement[y_max, :] #on prend le deplacement de la surface deformee qui correspond a cette pression
p_cut = model.traction[y_max, :] #on prend le profil de pression de la ligne qui correspond a cette pression

#h_cut : profil de la surface rigide sur la ligne choisie
#u_cut : déplacement vertical de la surface déformée 



#on cherche le point de pression max sur la ligne du ymax
# et on force le solide déformable à toucher la surface exactement à cet endroit.
x_contact = np.argmax(p_cut)
offset = h_cut[x_contact] - u_cut[x_contact]
u_plot = (u_cut + offset) 


#tracé des surfaces
ax1.plot(x, h_cut , 'k', label='Solide rigide', lw=1.5)
ax1.plot(x, u_cut, 'b-', label='Solide déformable', lw=1.5)
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
    f.write(f"Ft = {ft}\nmu = {mu_final}")

#tracé de fx et mu
fig_fx, ax_fx = plt.subplots(figsize=(8, 4))
ax_fx.plot(temps, historique_ft, 'r-', lw=1.5)
ax_fx.set(xlabel="Temps (s)", ylabel="Force de frottement Fx (N)", title=f"Évolution du frottement (Nombre de pas totaux = {pas})")
ax_fx.grid()

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
