import tamaas as tm
import numpy as np
import matplotlib.pyplot as plt
N=100
L =1.
load=1e-5
pas=20
#surface sinusoidale
x_tmp = np.linspace(0, L, N, endpoint=False)
y_tmp = np.linspace(0, L, N, endpoint=False)
xx, yy = np.meshgrid(x_tmp, y_tmp, indexing='ij')

h0=1e-6 #ampltiude des bosses
# Surface simple 2D (4 bosses spatiales)
surface = h0 * np.sin(2 * np.pi * 4 * xx / L) * np.sin(2 * np.pi * 4 * yy / L)

x = np.linspace(0, L, N, endpoint=False)

model = tm.Model(tm.model_type.basic_2d, [L, L], [N, N])
model.E= 1.
nu=0.5
model.nu = nu


pas_temps=0.01
G_i = np.array([3.0])
tau_i = np.array([0.1])

solver = tm.MaxwellViscoelastic(model, surface, 1e-9,
                                time_step=pas_temps,
                                shear_moduli=G_i,
                                characteristic_times=tau_i)

dx=L/N

pente_y, pente_x = np.gradient(surface, dx)
#on boucle exactement 'pas + 1' fois pour s'arrêter sur le pas demandé
for i in range(pas + 1):
    solver.solve(load)

    if i < pas:
        surface[:] = np.roll(surface, shift=-1, axis=1)
        pente_x[:] = np.roll(pente_x, shift=-1, axis=1) #on décale la pente avec la surface
        
u_tot_2d = model.displacement

y_max = np.argmax(np.max(model.traction, axis=1))
u_cut_total = u_tot_2d[y_max, :] #on prend le deplacement de la surface deformee qui correspond a cette pression
h_cut = surface[y_max, :] #on prend la ligne de la surface rugueuse qui correspond à cette pression
p_cut = model.traction[y_max, :] #on prend le profil de pression de la ligne qui correspond a cette pression

u_plot = u_cut_total

fig_def, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(x, h_cut , 'k', label='Solide rigide')
ax1.plot(x, u_plot, 'b-', label='Solide déformable')
ax2 = ax1.twinx()
ax2.fill_between(x, 0, p_cut, color='green', alpha=0.3, label='Pression')
ax2.set_ylabel("Pression", color='green')
plt.show()