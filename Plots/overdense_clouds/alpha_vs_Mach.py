import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib.colors as mcolors

cs = 0.6209301549496956
t_cool_mix = 0.21648962
chi = 1000

def get_mass_ratio(path):
    data = np.loadtxt(path, comments="#")
    data = data[data[:, 0].argsort()]
    cold_mass = data[:, 13]
    cold_mass = cold_mass[cold_mass != 0]  
    return np.log10(np.mean(cold_mass[-50:]) / np.mean(cold_mass[0:10]))


hst_Test_15_path = "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_15/alpha_0.01/Turb/Turb.hst"
hst_Test_14_path = "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_14/alpha_0.1/Turb/Turb.hst"
hst_Test_16_path = "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_16/alpha_0.01/Turb/Turb.hst"
hst_Test_17_path = "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Turb/Turb.hst"

data_Test_15 = np.loadtxt(hst_Test_15_path, comments="#")
mass_Test_15 = data_Test_15[:, 2]
KE_Test_15 = data_Test_15[:, 6] + data_Test_15[:, 7] + data_Test_15[:, 8]
vturb_Test_15_array = np.sqrt(2 * KE_Test_15 / mass_Test_15)
vturb_Test_15 = np.mean(vturb_Test_15_array[-500:])
Mach_Test_15 = vturb_Test_15 / cs
r_cl_Test_15s = np.loadtxt("/u/ageorge/athena_fork_turb_box/M0.75_simulation_data.csv", delimiter=",", skiprows=1)[:-1,1]
t_cc_Test_15 = np.sqrt(chi) * r_cl_Test_15s / vturb_Test_15

mass_ratios_15 = [ get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_15/alpha_0.001/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_15/alpha_0.01/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_15/alpha_0.1/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_15/alpha_1/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_15/alpha_10/Cloud/Turb.hst")]

data_Test_14 = np.loadtxt(hst_Test_14_path, comments="#")
mass_Test_14 = data_Test_14[:, 2]
KE_Test_14 = data_Test_14[:, 6] + data_Test_14[:, 7] + data_Test_14[:, 8]
vturb_Test_14_array = np.sqrt(2 * KE_Test_14 / mass_Test_14)
vturb_Test_14 = np.mean(vturb_Test_14_array[-500:])
Mach_Test_14 = vturb_Test_14 / cs
r_cl_Test_14s = np.loadtxt("/u/ageorge/athena_fork_turb_box/M0.75_simulation_data.csv", delimiter=",", skiprows=1)[1:-2,1]
t_cc_Test_14 = np.sqrt(chi) * r_cl_Test_14s / vturb_Test_14

mass_ratios_14 = [ get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_14/alpha_0.01/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_14/alpha_0.1/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_14/alpha_1/Cloud/Turb.hst")]
                   #get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_14/alpha_10/Cloud/Turb.hst")]


data_Test_16 = np.loadtxt(hst_Test_16_path, comments="#")
mass_Test_16 = data_Test_16[:, 2]
KE_Test_16 = data_Test_16[:, 6] + data_Test_16[:, 7] + data_Test_16[:, 8]
vturb_Test_16_array = np.sqrt(2 * KE_Test_16 / mass_Test_16)
vturb_Test_16 = np.mean(vturb_Test_16_array[-500:])
Mach_Test_16 = vturb_Test_16 / cs
r_cl_Test_16s = np.loadtxt("/u/ageorge/athena_fork_turb_box/M0.25_simulation_data.csv", delimiter=",", skiprows=1)[:-1,1]
t_cc_Test_16 = np.sqrt(chi) * r_cl_Test_16s / vturb_Test_16

mass_ratios_16 = [ get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_16/alpha_0.001/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_16/alpha_0.01/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_16/alpha_0.1/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_16/alpha_1/Cloud/Turb.hst"),
                   get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_16/alpha_10/Cloud/Turb.hst")]


data_Test_17 = np.loadtxt(hst_Test_17_path, comments="#")
mass_Test_17 = data_Test_17[:, 2]
KE_Test_17 = data_Test_17[:, 6] + data_Test_17[:, 7] + data_Test_17[:, 8]
vturb_Test_17_array = np.sqrt(2 * KE_Test_17 / mass_Test_17)
vturb_Test_17 = np.mean(vturb_Test_17_array[-500:])
Mach_Test_17 = vturb_Test_17 / cs
r_cl_Test_17s_temp = np.loadtxt("/u/ageorge/athena_fork_turb_box/M0.5_simulation_data.csv", delimiter=",", skiprows=1)[:-1,1]  #Problem because there are 2 sims with the same r_cl values
r_cl_Test_17s = []
for i in r_cl_Test_17s_temp:
    r_cl_Test_17s.append(i + i * 0.25)
    r_cl_Test_17s.append(i)
    r_cl_Test_17s.append(i)
    r_cl_Test_17s.append(i - i * 0.25)
r_cl_Test_17s = np.array(r_cl_Test_17s)  

#Defining a different Mach number to get some offset
Mach_number_17 = []
for i in range(len(r_cl_Test_17s_temp)):
    Mach_number_17.append(vturb_Test_17/cs * 1 )
    Mach_number_17.append(vturb_Test_17/cs * 0.94)
    Mach_number_17.append(vturb_Test_17/cs * 1.06)
    Mach_number_17.append(vturb_Test_17/cs * 1 )

t_cc_Test_17 = np.sqrt(chi) * r_cl_Test_17s / vturb_Test_17

mass_ratios_17 = [      get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud_4/Turb.hst"),
                        get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud_3/Turb.hst"),
                        get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud_2/Turb.hst"),    
                        get_mass_ratio("/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud/Turb.hst"),    
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud_4/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud_3/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud_2/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud_4/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud_3/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud_2/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud_4/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud_3/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud_2/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud_4/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud_3/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud_2/Turb.hst'),
                        get_mass_ratio('/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud/Turb.hst')     ]
            



all_mass_ratios = mass_ratios_14 + mass_ratios_15 + mass_ratios_16 + mass_ratios_17
vmin, vmax = np.min(all_mass_ratios), np.max(all_mass_ratios)

plt.figure(figsize=(5, 4))
#plt.figure(figsize=(5, 6))
divnorm = mcolors.TwoSlopeNorm(vmin=-.5, vcenter=0, vmax=.5)

x_vals_100 = np.linspace(0.1, 0.9, 1000)  
y_vals_100 = -0.6 * x_vals_100

plt.fill_between(x_vals_100, y_vals_100 - 0.05, y_vals_100 + 0.05, color='gray', alpha=0.5, label = r"$\chi = 100$")
#### Maybe it needs a better label ?

x_vals_1000 = np.linspace(0.1, 0.9, 1000)  
y_vals_1000 = -0.6 * x_vals_1000 - 1  

plt.fill_between(x_vals_1000, y_vals_1000 - 0.05, y_vals_1000 + 0.05, color='black', alpha=0.5, label = r"$\chi = 1000$")

sc = plt.scatter(
    (np.concatenate([
        np.ones(len(r_cl_Test_15s)) * Mach_Test_15,
        np.ones(len(r_cl_Test_16s)) * Mach_Test_16,
        np.ones(len(r_cl_Test_14s)) * Mach_Test_14,
        (Mach_number_17 )
    ])),
    np.log10(np.concatenate([t_cool_mix / t_cc_Test_15, t_cool_mix / t_cc_Test_16, t_cool_mix / t_cc_Test_14, t_cool_mix / t_cc_Test_17])),
    c=np.concatenate([mass_ratios_15, mass_ratios_16, mass_ratios_14, mass_ratios_17]),
    cmap=cmr.prinsenvlag,  # Using a diverging colormap
    norm=divnorm,  # Ensure zero is white
)

cbar = plt.colorbar(sc, extend = 'both')
cbar.set_label(r"$ \log_{10}(m_{\rm final} / m_{\rm initial})_{\text{cold}}$", fontsize=16)

plt.text(-0.17, 0.96, "(c)", transform=plt.gca().transAxes, fontsize=16, fontweight="bold")

plt.xlabel(r"Mach Number ($\mathcal{M}$)", fontsize=16)
plt.ylabel(r"$\log(t_{\rm cool,mix} / t_{\rm cc})$", fontsize=16)
plt.xlim(left = 0.1, right = 0.9)
plt.legend(framealpha = 0.1, loc = "lower left")
#plt.yscale('log')

plt.tight_layout()
plt.savefig("/u/ageorge/athena_fork_turb_box/plots/alpha_vs_Mach.png" ,dpi = 900)
#plt.savefig("/u/ageorge/athena_fork_turb_box/plots/alpha_vs_Mach.pdf")
#plt.show()
