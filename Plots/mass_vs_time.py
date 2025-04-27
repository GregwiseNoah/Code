import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr


def load_mass_change(path):
    """Load cold mass and corresponding time, removing zero-mass entries."""
    data = np.loadtxt(path, comments="#")
    data = data[data[:, 0].argsort()]
    cold_mass = data[:, 13]
    cold_mass = cold_mass[cold_mass != 0]
    time = data[:len(cold_mass), 0]
    return cold_mass / np.mean(cold_mass[:128]), time


def extract_alpha_key(path):
    """Extract alpha value from path string."""
    key = path.split("/")[6].replace("alpha_", "")
    try:
        float(key)
        return key
    except ValueError:
        return None


#### Defining convolve function


def smooth(y, box_pts):
    
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# Load reference turbulent velocity from Turb.hst
ref_path = "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Turb/Turb.hst"
ref_data = np.loadtxt(ref_path, comments="#")
mass_ref = ref_data[:, 2]
KE_ref = ref_data[:, 6] + ref_data[:, 7] + ref_data[:, 8]
vturb = np.mean(np.sqrt(2 * KE_ref / mass_ref)[-200:])

# Load eddy turnover lengths
Ls_raw = np.loadtxt("/u/ageorge/athena_fork_turb_box/M0.5_simulation_data.csv", delimiter=",", skiprows=1)
Ls = np.repeat(Ls_raw[:-1, 2], 4)  # Each Ls used 4 times

Ls_highres = [Ls_raw[1, 2], Ls_raw[-3, 2]]

# Simulation paths
paths = [
    "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud_4/Turb.hst",
    "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud_3/Turb.hst",
    "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud_2/Turb.hst",
    "/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.001/Cloud/Turb.hst",
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud_4/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud_3/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud_2/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.01/Cloud/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud_4/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud_3/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud_2/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/alpha_0.1/Cloud/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud_4/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud_3/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud_2/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/1/Cloud/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud_4/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud_3/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud_2/Turb.hst',
    '/u/ageorge/athena_fork_turb_box/Turb_v2_init/Test_17/10/Cloud/Turb.hst'
]

paths_highres = [
    "/ptmp/mpa/ageorge/athena/Cloud_0.01/Turb.hst",
    "/ptmp/mpa/ageorge/athena/Cloud_1/Turb.hst"
]

# Group and normalize data
grouped_data = {}
grouped_highres = {}

for i, path in enumerate(paths):
    cold_mass, time = load_mass_change(path)

    #Smoothening the curve a bit
    cold_mass = smooth(cold_mass, 51)
    ############################
    
    t_eddy = Ls[i] / vturb
    normalized_time = time / t_eddy
    key = extract_alpha_key(path)
    if key:
        grouped_data.setdefault(key, []).append((normalized_time, cold_mass))

for i, path in enumerate(paths_highres):
    cold_mass, time = load_mass_change(path)

    #Smoothening the curve a bit
    cold_mass = smooth(cold_mass, 11)
    ############################
    
    t_eddy = Ls_highres[i] / vturb
    normalized_time = time / t_eddy
    # Manually assign keys to avoid extraction errors (e.g., folder "1" instead of "alpha_1")
    key = ["0.01", "1"][i]
    grouped_highres.setdefault(key, []).append((normalized_time, cold_mass))


# Plot

all_alpha_keys = sorted(grouped_data.keys(), key=lambda x: float(x))
colors = cmr.pride(np.linspace(0, 1, len(all_alpha_keys)))
color_map = {key: colors[i] for i, key in enumerate(all_alpha_keys)}
#colors = cmr.pride(np.linspace(0, 1, len(grouped_data)))

plt.figure(figsize=(5, 4))

# Plot low-res data (solid)
for group_key in all_alpha_keys:
    color = color_map[group_key]
    if group_key in grouped_data:
        for time, mass in grouped_data[group_key]:
            plt.plot(time[25:], mass[25:], color=color, alpha=0.8)
        plt.plot([], [], color=color, linestyle="-", label=group_key)






for group_key in all_alpha_keys:
    color = color_map[group_key]
    if group_key in grouped_highres:
        for time, mass in grouped_highres[group_key]:
            plt.plot(time, mass, color=color, linestyle=(0, (6, 2)), linewidth=1.2, alpha=0.8)

#########################
#### PLOTTING t_grow ####
#########################

# t_cold  = 0.02827667
# cs_cool = 0.02150
# lshatter = cs_cool * t_cold
# chi = 1000
# cs_hot = 0.6209301549496956
# M = vturb / cs_hot #Vturb already defined up
# t_grow = chi * np.sqrt(r_cl / (M * lshatter)) * (r_cl / L)**(-1/6 ) * t_cold


# Axis settings
plt.xlim(0, 4)
plt.ylim(1e-2, None)
plt.yscale("log")
plt.xlabel(r"$t / t_{\rm eddy}$", fontsize=16)
plt.ylabel(r"$m_{\text{cold}} / m_{\text{cold, initial}}$", fontsize=16)

#plt.legend(title="$t_{cool,mix} / t_{cc}$", fontsize=12, loc="upper left", bbox_to_anchor=(1.05, 1))

# Main legend (top right)
legend = plt.legend(title=r"$t_{\text{cool, mix}} / t_{\text{cc}}$", fontsize=9,
                     loc="lower left", frameon=True)
plt.gca().add_artist(legend)



plt.text(0.03, 0.93,
         r"─── : $128^3$ box" + "\n" + r"--- : $256^3$ box",
         transform=plt.gca().transAxes,
         fontsize=14,
         verticalalignment="top",
         horizontalalignment="left",
         family='monospace',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.text(-0.17, 0.96, "(b)", transform=plt.gca().transAxes,
         fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig("/u/ageorge/athena_fork_turb_box/plots/massvstime_small.png", bbox_inches="tight", dpi=900)
plt.savefig("/u/ageorge/athena_fork_turb_box/plots/massvstime_small.pdf", bbox_inches="tight")
plt.show()
