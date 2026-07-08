import numpy as np
import matplotlib.pyplot as plt
import athena_read
import cmasher as cmr  
import matplotlib as mt
from tqdm.notebook import tqdm

# t_eddy from input file
t_eddy_01 = 27.3
t_eddy_1  = 0.273

#Normalising density with rho_floor * r_cl
p_floor = 0.001
rcl_01 = 0.21254448
rcl_1 = 0.00212544

def process_file(filepath, label):
    data = athena_read.athdf(filepath)
    projection = np.sum(data["rho"], axis=2) * 1/256 #spatially averaged
    #Normalising factor of rho_floor * rcl
    normalising_factor = p_floor#*(rcl_01 if label == "Cloud_0.01" else rcl_1)
    norm_proj = projection / normalising_factor                                  
    time = data["Time"]
    return norm_proj, time

selected_files = {
    "Cloud_0.01": [
        "/ptmp/mpa/ageorge/athena/Cloud_0.01/Turb.out2.00002.athdf",
        "/ptmp/mpa/ageorge/athena/Cloud_0.01/Turb.out2.00032.athdf",
        "/ptmp/mpa/ageorge/athena/Cloud_0.01/Turb.out2.00067.athdf"
    ],
    "Cloud_1": [
        "/ptmp/mpa/ageorge/athena/Cloud_1/Turb.out2.00002.athdf",
        "/ptmp/mpa/ageorge/athena/Cloud_1/Turb.out2.00032.athdf",
        "/ptmp/mpa/ageorge/athena/Cloud_1/Turb.out2.00067.athdf"
    ]
}
print("starting to process")
data = {label: [process_file(f, label) for f in files] for label, files in selected_files.items()}
print("data processed")
vmin = min(np.min(proj) for projections in data.values() for proj, _ in projections)
vmax = max(np.max(proj) for projections in data.values() for proj, _ in projections)

#### Manual layout ####
fig = plt.figure(figsize=(9, 6))
ax = []

left0 = 0.06
bottom0 = 0.15 # slight raise to give titles space
width = 0.27
height = 0.4
edge_correction = 1.05

bottom = bottom0
ax += [[
    fig.add_axes([left0 + i * width * edge_correction, bottom, width, height])
    for i in range(3)
]]

bottom += height + 0.02
ax += [[
    fig.add_axes([left0 + i * width * edge_correction, bottom, width, height])
    for i in range(3)
]]

ax = np.array(ax)

labels = list(data.keys())
for row in range(2):  # 2 rows
    for col in range(3):  # 3 columns
        label = labels[row]
        projection, time = data[label][col]
        # normalise_factor = (p_floor*rcl_01)  if label == "Cloud_0.01" else (p_floor*rcl_1)
        # print(normalise_factor / p_floor)
        
        log_proj = np.log10(projection) # / normalise_factor )
        axis = ax[row, col]
        img = axis.imshow(log_proj, cmap='viridis', origin='lower',
                          vmin=np.log10(vmin), vmax=np.log10(vmax))
        axis.axis('off')
        t_eddy = t_eddy_01 if label == "Cloud_0.01" else t_eddy_1
        axis.text(0.05, 0.9, rf"$t/t_{{eddy}}$ = {(time / t_eddy):.1f}", color="white", fontsize=12,
                  transform=axis.transAxes, bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))


# ax[0, 1].set_title("Cloud Growth", fontsize=16, fontweight="bold", pad=5)
# ax[1, 1].set_title("Cloud Death", fontsize=16, fontweight="bold", pad=5)

fig.text(0.05, bottom0 + height/2, "Cloud Growth", va='center', ha='center', rotation='vertical',
         fontsize=16, fontweight="bold")
fig.text(0.05, bottom + height/ 2, "Cloud Death", va='center', ha='center', rotation='vertical',
         fontsize=16, fontweight="bold")

ax[0, 0].text(0.05, 0.05, r"$t_{\mathrm{cool,mix}} / t_{\mathrm{cc}} = 0.01$",
              transform=ax[0, 0].transAxes, color="white", fontsize=12,
              bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

ax[1, 0].text(0.05, 0.05, r"$t_{\mathrm{cool,mix}} / t_{\mathrm{cc}} = 1$",
              transform=ax[1, 0].transAxes, color="white", fontsize=12,
              bbox=dict(facecolor="black", alpha=0.3, edgecolor="none"))


plt.suptitle("(a)", fontsize=14, fontweight="bold", x=0.001, ha='left')

#color_range = [np.log10(vmin), np.log10(vmax)]
color_range  = [vmin, vmax]
cmap = cmr.bubblegum_r  # Or any other you like from cmasher

# Create custom colorbar
cax = fig.add_axes([
    left0 + 3 * width * edge_correction - 0.005,  # moved left
    bottom0,
    0.02,
    2*height + 0.02
])

cbar1 = plt.colorbar(
    mt.cm.ScalarMappable(
        norm=mt.colors.Normalize(vmin=color_range[0], vmax=color_range[1]),
        cmap='viridis',
    ),
    extend="both",
    cax=cax,
    shrink=0.9,
)
cbar1.set_label(r"$\int \rho\,dz\ /(\rho_{\text{hot}} L_{\text{box}})$", fontsize = 14, fontweight = 'bold')
cbar1.ax.tick_params(labelsize=9)

# Save
plt.savefig("/u/ageorge/athena_fork_turb_box/plots/projections.png", dpi=900)
plt.savefig("/u/ageorge/athena_fork_turb_box/plots/projections.pdf")
plt.show()