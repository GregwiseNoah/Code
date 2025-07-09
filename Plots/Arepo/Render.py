import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import h5py
from tqdm.notebook import tqdm
import cmasher as cmr
import astropy.units as u
from scipy import spatial
from matplotlib.colors import LogNorm 


cmap =cmr.ember

def video_maker_density(files, savefilename):
    # Parameters
    Nplotx = 256 
    Nploty = 128
    Boxsize = 3978 
    r_shock = 0.01 * Boxsize
    
    Edges1dx = np.linspace(0.5 * Boxsize - 1.3 * r_shock,
                          0.5 * Boxsize + 1.4 * r_shock,
                          Nplotx + 1,
                          endpoint=True,
                          dtype=np.float64)
    Grid1dx = 0.5 * (Edges1dx[1:] + Edges1dx[:-1])
    
    Edges1dy = np.linspace(0.5 * Boxsize - 1.3 * r_shock,
                          0.5 * Boxsize + 1.4 * r_shock,
                          Nploty + 1,
                          endpoint=True,
                          dtype=np.float64)
    Grid1dy = 0.5 * (Edges1dy[1:] + Edges1dy[:-1])
    
    xx, yy = np.meshgrid(Grid1dx, Grid1dy)
    Grid2D = np.array([
        xx.reshape(Nploty * Nplotx),
        yy.reshape(Nploty * Nplotx),
        np.ones(Nplotx*Nploty) * 0.5 * Boxsize
    ]).T
    
    init_file = h5py.File(files[15], 'r')
    VoronoiPos = np.array(init_file['PartType0']['Coordinates'], dtype=np.float64)
    Density = np.array(init_file['PartType0']['Density'], dtype=np.float64) * 6.769898014440631e-31
    vmin, vmax = Density.min(), Density.max()
    dist, cells = spatial.KDTree(VoronoiPos[:]).query(Grid2D, k=1)
    
    tree = spatial.KDTree(VoronoiPos)
    init_file.close()
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(np.zeros((Nploty, Nplotx)),
                    extent=(Grid1dx[0], Grid1dx[-1], Grid1dy[0], Grid1dy[-1]),
                    origin='lower',
                    cmap=cmap,
                    norm=LogNorm(vmin=vmin, vmax=vmax) ) 
    ax.set_aspect( 'auto' )
    time_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha='center')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label('Density [g/cm³]')
    plt.tight_layout()
    
    
    
    def update_density(frame):
        file = h5py.File(files[frame], 'r')
        coords = file['PartType0']['Coordinates'][:]
        density = file['PartType0']['Density'][:] * 6.769898014440631e-31
        time = file['Header'].attrs['Time'] * time_conversion_factor.to(u.Myr)
        
        
        #print(density.min(), density.max())
    
        file.close()
    
        tree = spatial.KDTree(coords)  # REBUILD TREE EACH FRAME
        _, nearest_cells = tree.query(Grid2D, k=1)
        sampled_density = density[nearest_cells].reshape((Nploty, Nplotx))
    
        img.set_data(sampled_density)
        time_text.set_text(f"Frame {frame+1}/{len(files)}, Time: {time:.2f}")
        return img,
    
    ani = FuncAnimation(fig, update_density, frames=len(files))
    writer = FFMpegWriter(fps=5, bitrate=1800)
    ani.save(savefilename+".mp4", writer=writer)
    print("video saved")

