import torch
from torch.nn import functional as F
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt


def plt_images_2(slices_list, z,title_list,row_list,vmax_list):
    
    
    """
    
    :param slices_list: Ex : [10,11,12,13,14,15]
    :param z: The list that contains all tensors. Ex : To plot (img+,img-) -> z = [img_pos,img_inv]
    Each image is expected to have shape (Depth, Height , Width) or (B,C,D,H,W)
    :return:
    """

    
    
    # minmin = torch.min(torch.min(z[0]), torch.min(z[-1]))
    # maxmax = torch.max(torch.max(z[0]), torch.max(z[-1]))
    slices = slices_list
    i = 0
    rows = len(z)
    cols = len(slices)
    f, ax = plt.subplots(rows, cols)
    f.set_figheight(18)
    f.set_figwidth(22)
    # f.suptitle(title,fontsize=18)
    # f.supylabel("fsdfsd")
    # f.set_ylabel("fdsfsd")
      
    for r in range(rows):
        for c in range(cols):
            if( len(z[r].shape) == 5):
                z[r] = z[r].squeeze(0).squeeze(0)
            if( len(z[r].shape) == 4):
                z[r] = z[r].squeeze(0)
            temp = z[r]
            # temp = z[r].permute(2,0,1)

            temp = temp.detach().cpu()
            
            slice_num = slices[c]
            temp = ax[r, c].imshow(temp[slice_num,],vmin=0,vmax =vmax_list[r],cmap='RdBu')
            ax[r, c].set_title("Slice = " + str(slice_num),fontsize=13,loc='center',color='m')
            ax[r,c].grid(True)
            
#             ax[r, c].tick_params(labelsize=10)
            # ax[r,:].set_ylabel("fsfdfsdfs")

#             ax[r, c].axis('off')

            cbar = f.colorbar(temp,ax=ax[r,c],ticklocation="bottom",shrink=0.5,orientation='vertical')
            cbar.ax.tick_params(labelsize=10)
#             f.colorbar(temp,ax=ax)
            # cbar.set_ticks(ticks=[minmin,maxmax], labels = [float("{:.f}".format(minmin)),float("{:.4f}".format(maxmax))])
        
#           i += 1
#         i = 0

    # Label rows and columns
    # for ax, ve in zip(axs[0], [0.1, 1, 10]):
    #     ax.set_title(f'{ve}', size=18)
    for ax, v_title,h_title  in zip(ax[:, 0], title_list,row_list):
#         print(v_title)
        ax.set_ylabel(v_title, size=16,color='blue')
        # ax.text(-25,5,h_title, size=9,verticalalignment='center',horizontalalignment='center',rotation=0)
#     plt.subplots_adjust(wspace=0, hspace=0.2)
    f.tight_layout()
    # cbar = f.colorbar(temp,ax=ax,ticklocation="bottom",shrink=0.35,orientation='vertical')
    # cbar.ax.tick_params(labelsize=10)
    # f.subplots_adjust(right=0.8)
    # cmap = plt.get_cmap=('hot')
    # plt.set_cmap(cmap)
    # plt.grid(True)
    plt.show()




def plot_3d_svf(slices_list, svf):
    """
    expected shape (1,3,*spatial)
    :param slices_list: Ex : [10,11,12,13,14,15]
    :param z: The list that contains all tensors. Ex : To plot (img+,img-) -> z = [img_pos,img_inv]
    :return:
    """
    facecolors = [cm.jet(x) for x in np.random.rand(3)]
    # minmin = torch.min(torch.min(z[0]), torch.min(z[-1]))
    # maxmax = torch.max(torch.max(z[0]), torch.max(z[-1]))
    slices = slices_list
    i = 0
    rows = 2
    cols = len(slices_list)
    f, ax = plt.subplots(1,cols)
    f.set_figheight(18)
    f.set_figwidth(32)
    f.suptitle("Stationary Velocity Field",fontsize=25,color='blue')

    svf = svf.squeeze(0).detach().cpu() #now shape = (3,*spatial)
    # svf = svf.permute(3,0,1,2) #5,76,116,116). 5 originally were -> (1,1,116,116,76,5)
    svf = svf.detach()
    for c in range(cols):
        slice_num = slices[c]
        dx = svf[1][slice_num]
        dy = svf[2][slice_num]
#         print(dx.shape,dy.shape)
        # print((dy - svf[2][slice_num+1]).abs().sum())
        # temp = ax[c].quiver(dx,dy, cmap='RdBu_r', edgecolor = 'red',facecolors=facecolors)
        # temp = ax[c].quiver(dx,dy, facecolors=facecolors)
        temp = ax[c].quiver(dx,dy, angles='xy', color ='g')
        ax[c].set_title("Slice = " + str(slice_num),fontsize=20,color='m')
        # ax[c].axis('off')
        # ax[c].grid(True)
        ax[c].invert_yaxis()
        ax[c].tick_params(labelsize=14)

        # cbar = f.colorbar(temp,ticklocation="bottom",shrink=0.25,ax=ax[c],extend='max')
        # cbar.ax.tick_params(labelsize=14)
        # cbar.set_ticks(ticks=[minmin,maxmax], labels = [float("{:.f}".format(minmin)),float("{:.4f}".format(maxmax))])
#     f.tight_layout()  
    
    plt.show()
    # plt.savefig('/its/home/pi58/projects/distortion_correction_1.0/svf.png')
    
    return f
