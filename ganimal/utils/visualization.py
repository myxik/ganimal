import matplotlib.pyplot as plt

def plot_images(images, num_rows, num_cols):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().detach().numpy().transpose(1, 2, 0))
        ax.axis('off')
    plt.tight_layout()
    return fig