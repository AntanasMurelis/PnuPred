import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, LayerGradCam, visualization, LayerAttribution
import matplotlib as mpl


def visualize_attributions(model, input_image, target_label, fig=None, axis=None, sign='all'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert input image to a PyTorch tensor
    input_tensor = torch.tensor(input_image).float().to(device).unsqueeze(0).unsqueeze(0)

    # Create an IntegratedGradients object for the model
    ig = IntegratedGradients(model)

    # Compute the attributions using Integrated Gradients
    attributions = ig.attribute(input_tensor, target=target_label)

    # Convert the attributions tensor to a numpy array
    attributions_np = attributions.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
    input_image = input_image[:, :, np.newaxis]

    # Visualize the attributions
    fig, axis = visualization.visualize_image_attr(
        attributions_np,
        input_image,
        cmap='seismic',
        method="heat_map",
        sign=sign,
        show_colorbar=False,
        alpha_overlay=0.7,
        use_pyplot=False,
        plt_fig_axis=(fig, axis)
    )

    return fig, axis

def visualize_attributions_panel(model, images, labels, sign='all'):
    # Split the images and labels into positive and negative lists
    pos_images, pos_labels, neg_images, neg_labels = [], [], [], []
    for i in range(len(images)):
        if labels[i] == 1:
            pos_images.append(images[i])
            pos_labels.append(labels[i])
        else:
            neg_images.append(images[i])
            neg_labels.append(labels[i])

    # Create the panel of images
    num_pos_images = len(pos_images)
    num_neg_images = len(neg_images)
    fig, axs = plt.subplots(nrows=2, ncols=max(num_pos_images, num_neg_images), figsize=(6 * num_pos_images, 12))

    # Plot the positive images
    for i in range(num_pos_images):
        # Get the input image and target label
        image = pos_images[i]
        label = pos_labels[i]

        # Visualize the attributions
        fig, axs[0, i] = visualize_attributions(model, image, label, fig, axs[0, i], sign)

        axs[0, i].set_title(f"Image {i + 1}\nLabel: Healthy")
        axs[0, i].axis("off")

    # Plot the negative images
    for i in range(num_neg_images):
        # Get the input image and target label
        image = neg_images[i]
        label = neg_labels[i]

        # Visualize the attributions
        fig, axs[1, i] = visualize_attributions(model, image, label, fig, axs[1, i], sign)

        axs[1, i].set_title(f"Image {num_pos_images + i + 1}\nLabel: Infected")
        axs[1, i].axis("off")

    fig.tight_layout()

    # Set color bar range based on the sign parameter
    if sign == 'positive' or sign == 'negative':
        vmin, vmax = 0, 1
    else:
        vmin, vmax = -1, 1

    # Add color bar
    cax = fig.add_axes([0.25, -0.05, 0.5, 0.03])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='seismic'),
                        cax=cax,
                        orientation='horizontal',
                        label='Attribution')
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    plt.show()
    return fig, axs


def layer_attribution_visualisation(model, input_image, layer_name, target_label, fig=None, axis=None, show=False):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert the numpy array to a PIL image object
    target_image_pil = PIL.Image.fromarray(input_image)
    # Preprocess the image
    preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    input_tensor = preprocess(target_image_pil).unsqueeze(0).to(device)

    # Instantiate the LayerGradCam object
    layer_gc = LayerGradCam(model, dict(model.named_modules())[layer_name])
    # Compute the attributions using GradCAM
    attributions = layer_gc.attribute(input_tensor, target=target_label)

    # Convert the attributions tensor to a numpy array
    attributions_np = attributions.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
    input_image = input_image[:, :, np.newaxis]

    if attributions_np.shape[0] < 128:
        attributions_np = LayerAttribution.interpolate(attributions.squeeze(0).cpu().unsqueeze(0), input_image.shape[:2]).squeeze(0).permute(1,2,0).detach().numpy()

    # Visualize the attributions
    fig, axis = visualization.visualize_image_attr(
        original_image=input_image,
        attr=attributions_np,
        sign='all',
        cmap='seismic',
        method='blended_heat_map',
        show_colorbar=False,
        plt_fig_axis=(fig, axis),
        alpha_overlay=0.3,
        use_pyplot = show
    )

    return fig, axis

def visualize_attributions_layer_panel(model, images, labels, layer_names):
    pos_images, pos_labels, neg_images, neg_labels = [], [], [], []
    for i in range(len(images)):
        if labels[i] == 1:
            pos_images.append(images[i])
            pos_labels.append(labels[i])
        else:
            neg_images.append(images[i])
            neg_labels.append(labels[i])

    num_pos_images = len(pos_images)
    num_neg_images = len(neg_images)
    num_layers = len(layer_names)
    fig, axs = plt.subplots(nrows=num_layers * 2, ncols=max(num_pos_images, num_neg_images), figsize=(6 * num_pos_images-3, 4 * num_layers * 2 + 3))
    fig.tight_layout(pad=0)

    for j, layer_name in enumerate(layer_names):
        for i in range(num_pos_images):
            image = pos_images[i]
            label = pos_labels[i]

            fig, axs[j * 2, i] = layer_attribution_visualisation(model, image, layer_name, label, fig, axs[j * 2, i], show=False)

            axs[j * 2, i].set_title(f"Image {i + 1}\nLayer: {layer_name}\nLabel: Healthy")
            axs[j * 2, i].axis("off")

            if i == 0:
                axs[j * 2, i].text(-0.3, 0.5, layer_name, fontsize=12, ha='center', va='center', rotation='vertical', transform=axs[j * 2, i].transAxes)

        for i in range(num_neg_images):
            image = neg_images[i]
            label = neg_labels[i]

            fig, axs[j * 2 + 1, i] = layer_attribution_visualisation(model, image, layer_name, label, fig, axs[j * 2 + 1, i], show=False)

            axs[j * 2 + 1, i].set_title(f"Image {num_pos_images + i + 1}\nLayer: {layer_name}\nLabel: Infected")
            axs[j * 2 + 1, i].axis("off")

            if i == 0:
                axs[j * 2 + 1, i].text(-0.3, 0.5, layer_name, fontsize=12, ha='center', va='center', rotation='vertical', transform=axs[j * 2 + 1, i].transAxes)

    # Create a custom color map
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

    # Add a color bar on the right side of the plot
    cax = fig.add_axes([1.0, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Attribution", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()
    return fig, axs

