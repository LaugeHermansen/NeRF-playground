from matplotlib import pyplot as plt
import numpy as np
import pdf2image
import os
import pickle
import pyperclip
from argparse import ArgumentParser

# A4 paper size in inches
HEIGHT = 11.7
WIDTH = 8.3


def get_first_nonzero(mask_1d):
    """
    get index of first nonzero element in a 1d boolean array
    """
    for i in range(len(mask_1d)):
        if mask_1d[i]: return i
    return -1

def get_bound(mask_2d, backward):
    """
    compute the boundary of a 2d boolean array
    boundary is defined as the median of the first nonzero element in each row, i.e. median (first_nonzero(row) for row in mask_2d)
    """
    first_ids = np.array([get_first_nonzero(mask_1d) for mask_1d in (mask_2d[:,::-1] if backward else mask_2d)])
    if backward: first_ids = mask_2d.shape[1] - first_ids
    return first_ids, np.median(first_ids)

def get_bounds(mask_2d):
    """
    get the left and right boundary of a 2d boolean array, bound defined as in get_bound
    """
    first_ids_1, median_1 = get_bound(mask_2d, backward=False)
    first_ids_2, median_2 = get_bound(mask_2d, backward=True)
    return (int(median_1), int(median_2))


def preprocess_pdf(pdf_path, dpi=300, return_images=False):
    """
    import the pdf file and preprocess it:
    convert to image, and compute the boundary of the used area
    
    """

    filename = os.path.basename(pdf_path)
    datafilename = f"{filename.split('.pdf')[0]}_dpi={dpi}.pkl"
    data_path = os.path.join(os.path.dirname(pdf_path), "margin_extractor_data")
    datafilepath = os.path.join(data_path, datafilename)

    #create the directory if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(datafilepath))

    # load the data if it exists
    if os.path.exists(datafilepath) and not return_images:
        with open(datafilepath, 'rb') as f:
            data = pickle.load(f)
        images = None
        image_shape = data['image_shape']
        bounds_hori = data['bounds_hori']
        bounds_vert = data['bounds_vert']
    
    # otherwise compute the data
    else:
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)#, size=(None, None), first_page=1, last_page=1, fmt='ppm', thread_count=1, userpw=None, use_cropbox=False, strict=False, transparent=False, single_file=False, output_file='01-NeRF.pdf', poppler_path=None, grayscale=False, paths_only=False)
        images = np.array([np.array(image.convert('L')) for image in images])
        images = 255 - images

        mask_hori = np.sum(images, axis=1) > 0
        mask_vert = np.sum(images, axis=2) > 0

        bounds_hori = np.array(get_bounds(mask_hori))
        bounds_vert = np.array(get_bounds(mask_vert))

        image_shape = images.shape[1:]

        with open(datafilepath, 'wb') as f:
            pickle.dump({'image_shape': image_shape, 'bounds_hori': bounds_hori, 'bounds_vert': bounds_vert}, f)

        if not return_images: images = None

    return images, image_shape, bounds_hori, bounds_vert

def plot_pdf_page(image, bounds_hori, bounds_vert):
    """
    plot a page with the computed boundaries - used for testing the get_bounds function
    """

    plt.imshow(255-image, cmap='gray')
    # plot the boundary of the image with a red rectangle
    plt.plot([bounds_hori[0], bounds_hori[0]], [bounds_vert[0], bounds_vert[1]], 'r')
    plt.plot([bounds_hori[1], bounds_hori[1]], [bounds_vert[0], bounds_vert[1]], 'r')
    plt.plot([bounds_hori[0], bounds_hori[1]], [bounds_vert[0], bounds_vert[0]], 'r')
    plt.plot([bounds_hori[0], bounds_hori[1]], [bounds_vert[1], bounds_vert[1]], 'r')
            
    plt.show()


def get_scale_params(image_shape, bounds_hori, bounds_vert, dpi, left, top, bottom, right):

    """
    compute the scaling and translation parameters for the latex includepdf command
    image_shape: shape of the images produced by preprocess_pdf
    bounds_hori: horizontal bounds of the used area (computed by preprocess_pdf)
    bounds_vert: vertical bounds of the used area (computed by preprocess_pdf)
    dpi: dpi used for the pdf conversion - should be the same as in preprocess_pdf
    left, top, bottom, right: margins of the page in inches
    
    """
    
    # get the size of the page
    page_width = image_shape[1]/dpi
    page_height = image_shape[0]/dpi

    # get the size of the used area
    print_width = (bounds_hori[1] - bounds_hori[0] + 1)/dpi
    print_height = (bounds_vert[1] - bounds_vert[0] + 1)/dpi

    # compute the scale parameters
    scale_x = (WIDTH - left - right)/print_width
    scale_y = (HEIGHT - top - bottom)/print_height
    scale = min(scale_x, scale_y)

    # compute the translation parameters
    bounds_hori_left = bounds_hori[0]/dpi
    bounds_hori_right = page_width - (bounds_hori[1]+1)/dpi
    trans_x = -(right - left)/2 + (bounds_hori_left - bounds_hori_right)*scale/2

    bounds_vert_top = bounds_vert[0]/dpi
    bounds_vert_bottom = page_height - (bounds_vert[1]+1)/dpi
    trans_y = -(bottom - top)/2 + (bounds_vert_top - bounds_vert_bottom)*scale/2

    return scale, trans_x, trans_y


def generate_latex_code(pdf_path, dpi, latex_path, left, top, bottom, right):
    """
    the main function to generate the latex code
    filename: path to the pdf file
    dpi: dpi used for the pdf conversion - determines precision of the scaling and translation parameters
    latex_path: path to the latex file - used to compute the relative path to the pdf file in the latex project
    left, top, bottom, right: margins of the page in inches

    """
    latex_path = latex_path if not latex_path is None else os.path.relpath(os.path.dirname(pdf_path))
    images, image_shape, bounds_hori, bounds_vert = preprocess_pdf(pdf_path, dpi=dpi)
    scale, trans_x, trans_y = get_scale_params(image_shape, bounds_hori, bounds_vert, dpi, left, top, bottom, right)
    return f"\\includepdf[pages=-,noautoscale=true, scale={scale:.3f}, offset={trans_x:.3f}in {trans_y:.3f}in]{{{os.path.join(latex_path,os.path.basename(pdf_path))}}}"



images, _, bounds_hori, bounds_vert = preprocess_pdf("Papers/03-InstantNGP.pdf", dpi=20, return_images=True)
plot_pdf_page(images[3], bounds_hori, bounds_vert)



# output = generate_latex_code("Papers/03-InstantNGP.pdf", 500, "", 0.3, 0.5, 0.5, 1.)
# output = generate_latex_code("Papers/03-InstantNGP.pdf", 500, "", 0,0,0,0)
# pyperclip.copy(output)

# if __name__ == "__main__":
#     parser = ArgumentParser(description='Generate latex code for pdfs')
#     parser.add_argument('pdf_path', type=str, help='path to the pdf file')
#     parser.add_argument('--dpi', type=int, default=100, help='dpi used for the pdf conversion - determines precision of the scaling and translation parameters')
#     parser.add_argument('--latex_path', type=str, default="", help='path to the latex file - used to compute the relative path to the pdf file in the latex project')
#     parser.add_argument('--left', type=float, default=0.6, help='left margin of the page in inches')
#     parser.add_argument('--top', type=float, default=0.2, help='top margin of the page in inches')
#     parser.add_argument('--bottom', type=float, default=0.2, help='bottom margin of the page in inches')
#     parser.add_argument('--right', type=float, default=2, help='right margin of the page in inches')
#     args = parser.parse_args()
#     print("Margin extractor was run with input:", args)
#     output = generate_latex_code(args.pdf_path, args.dpi, args.latex_path, args.left, args.top, args.bottom, args.right)
#     pyperclip.copy(output)
#     print("Output has been copied to clipboard:", output, sep="\n")
    