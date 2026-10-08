import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CHARGEMENT
# ============================================================
image = plt.imread("lenna.png")

if image.dtype == np.float32 or image.dtype == np.float64:
    image = (image * 255).astype(np.uint8)

print("Taille :", image.shape, "| Type :", image.dtype)


# ============================================================
# FONCTIONS (les mêmes qu'avant)
# ============================================================
def symetrie(im):
    n, p, c = im.shape
    im_sym = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_sym[i, j] = im[i, p - 1 - j]
    return im_sym


def negatif(im):
    return 255 - im   # version vectorisée rapide


def niveaudegris(im):
    return np.round(
        0.2126 * im[:, :, 0] +
        0.7152 * im[:, :, 1] +
        0.0722 * im[:, :, 2]
    ).astype(np.uint8)


def image_rouge(im):
    r = np.zeros_like(im)
    r[:, :, 0] = im[:, :, 0]
    return r


def image_verte(im):
    v = np.zeros_like(im)
    v[:, :, 1] = im[:, :, 1]
    return v


def image_bleue(im):
    b = np.zeros_like(im)
    b[:, :, 2] = im[:, :, 2]
    return b


def histo(im):
    H = np.zeros(256)
    for i in range(len(im)):
        for j in range(len(im[0])):
            H[im[i, j]] += 1
    return H


def egaliser_histogramme(H, gray_image):
    hc = np.zeros(len(H))
    hc[0] = H[0]
    for i in range(1, len(H)):
        hc[i] = hc[i - 1] + H[i]
    return (hc / gray_image.size) * 255


def augmenter_contraste(gray_image):
    H = histo(gray_image)
    he = egaliser_histogramme(H, gray_image)
    table = np.round(he).astype(np.uint8)
    return table[gray_image]   # version vectorisée


# ============================================================
# CALCUL DE TOUTES LES IMAGES
# ============================================================
lena_gris = niveaudegris(image)
lena_eq   = augmenter_contraste(lena_gris)


# ============================================================
# AFFICHAGE : TOUT DANS UNE SEULE FENÊTRE (3 lignes x 3 colonnes)
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

axes[0, 0].imshow(image);              axes[0, 0].set_title("Originale")
axes[0, 1].imshow(symetrie(image));    axes[0, 1].set_title("Symétrie")
axes[0, 2].imshow(negatif(image));     axes[0, 2].set_title("Négatif")

axes[1, 0].imshow(lena_gris, cmap='gray');  axes[1, 0].set_title("Niveaux de gris")
axes[1, 1].imshow(image_rouge(image));      axes[1, 1].set_title("Canal rouge")
axes[1, 2].imshow(image_verte(image));      axes[1, 2].set_title("Canal vert")

axes[2, 0].imshow(image_bleue(image));      axes[2, 0].set_title("Canal bleu")
axes[2, 1].imshow(lena_eq, cmap='gray');    axes[2, 1].set_title("Contraste augmenté")
axes[2, 2].plot(np.arange(256), histo(lena_gris)); axes[2, 2].set_title("Histogramme")

for ax in axes.flat:
    if not ax.lines:
        ax.axis('off')

plt.tight_layout()
plt.show()