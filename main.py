import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CHARGEMENT DE L'IMAGE
# ============================================================
image = plt.imread("lenna.png")

# ⚠️ CORRECTION CRUCIALE : convertir float32 -> uint8
if image.dtype == np.float32 or image.dtype == np.float64:
    image = (image * 255).astype(np.uint8)

print("Taille de l'image :", image.shape)
print("Type des pixels :", image.dtype)   # ← DOIT afficher uint8

plt.imshow(image)
plt.title("Lena - image originale")
plt.show()          # ← tu fermes cette fenêtre → la suivante s'ouvre


# ============================================================
# QUESTION 1 : Symétrie verticale (miroir horizontal)
# ============================================================
def symetrie(im):
    n, p, c = im.shape
    im_sym = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_sym[i, j] = im[i, p - 1 - j]
    return im_sym


plt.imshow(symetrie(image))
plt.title("Lena - symétrie verticale")
plt.show()          # ← tu fermes → la suivante s'ouvre


# ============================================================
# QUESTION 2 : Négatif
# ============================================================
def negatif(im):
    n, p, c = im.shape
    im_neg = np.zeros_like(im)
    for i in range(n):
        for j in range(p):
            im_neg[i, j] = 255 - im[i, j]
    return im_neg


plt.imshow(negatif(image))
plt.title("Lena - négatif")
plt.show()


# ============================================================
# QUESTION 4 : Niveaux de gris
# ============================================================
def niveaudegris(im):
    n, p, c = im.shape
    im_lum = np.zeros((n, p), dtype=np.uint8)
    for i in range(n):
        for j in range(p):
            r = im[i, j, 0]
            g = im[i, j, 1]
            b = im[i, j, 2]
            im_lum[i, j] = np.round(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return im_lum


lena_gris = niveaudegris(image)

plt.imshow(lena_gris, cmap='gray')
plt.title("Lena - niveaux de gris")
plt.show()


# ============================================================
# QUESTIONS 5, 6, 7 : Canaux R, V, B
# ============================================================
def image_rouge(im):
    photo_rouge = np.zeros(im.shape, dtype=int)
    photo_rouge[:, :, 0] = im[:, :, 0]
    return photo_rouge


def image_verte(im):
    photo_verte = np.zeros(im.shape, dtype=int)
    photo_verte[:, :, 1] = im[:, :, 1]
    return photo_verte


def image_bleue(im):
    photo_bleue = np.zeros(im.shape, dtype=int)
    photo_bleue[:, :, 2] = im[:, :, 2]
    return photo_bleue


plt.imshow(image_rouge(image))
plt.title("Lena - canal rouge")
plt.show()

plt.imshow(image_verte(image))
plt.title("Lena - canal vert")
plt.show()

plt.imshow(image_bleue(image))
plt.title("Lena - canal bleu")
plt.show()


# ============================================================
# QUESTION 8 : Histogramme
# ============================================================
def histo(im):
    H = np.zeros(256)
    for i in range(len(im)):
        for j in range(len(im[0])):
            H[im[i, j]] += 1
    return H


plt.plot(np.arange(256), histo(lena_gris))
plt.title("Histogramme de Lena en niveaux de gris")
plt.xlabel("Niveau de gris")
plt.ylabel("Nombre de pixels")
plt.show()


# ============================================================
# QUESTION 9 : Égalisation de l'histogramme
# ============================================================
def egaliser_histogramme(H, gray_image):
    hc = np.zeros(len(H))
    hc[0] = H[0]
    for i in range(1, len(H)):
        hc[i] = hc[i - 1] + H[i]
    he = (hc / gray_image.size) * 255
    return he


# ============================================================
# QUESTION 10 : Augmentation du contraste
# ============================================================
def augmenter_contraste(gray_image):
    H = histo(gray_image)
    he = egaliser_histogramme(H, gray_image)
    table = np.round(he).astype(np.uint8)

    new_image = gray_image.copy()
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            new_image[i, j] = table[gray_image[i, j]]
    return new_image


# ============================================================
# RÉSULTATS FINAUX : avant / après égalisation
# ============================================================
lena_eq = augmenter_contraste(lena_gris)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(lena_gris, cmap='gray')
axes[0, 0].set_title("Lena originale (gris)")
axes[0, 0].axis('off')

axes[0, 1].imshow(lena_eq, cmap='gray')
axes[0, 1].set_title("Lena contraste augmenté")
axes[0, 1].axis('off')

axes[1, 0].plot(np.arange(256), histo(lena_gris))
axes[1, 0].set_title("Histogramme avant")

axes[1, 1].plot(np.arange(256), histo(lena_eq))
axes[1, 1].set_title("Histogramme après")

plt.tight_layout()
plt.show()