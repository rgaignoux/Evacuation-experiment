import numpy as np
import cv2

def blur_edge(img, d=31):
    h, w = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d, d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

if __name__ == '__main__':
    fn = "C:\\Users\\Robin\\Documents\\Stage2024\\Code\\images\\color_images\\motion_blur\\motion_blur929.png"

    img = cv2.imread(fn, 0)
    img = np.float32(img) / 255.0
    cv2.imshow('input', img)
    
    # Définir les valeurs par défaut pour les barres de défilement
    ang = 135
    d = 10
    SNR = 15

    win = 'deconvolution'

    IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    def update(_):
        ang = np.deg2rad(cv2.getTrackbarPos('angle', win))
        d = cv2.getTrackbarPos('d', win)
        noise = 10**(-0.1 * cv2.getTrackbarPos('SNR (db)', win))

        psf = motion_kernel(ang, d)
        cv2.imshow('psf', psf)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
        RES = cv2.mulSpectrums(IMG, iPSF, 0)
        res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        res = np.roll(res, -kh // 2, 0)
        res = np.roll(res, -kw // 2, 1)      

        cv2.imshow('deconvolved', res)
        cv2.imwrite("deblurred.png", res*255)

    cv2.namedWindow(win)
    # Création des trackbars
    cv2.createTrackbar('angle', win, ang, 180, update)
    cv2.createTrackbar('d', win, d, 50, update)
    cv2.createTrackbar('SNR (db)', win, SNR, 50, update)
    
    # Appel initial de la fonction update pour afficher les images
    update(None)

    while True:
        ch = cv2.waitKey()
        if ch == 27:  # Échapper pour quitter
            break
        if ch == ord(' '):  # Espaces pour mettre à jour l'image
            update(None)
