import torch
from matplotlib import image
from config import cfg
import cv2
import numpy as np
from PIL import Image

def post_process(batch):

    new_batch = torch.tensor(np.zeros(batch.shape))
    for i in range(len(batch)):
        img = batch[i]
        img = torch.argmax(img, dim=0).cpu().detach().numpy()*255
        img = np.uint8(img)
        img_eq = cv2.equalizeHist(img)

        # Appliquer un filtre Gaussien
        img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

        # Appliquer le filtre de Sobel pour détecter les contours
        sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

        # Binariser l'image pour détecter les contours
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Appliquer une fermeture morphologique pour connecter les éléments proches
        kernel = np.ones((30,30),np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Trouver les contours de l'image fermée
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remplir les contours avec du blanc
        filled_image = np.zeros((closing.shape[0], closing.shape[1]), dtype=np.uint8)
        cv2.fillPoly(filled_image, contours, 255)

        result = torch.nn.functional.one_hot(torch.tensor(filled_image/255).type(torch.int64), cfg.DATASET.NBR_CLASSE).float()
        result = torch.swapaxes(result, -1, 0)
        result = torch.swapaxes(result, -1, 1)
        new_batch[i] = result
    
    return new_batch

if __name__=='__main__':

    y_data = torch.tensor(image.imread('./../models/try_2/validation_visualisation/img_pred_val_33_sample_5.png'))
    y_data = torch.mean(y_data, dim=-1)
    y_data = y_data.type(torch.int64)

    y_data = torch.nn.functional.one_hot(y_data, cfg.DATASET.NBR_CLASSE).float()
    y_data = torch.swapaxes(y_data, -1, 0)
    y_data = torch.swapaxes(y_data, -1, 1)
    y_data = torch.unsqueeze(y_data,0)

    result = post_process(y_data)

    img = torch.squeeze(result, dim=0)
    img = torch.argmax(img, dim=0)
    img = torch.unsqueeze(img, dim=-1)
    img = torch.cat([img,img,img], dim=-1)

    img = img*255
    img = img.numpy()
    img = Image.fromarray(np.uint8(img))

    img.save(f'./img_postprocess.png')