import cv2
import glob, pdb
from tqdm import tqdm

if __name__ == "__main__":
    train = glob.glob("TIF/train/*.tif")
    test = glob.glob("TIF/test/*.tif")
    files = train + test
    for f in tqdm(files):
        img = cv2.imread(f,0)
        wp = f.replace(".tif",".jpg").replace("TIF/","")
        cv2.imwrite(wp,img)

    print("Fin")
