import numpy as np
import cv2
import glob
import pandas as pd

if __name__ == "__main__":
    import pdb
    np.random.seed(1)
    drivers = pd.read_csv("../driver_imgs_list.csv")


    allCsv = pd.read_csv("train.csv")
    allCsv["img"] = allCsv.path.apply(lambda x: x.split("/")[-1])
    allInfo = pd.merge(allCsv,drivers,on="img")


    trainSubjects = allInfo.subject.unique()[:-5]
    testSubjects = allInfo.subject.unique()[-5:]
    trainCV, testCV = [allInfo.loc[allInfo.subject.isin(driverSubjects)] for driverSubjects in [trainSubjects, testSubjects]]

    toDrop = ["classname","subject","img"]
    trainCV.drop(toDrop,1).to_csv("trainCV.csv",index=0)
    testCV.drop(toDrop,1).to_csv("testCV.csv",index=0)

