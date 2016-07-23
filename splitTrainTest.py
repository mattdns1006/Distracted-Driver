import numpy as np
import cv2
import glob
import pandas as pd

if __name__ == "__main__":
    trainPath = "train/"
    np.random.seed(1)
    drivers = pd.read_csv("driver_imgs_list.csv")
    trainSubjects = drivers.subject.unique()[:-5]
    testSubjects = drivers.subject.unique()[-5:]
    train, test = [drivers.loc[drivers.subject.isin(driverSubjects)] for driverSubjects in [trainSubjects, testSubjects]]
    assert train.shape[0] + test.shape[0] == drivers.shape[0], "train and test cross validation sets do not add to total training set size"

    train.to_csv("trainCV.csv",index=0)
    test.to_csv("testCV.csv",index=0)

    print("Intersection of train and test",[i for i in train if i in test])
    print("Number of train/test samples (cross validtion test) = %d, %d" %(train.shape[0],test.shape[0]))


    def test():
        allTestFiles = []
        for f in glob.glob("test/*.jpg"):
            allTestFiles.append(f)
        testActualDf = pd.DataFrame(allTestFiles,columns=["img"])
        print("Number of test samples (actual test) = %d" %(testActualDf.size))
        testActualDf.to_csv("test.csv",index=0)




