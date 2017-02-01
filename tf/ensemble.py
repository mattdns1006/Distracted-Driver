import pandas as pd

if __name__ == "__main__":
    pd.set_option("display.line_width",200)
    pd.set_option("display.precision",3)
    pd.set_option("display.float_format",lambda x: '%.3f'%x)
    import pdb
    import glob
    submissions = glob.glob("submissions/*.csv")
    pdb.set_trace

    df = pd.read_csv(submissions[0])
    df.set_index(["img"],inplace=1)
    for submission in submissions[1:]:
        toAdd = pd.read_csv(submission)
        toAdd.set_index(["img"],inplace=1)
        df = df.add(toAdd)

        pass
    df /= len(submissions)
    print("Averaged submissions {0}".format(submissions))
    df.to_csv("ensemble.csv")


