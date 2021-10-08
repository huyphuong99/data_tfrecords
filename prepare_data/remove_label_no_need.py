import pandas as pd


def remove_row(pd, path):
    label = pd["label"]
    data = pd["filename"]
    size_data = len(data)
    index = []
    for i in range(size_data):
        if label[i].isdigit():
            index.append(i)
    pd = pd.drop(labels=index, axis=0)
    pd.to_csv(path, index=False)

if __name__ =="__main__":
    data = "/media/huyphuong99/huyphuong99/tima/project/vr/info_vr/REFORMAT_DATA/NAME/name.csv"
    path_out =  "/media/huyphuong99/huyphuong99/tima/project/vr/info_vr/REFORMAT_DATA/NAME/name_1.csv"
    pd = pd.read_csv(data)
    remove_row(pd, path_out)

