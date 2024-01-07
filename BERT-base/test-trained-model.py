#test a modele already train with dev.tsv data

from farm.infer import Inferencer
from farm.eval import Evaluator

data_path = "saved_models/mrpc/"
#function to create a dictionary from tsv dataset
def Dictionary(tsv):
    import re
    import csv
    d={}
    with open(tsv, 'r') as myfile: 
        with open(data_path + "valid.csv", 'w+') as csv_file:
            for line in myfile:
                fileContent = re.sub("\t", ",", line)
                csv_file.write(fileContent)

    with open(data_path + "valid.csv", mode="r") as inp:
        reader = csv.reader(inp)
        L = []
        for rows in reader:
            L.append({"text": rows[0]})
    return L


save_dir = "saved_models/glue_mrpc/"
data_set = "saved_models/mrpc/dev.tsv"
model = Inferencer.load(save_dir)
print(Dictionary(data_set))



result = model.run_inference(dicts=Dictionary(data_set))

for r in result:
    print(r)