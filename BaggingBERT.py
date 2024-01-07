# On importe les packages
from transformers import BertForSequenceClassification
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# On recupere les donnees du dataset
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# On definit la fonction pour tokenizer nos phrases
def tokenize_function(sentence):
    return tokenizer(sentence["sentence1"], sentence["sentence2"], truncation=True)

# On tokenise les donnees du dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# On nettoie les donnees tokeniser
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

# On telecharge tout les modeles entraines
model_0 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-0")
model_1 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-1")
model_2 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-2")
model_3 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-3")
model_4 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-4")
model_5 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-5")
model_6 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-6")
model_7 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-7")
model_8 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-8")
model_9 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-9")
model_10 = BertForSequenceClassification.from_pretrained("Glue-fine-tune/glue-fine-tune-10")

# On recupere seulement les donnees test
tokens = tokenized_datasets['test']

# On initialise les variables
vrai_positif = 0
faux_positif = 0
vrai_negatif = 0
faux_negatif = 0
accuracy = 0

# On regarde pour chaque donnees de test
for i in range(len(tokens)):
    # On regarde les resultats obtenus suivant les donnnees
    input_ids = torch.tensor(tokens[i]['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokens[i]['attention_mask']).unsqueeze(0)
    with torch.no_grad():
        outputs_0 = model_0(input_ids, attention_mask=attention_mask)
        outputs_1 = model_1(input_ids, attention_mask=attention_mask)
        outputs_2 = model_2(input_ids, attention_mask=attention_mask)
        outputs_3 = model_3(input_ids, attention_mask=attention_mask)
        outputs_4 = model_4(input_ids, attention_mask=attention_mask)
        outputs_5 = model_5(input_ids, attention_mask=attention_mask)
        outputs_6 = model_6(input_ids, attention_mask=attention_mask)
        outputs_7 = model_7(input_ids, attention_mask=attention_mask)
        outputs_8 = model_8(input_ids, attention_mask=attention_mask)
        outputs_9 = model_9(input_ids, attention_mask=attention_mask)
        outputs_10 = model_10(input_ids, attention_mask=attention_mask)

    # Obtenir les probabilites de chaque classe
    probabilities_0 = torch.nn.functional.softmax(outputs_0.logits, dim=-1)
    probabilities_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1)
    probabilities_2 = torch.nn.functional.softmax(outputs_2.logits, dim=-1)
    probabilities_3 = torch.nn.functional.softmax(outputs_3.logits, dim=-1)
    probabilities_4 = torch.nn.functional.softmax(outputs_4.logits, dim=-1)
    probabilities_5 = torch.nn.functional.softmax(outputs_5.logits, dim=-1)
    probabilities_6 = torch.nn.functional.softmax(outputs_6.logits, dim=-1)
    probabilities_7 = torch.nn.functional.softmax(outputs_7.logits, dim=-1)
    probabilities_8 = torch.nn.functional.softmax(outputs_8.logits, dim=-1)
    probabilities_9 = torch.nn.functional.softmax(outputs_9.logits, dim=-1)
    probabilities_10 = torch.nn.functional.softmax(outputs_10.logits, dim=-1)

    # Pour chaque modele on regarde si ses probabilites sont superieures a celle enregistrees, et on stocke la classe associee
    if(float(probabilities_0[0][0]) > float(probabilities_0[0][1])):
        classe = 0
    else:
        classe = 1
    # On stocke la probabilite
    proba = max(float(probabilities_0[0][0]), float(probabilities_0[0][1]))

    if(max(float(probabilities_1[0][0]), float(probabilities_1[0][1])) > proba):
        if(float(probabilities_1[0][0]) > float(probabilities_1[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_1[0][0]), float(probabilities_1[0][1]))
    
    if(max(float(probabilities_2[0][0]), float(probabilities_2[0][1])) > proba):
        if(float(probabilities_2[0][0]) > float(probabilities_2[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_2[0][0]), float(probabilities_2[0][1]))

    if(max(float(probabilities_3[0][0]), float(probabilities_3[0][1])) > proba):
        if(float(probabilities_3[0][0]) > float(probabilities_3[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_3[0][0]), float(probabilities_3[0][1]))

    if(max(float(probabilities_4[0][0]), float(probabilities_4[0][1]) > proba)):
        if(float(probabilities_4[0][0]) > float(probabilities_4[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_4[0][0]), float(probabilities_4[0][1]))

    if(max(float(probabilities_5[0][0]), float(probabilities_5[0][1])) > proba):
        if(float(probabilities_5[0][0]) > float(probabilities_5[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_5[0][0]), float(probabilities_5[0][1]))

    if(max(float(probabilities_6[0][0]), float(probabilities_6[0][1])) > proba):
        if(float(probabilities_6[0][0]) > float(probabilities_6[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_6[0][0]), float(probabilities_6[0][1]))

    if(max(float(probabilities_7[0][0]), float(probabilities_7[0][1])) > proba):
        if(float(probabilities_7[0][0]) > float(probabilities_7[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_7[0][0]), float(probabilities_7[0][1]))

    if(max(float(probabilities_8[0][0]), float(probabilities_8[0][1])) > proba):
        if(float(probabilities_8[0][0]) > float(probabilities_8[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_8[0][0]), float(probabilities_8[0][1]))

    if(max(float(probabilities_9[0][0]), float(probabilities_9[0][1])) > proba):
        if(float(probabilities_9[0][0]) > float(probabilities_9[0][1])):
            classe = 0
        else:
            classe = 1
        proba = max(float(probabilities_9[0][0]), float(probabilities_9[0][1]))

    if(max(float(probabilities_10[0][0]), float(probabilities_10[0][1])) > proba):
        if(float(probabilities_10[0][0]) > float(probabilities_10[0][1])):
            classe = 0
        else:
            classe = 1

    # On calcule les vrais positifs et les faux positifs
    if(classe == 1):
        if(int(tokens[i]["labels"]) == 1):
            vrai_positif += 1
            accuracy += 1
        else:
            faux_positif += 1
    # On calcule les vrais negatifs et les faux negatifs
    else:
        if(int(tokens[i]["labels"]) == 0):
            vrai_negatif += 1
            accuracy += 1
        else:
            faux_negatif += 1

# On calcule le recall et la precision pour les 2 classes
recall_positif = vrai_positif/(vrai_positif + faux_negatif)
precision_positif = vrai_positif/(vrai_positif + faux_positif)

recall_negatif = vrai_negatif/(vrai_negatif + faux_positif)
precision_negatif = vrai_negatif/(vrai_negatif + faux_negatif)

# On affiche les resultats
print("accuracy du modèle:", accuracy / len(tokens))
print("\nClasse 0 :")
print("precision :", precision_negatif)
print("F1_score :", 2 / (1/recall_negatif + 1/precision_negatif))
print("\nClasse 1 :")
print("precision :", precision_positif)
print("F1_score :", 2 / (1/recall_positif + 1/precision_positif))