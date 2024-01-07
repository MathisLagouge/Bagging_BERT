Vous trouverez nos 2 dossiers, l'un contenant un modèle BERT base ainsi que le code associé et l'autre un code pour recréer un BaggingBERT ainsi que des modèles BERT bases déjà entrainés.
Les modèles entrainés sont malheureusement trop lourd pour pouvoir les push sur GitHub, nous ne pouvons donc pas vous les transmettre de cette façon.
Il serait possible de vous les fournir via un lien google drive si vous désirez les observer ou les tester.

Dans BERT-base:
 - test-trained-model.py pour tester un modèle déjà entrainé dans saved_model/trained_model 
 - BERT_base.py : entraînemement d'un modèle BERT base
 
Dans BaggingBERT:
 - BaggingBERT.py pour utiliser les modèles déjà entrainés en simulant un BaggingBERT
 - CreerBERT.py : entraînemement de plusieurs modèles BERT base
 
Vous trouverez aussi un requirements.txt dans le BERT-base, mais ayant du faire de nombreuses manipulations durant les séances de cours nous n'avons malheureusement pas tout repertorié dans ce dernier.
Vous pourrez surement être amener à faire certaines installations si nécessaire en exécutant notre code.
