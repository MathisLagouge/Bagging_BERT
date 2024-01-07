#code for the training of a base BERT model for text classification

import torch
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import BertTokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds

#check directory
import os
os.chdir('../')
print("Current working directory is {}".format(os.getcwd()))

#set model parameters
data_dir = "saved_models/mrpc/"     #data from glue, "MRPC" for text classification
save_dir = "saved_models/glue_mrpc"
batch_size = 16
evaluate_every = 100
lang_model = "bert-base-uncased"
n_epochs = 3
max_seq_len = 64
embeds_dropout_prob = 0.5
learning_rate=2e-5
warmup_proportion=0.1

#set seed
set_all_seeds(seed=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_gpu = 0

#Create a tokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=False)

#Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset

#label list and metric
label_list = ["0", "1"]
metric = "f1_macro"

processor = TextClassificationProcessor(tokenizer=tokenizer,
                                        max_seq_len=max_seq_len,
                                        data_dir=data_dir,
                                        labels=label_list,
                                        metric=metric,
                                        source_field="label"
                                        )

#Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size)

#Create an AdaptiveModel
#which consists of a pretrained language model as a basis
language_model = Bert.load(lang_model)
#and a prediction head on top that is suited for our task => Text classification
prediction_head = TextClassificationHead(layer_dims=[768, 2])

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[prediction_head],
    embeds_dropout_prob=embeds_dropout_prob,
    lm_output_types=["per_sequence"],
    device=device)

#Create an optimizer
optimizer, warmup_linear = initialize_optimizer(
    model=model,
    learning_rate=learning_rate,
    warmup_proportion=warmup_proportion,
    n_batches=len(data_silo.loaders["train"]),
    n_epochs=n_epochs)

#Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
trainer = Trainer(
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=n_gpu,
    warmup_linear=warmup_linear,
    evaluate_every=evaluate_every,
    device=device)

#Let it grow
model = trainer.train(model)
model.save(save_dir)
processor.save(save_dir)

#save model and processor
model.save(save_dir)
processor.save(save_dir)