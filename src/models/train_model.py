import torch
import hydra 
import numpy as np
import time
import random

from transformers import AdamW, BertConfig, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from src.data.make_dataset import create_loaders
from src.utils import format_time

from src.models import _SRC_PATH

@hydra.main(config_path=_SRC_PATH, config_name="config.yaml")
def main(cfg):
    seed_val = cfg.hyperparameters.seed_val # 42
    lr = cfg.hyperparameters.optimizer_adam_lr # 2e-5
    eps = cfg.hyperparameters.optimizer_adam_eps # 1e-8
    epochs = cfg.hyperparameters.epochs # 4
    num_warmup_steps = cfg.hyperparameters.learning_rate_warmups # 0
    loader_batch_size = cfg.hyperparameters.loader_batch_size # 32
    train_size_percentage = cfg.hyperparameters.train_size_percentage # 0.9
    sentence_max_length = cfg.hyperparameters.sentence_max_length # 64

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    train_dataloader, _ = create_loaders(loader_batch_size, train_size_percentage, sentence_max_length)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 5, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    optimizer = AdamW(model.parameters(),
                    lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = eps # args.adam_epsilon  - default is 1e-8.
                    )

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = num_warmup_steps, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)

            loss = result.loss
            logits = result.logits

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    torch.save(model, 'model.pt')

if __name__ == "__main__":
    main()
