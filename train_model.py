import sys
import numpy as np
import torch
import os

from torch.nn import Module, Linear, Embedding, CrossEntropyLoss 
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, inputs_filename, output_filename):
    self.inputs = np.load(inputs_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    self.embedding = Embedding(num_embeddings=word_types, embedding_dim=128)
    self.hidden = Linear(in_features=768, out_features=128)
    self.output = Linear(in_features=128, out_features=outputs)

  def forward(self, inputs):
    embedded = self.embedding(inputs)
    flattened = embedded.view(embedded.shape[0], -1)
    hidden = relu(self.hidden(flattened))
    logits = self.output(hidden)
    return logits


def train(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_function = CrossEntropyLoss(reduction='mean')
    LEARNING_RATE = 0.01
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)
    
    tr_loss = 0
    tr_steps = 0
    correct = 0
    total = 0
    
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(loader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        predictions = model(inputs)
        loss = loss_function(predictions, torch.argmax(targets, dim=1))
        tr_loss += loss.item()
        tr_steps += 1
        
        if idx % 1000 == 0:
            curr_avg_loss = tr_loss / tr_steps
            print(f"Current average loss: {curr_avg_loss}")
        
        # Compute training accuracy for this batch
        correct += (torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1)).sum().item()
        total += len(inputs)
        
        # Run the backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss = tr_loss / tr_steps
    acc = correct / total
    print(f"Training loss epoch: {epoch_loss}, Accuracy: {acc}")


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 32, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(10): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 
