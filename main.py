import torch
from torch import nn
from model import Classifier
from transformers import AutoTokenizer, AutoModel
from dataset import TextDataset
from torch.utils.data import DataLoader
from torch import optim
import argparse

def collate_fn(batch):
    return batch

def fix_batching(batch):
    batch_text = []
    batch_label = []
    for i in range(len(batch)):
        batch_text.append(batch[i]['input'])
        batch_label.append(batch[i]['label'])
    return batch_text, torch.Tensor(batch_label)

class Trainer(nn.Module):
    def __init__(self, args, pred=False):
        super(Trainer, self).__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.get_model(self.args.features, self.args.model_name) # Creating Pre-Trained Model and Classifier
        if not pred:
            self.get_data(self.args.seed, self.args.batch_size) # Creating Dataloaders for Train and Test
            self.get_training_utils() # Creating Optimizer

        # Variables to track training
        self.loss = None
        self.acc = None 
        self.best_acc = None
        self.last_acc = None
        
    # Creating classifier and loading Pre-Trained Model
    def get_model(self, features=None, pretrained_model='sentence-transformers/paraphrase-MiniLM-L3-v2'):
        self.classifier = Classifier(features).to(self.device) 
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embedder = AutoModel.from_pretrained(pretrained_model).to(self.device) 

    # Creating Dataloaders and Datasets
    def get_data(self, seed, batch_size):
        self.trainloader = DataLoader(TextDataset(train=True, seed=seed), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.testloader = DataLoader(TextDataset(train=False, seed=seed), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Creating Optimizer
    def get_training_utils(self):
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.args.lr, amsgrad=True)

    # Encoding text batch using Pre-Trained Model
    def encode(self, x):
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        encoded_input = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        for key in list(encoded_input.keys()):
            encoded_input[key] = encoded_input[key].to(device)
        with torch.no_grad():
            embeddings = self.embedder(**encoded_input)
        return mean_pooling(embeddings, encoded_input['attention_mask'])

    # Forward pass to get sentiment probability
    def forward(self, x):
        sentence_embeddings = self.encode(x)
        return self.classifier(sentence_embeddings)

    # Training loop for one epoch using complete trainset
    def train_epoch(self, epoch):
        self.classifier.train()
        for batch_idx, batch in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            batch_text, batch_label = fix_batching(batch)
            output = self(batch_text)
            loss = nn.BCELoss()(output.squeeze(1), batch_label.to(self.device))
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(batch), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
        return loss

    # Testing loop for one epoch using complete testset
    def test(self, epoch):
        self.classifier.eval()
        correct = 0
        for _, batch in enumerate(self.testloader):
            batch_text, batch_label = fix_batching(batch)
            output = self(batch_text)
            pred = torch.round(output)
            correct += pred.eq(batch_label.cuda().data.view_as(pred)).long().cpu().sum()

        test_acc = 100. * correct / len(self.testloader.dataset)
        print("Test Accuracy at epoch {} is {}%".format(epoch, test_acc))
        return test_acc

    # Complete Training Loop
    def train(self):
        self.loss = []
        self.acc = []
        self.best_acc = 0
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(epoch)
            self.loss.append(loss)
            if epoch % self.args.test_interval == 0:
                acc = self.test(epoch)
                self.acc.append(acc)
                if acc > self.best_acc:
                    self.save('best')
                    self.best_acc = acc
            
        self.last_acc = self.test('INTMAX')
        self.save('last')

    # Saving Model State
    def save(self, star):
        save_dict = {
            'model':self.classifier.state_dict(), \
            'loss':self.loss, \
            'acc':self.acc, \
            'best_acc': self.best_acc, \
            'last_acc':self.last_acc, \
            'embedder': self.args.model_name
            }
        torch.save(save_dict, "{}_{}.pt".format(self.args.save, star))
        print("{} model saved".format(star))
    
    # Loading Model State
    def load(self, path):
        self.classifier.load_state_dict(torch.load(path, map_location=torch.device(self.device))['model'])

def main():
    parser = argparse.ArgumentParser(description='Null')
    parser.add_argument('--model_name', '-m', default='sentence-transformers/paraphrase-MiniLM-L3-v2', type=str)
    parser.add_argument('--epochs', '-e', default=100, type=int)
    parser.add_argument('--lr', '-l', default=0.01, type=float)
    parser.add_argument('--batch_size', '-b', default=8, type=int)
    parser.add_argument('--features', '-f', default=384, type=int)
    parser.add_argument('--seed', '-r', default=421, type=int)
    parser.add_argument('--log_interval', '-q', default=5, type=int)
    parser.add_argument('--test_interval', '-t', default=1, type=int)
    parser.add_argument('--save', '-s', default='./', type=str)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
