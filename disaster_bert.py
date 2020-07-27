import numpy as np
import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import time
from torch.utils.tensorboard import SummaryWriter


class Disasters(torch.utils.data.Dataset):
    def __init__(self, df, test=False):
        super(Disasters).__init__()
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text = df['text']
        self.classes = 2
        self.test = test

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.text[index],
            add_special_tokens=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        if self.test:
            return {
                'num': torch.tensor(self.df.loc[index]['id'], dtype=torch.long),
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
            }
        else:
            return {
                'num': torch.tensor(self.df.loc[index]['id'], dtype=torch.long),
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'targets': torch.tensor(self.df.loc[index]['target'], dtype=torch.long)
            }

    def __len__(self):
        return len(self.df.index)

class DisasterNet(torch.nn.Module):
    def __init__(self):
        super(DisasterNet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.drop = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, ids, mask):
        _, x = self.bert(input_ids=ids, attention_mask=mask)
        #x = self.drop(x)
        return self.fc(x)

LEARNING_RATE = 3e-6
OPTIM = "adam"
WEIGHT_DECAY = 1e-2
MOMENTUM = 0.98
BATCH_SIZE = 16
NUM_WORKERS = 4
VALIDATION = False
VALIDATION_SUBMIT = False
LR_STEP = 1
GAMMA = 0.3

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#loss = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
#loss.to(device)

if OPTIM == "sgd":
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
elif OPTIM == "adam":
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=GAMMA)
writer = SummaryWriter()

def train(trainloader, num_epochs=10, auto_val=False):
    for epoch in range(num_epochs):
        run_loss = 0.0
        correct = 0.0
        total = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels, mask = data['ids'].to(
                device), data['targets'].to(device), data['mask'].to(device)
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
            loss_val = outputs[0]
            loss_val.backward()
            total += BATCH_SIZE
            optimizer.step()
            run_loss += loss_val.item()
            correct += (outputs[1].data.max(1).indices == labels).sum().item()
            if i % 16 == 0 and i != 0:
                writer.add_scalar(
                    'Loss/train', (run_loss)/(50*BATCH_SIZE), i*BATCH_SIZE + epoch*len(trainloader))
                writer.add_scalar(
                    'Accuracy/train', (correct)/(50*BATCH_SIZE), i*BATCH_SIZE + epoch*len(trainloader))

                if i*BATCH_SIZE < 1000:
                    j = ' '
                else:
                    j = ''
                if len("{}".format(round(run_loss,5))) < 8:
                    k = ' '*(8-len("{}".format(round(run_loss,5))))
                else:
                    k = ''
                print("Epoch: {}, Elem: {},{} Loss: {},{} Acc: {}".format(
                    epoch+1, i*BATCH_SIZE, j, round(run_loss,5), k, round(correct/total,5)))
                run_loss = 0.0
                correct = 0.0
                total = 0.0
        delta = time.time() - start_time
        print("Epoch: {}, Time: {}.".format(epoch+1,time.strftime("%H:%M:%S",time.gmtime(delta))))
        if auto_val:
            test(val_loader, epoch, submit_mode=False)
        scheduler.step()


def test(testloader, count=1, submit_mode=True):
    if submit_mode:
        csv = open("submission.csv", "w+")
        csv.write("id,target\n")
        with torch.no_grad():
            for data in testloader:
                nums, inputs, mask = data['num'], data['ids'].to(
                    device), data['mask'].to(device)
                outputs = model(inputs, mask)
                result = int(outputs[0].data.max(1).indices)
                csv.write("{},{}\n".format(nums.item(), result))
    else:
        correct = 0.0
        total = 0.0
        run_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                inputs, labels, mask = data['ids'].to(
                    device), data['targets'].to(device), data['mask'].to(device)
                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                total += BATCH_SIZE
                correct += (outputs[1].data.max(1).indices == labels).sum().item()
                run_loss += outputs[0]
            writer.add_scalar('Accuracy/test', correct/total, count)
            print("Test: {},  Loss: {}, Acc: {}".format(count+1, run_loss, correct/total))
            print()


train_file = open("../input/nlp-getting-started/train.csv")
train_df = pd.read_csv(train_file, header = 0)
if VALIDATION:
    train_data, val_data = torch.utils.data.random_split(Disasters(train_df), [int(.8*len(train_df)), len(train_df)-int(.8*len(train_df))])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    train(train_loader, num_epochs=5, auto_val=True)
    if VALIDATION_SUBMIT:
        test_file = open("../input/nlp-getting-started/test.csv")
        test_df = pd.read_csv(test_file, header = 0)
        test_loader = torch.utils.data.DataLoader(Disasters(test_df, test=True), batch_size = 1, num_workers = 1, pin_memory=True)
        test(test_loader)
else:
    train_loader = torch.utils.data.DataLoader(Disasters(train_df), batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    train(train_loader, num_epochs=2)

    test_file = open("../input/nlp-getting-started/test.csv")
    test_df = pd.read_csv(test_file, header = 0)
    test_loader = torch.utils.data.DataLoader(Disasters(test_df, test=True), batch_size = 1, num_workers = 1, pin_memory=True)
    test(test_loader)