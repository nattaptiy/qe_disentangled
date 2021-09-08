import torch
import csv
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import random
import pickle
import pandas as pd

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lang_emb_layer = nn.Linear(1024, 1024)
        self.meaning_emb_layer = nn.Linear(1024, 1024)
        self.lang_iden_layer = nn.Linear(1024, 8)

    def forward(self, x):
        lang_emb = self.lang_emb_layer(x)
        meaning_emb = self.meaning_emb_layer(x)
        lang_iden = self.lang_iden_layer(lang_emb)
        return lang_emb, meaning_emb, lang_iden

train_num=5075
dev_num=563

def embed(sentence):
    inputs=tokenizer(sentence,padding='max_length',truncation=True,return_tensors="pt",max_length=128)
    with torch.no_grad():
        outputs = embed_model(**inputs.to(device))
    sentence_embedding = torch.index_select(outputs[0],1,torch.tensor([0]).to(device)).squeeze()
    return sentence_embedding

def cal_loss(batch_num,train_data='train'):
    src_emb = torch.load('data_batch/src_'+train_data+str(batch_num)+'.pt')
    trg_emb = torch.load('data_batch/trg_'+train_data+str(batch_num)+'.pt')
    with open('data_batch/src_lang_'+train_data+str(batch_num)+'.txt','rb') as inputfile:
        src_lang_batch = pickle.load(inputfile)
    with open('data_batch/trg_lang_'+train_data+str(batch_num)+'.txt','rb') as inputfile:
        trg_lang_batch = pickle.load(inputfile)
    if train_data == 'train':
        rand_id=random.randint(0,train_num-1)
        if rand_id == batch_num:
            rand_id+=1
            rand_id=rand_id%train_num
    else:
        rand_id=random.randint(0,dev_num-1)
        if rand_id == batch_num:
            rand_id+=1
            rand_id=rand_id%dev_num
    rand_src_emb = torch.load('data_batch/src_' + train_data + str(rand_id) + '.pt')
    rand_trg_emb = torch.load('data_batch/trg_' + train_data + str(rand_id) + '.pt')
    with open('data_batch/src_lang_'+train_data+str(rand_id)+'.txt','rb') as inputfile:
        rand_src_lang_batch = pickle.load(inputfile)
    with open('data_batch/trg_lang_'+train_data+str(rand_id)+'.txt','rb') as inputfile:
        rand_trg_lang_batch = pickle.load(inputfile)
    rand_src_lang = torch.zeros(512, 1024)
    rand_trg_lang = torch.zeros(512, 1024)
    for ind1 in range(512):
        chk = False
        rand_num=random.randint(0,511)
        while rand_num<512:
            if rand_src_lang_batch[rand_num]==src_lang_batch[ind1]:
                chk=True
                rand_src_lang[ind1]=rand_src_emb[rand_num]
                break
            rand_num+=1
        if not chk:
            for ind2 in range(512):
                if rand_src_lang_batch[ind2]==src_lang_batch[ind1]:
                    chk=True
                    rand_src_lang[ind1] = rand_src_emb[ind2]
                    break
            if not chk:
                for ind2 in range(512):
                    if src_lang_batch[ind2] == src_lang_batch[ind1] and ind1 != ind2:
                        chk = True
                        rand_src_lang[ind1] = src_emb[ind2]
                        break
                if not chk:
                    rand_src_lang[ind1] = src_emb[ind1]
                    print("BAD")
    for ind1 in range(512):
        chk = False
        rand_num = random.randint(0, 511)
        while rand_num < 512:
            if rand_trg_lang_batch[rand_num] == trg_lang_batch[ind1]:
                chk = True
                rand_trg_lang[ind1] = rand_trg_emb[rand_num]
                break
            rand_num += 1
        if not chk:
            for ind2 in range(512):
                if rand_trg_lang_batch[ind2] == trg_lang_batch[ind1]:
                    chk = True
                    rand_trg_lang[ind1] = rand_trg_emb[ind2]
                    break
            if not chk:
                for ind2 in range(512):
                    if trg_lang_batch[ind2] == trg_lang_batch[ind1] and ind1 != ind2:
                        chk = True
                        rand_trg_lang[ind1] = trg_emb[ind2]
                        break
                if not chk:
                    rand_trg_lang[ind1] = trg_emb[ind1]
                    print("BAD")

    lang_emb_src, meaning_emb_src, lang_iden_src = model(src_emb)
    lang_emb_trg, meaning_emb_trg, lang_iden_trg = model(trg_emb)
    _, meaning_emb_rand_src, lang_iden_rand_src = model(rand_src_emb)
    _, meaning_emb_rand_trg, lang_iden_rand_trg = model(rand_trg_emb)
    lang_emb_rand_src, _, _ = model(rand_src_lang.to(device))
    lang_emb_rand_trg, _, _ = model(rand_trg_lang.to(device))
    y = torch.ones(len(src_lang_batch)).to(device)
    loss_meaning = cos_fn_m(meaning_emb_src, meaning_emb_trg, y) + cos_fn_m(
        meaning_emb_trg, meaning_emb_rand_trg, -y) + cos_fn_m(
        meaning_emb_src, meaning_emb_rand_src, -y)
    loss_recov = mse_fn(lang_emb_src + meaning_emb_src, src_emb)+mse_fn(lang_emb_trg + meaning_emb_trg, trg_emb)
    loss_lang_iden = cross_fn(lang_iden_src, torch.tensor(src_lang_batch).to(device)) + cross_fn(
        lang_iden_trg, torch.tensor(trg_lang_batch).to(device))
    loss_lang_emb = [cos_fn(lang_emb_src, lang_emb_rand_src, y),cos_fn(lang_emb_trg, lang_emb_rand_trg, y),cos_fn(
        lang_emb_src, lang_emb_trg, -y)]
    return loss_meaning + loss_recov + loss_lang_iden + loss_lang_emb[0] + loss_lang_emb[1],loss_meaning,loss_recov,loss_lang_iden,loss_lang_emb

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    embed_model = AutoModel.from_pretrained('xlm-roberta-large')

    embed_model.to(device)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    embed_model.eval()

    lang_dict = {
        "en": 0,
        "de": 1,
        "zh": 2,
        "ro": 3,
        "et": 4,
        "ne": 5,
        "si": 6,
        "ru": 7,
    }

    lang_pairs_src = ["en", "en", "ro", "et", "ne", "si","ru"]
    lang_pairs_trg = ["de", "zh", "en", "en", "en", "en","en"]
    lang_pairs = ["en-de", "en-zh", "ro-en", "et-en", "ne-en", "si-en","ru-en"]

    src_text = list()
    trg_text = list()
    label_txt = list()
    src_lang = list()
    trg_lang = list()

    maxlen=2000
    random.seed(9001)
    seed = 9001
    torch.manual_seed(seed)
    model = MLP()

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    mse_fn = nn.MSELoss()
    cos_fn = nn.CosineEmbeddingLoss()
    cos_fn_m = nn.CosineEmbeddingLoss()
    cross_fn = nn.CrossEntropyLoss()
    min_val_loss = np.Inf
    epochs_no_improve = 0
    batch_size = 512
    ex_count=0
    PATH='model/best_val.pt'
    tmp_path='model/'

    epochs = 1000
    result_columns = list()

    for epoch in range(epochs):
        src_batch = list()
        trg_batch = list()
        src_lang_batch = list()
        trg_lang_batch = list()
        train_loss=0
        train_m=0
        train_r=0
        train_li=0
        train_le=[0]*3
        print("Epoch" + str(epoch+1))
        for i in range(train_num):
            if i % 500 == 0:
                print("Batch" + str(i+1))
            optimizer.zero_grad()
            loss,loss_m,loss_r,loss_li,loss_le = cal_loss(i,train_data='train')
            train_loss+=loss.item()
            train_m+=loss_m.item()
            train_r+=loss_r.item()
            train_li+=loss_li.item()
            train_le[0]+=loss_le[0].item()
            train_le[1] += loss_le[1].item()
            train_le[2] += loss_le[2].item()
            loss.backward()
            optimizer.step()
        print("Validating")
        val_loss = 0
        for i in range(dev_num):
            if i % 100 == 0:
                print("Batch" + str(i + 1))
            loss, loss_m, loss_r, loss_li, loss_le = cal_loss(i, train_data='dev')
            val_loss += loss.item()
        print("Train Loss: " + str(float(train_loss / train_num)))
        print("Val Loss: " + str(float(val_loss / dev_num)))
        torch.save(model.state_dict(), tmp_path+'model_by_epoch'+str(epoch+1)+'.pt')
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH)
        else:
            epochs_no_improve += 1
        src_batch = list()
        trg_batch = list()
        src_lang_batch = list()
        trg_lang_batch = list()

        lang_loss = 0
        recov_loss = 0
        total_count = 0
        total_acc_correct = 0
        acc=list()
        label_score = list()
        result = list()
        pearlang = list()
        for lang in lang_pairs:
            src_text = list()
            trg_text = list()
            label_txt = list()
            src_lang_test = list()
            trg_lang_test = list()
            count = 0
            acc_correct = 0
            langres = list()
            with open("data/" + lang + ".test.tsv") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
                line_count = 0
                for row in csv_reader:
                    if line_count > 0:
                        src_text.append(row[0])
                        trg_text.append(row[1])
                        label_txt.append(row[2])
                        src_lang_test.append(lang_dict[lang[0:2]])
                        trg_lang_test.append(lang_dict[lang[3:5]])
                    line_count += 1
                label_lang=list()
                for txt in label_txt:
                    if len(txt) > 0:
                        label_score.append(float(txt))
                        label_lang.append(float(txt))
                for i, (src_tmp, trg_tmp, src_lang_tmp, trg_lang_tmp) in enumerate(
                        zip(src_text, trg_text, src_lang_test, trg_lang_test)):
                    if i % batch_size == 0:
                        if i > 0:
                            src_emb = embed(src_batch)
                            trg_emb = embed(trg_batch)
                            lang_emb_src, meaning_emb_src, lang_iden_src = model(src_emb)
                            lang_emb_trg, meaning_emb_trg, lang_iden_trg = model(trg_emb)
                            lang_loss += cross_fn(lang_iden_src, torch.tensor(src_lang_batch).to(device)) + cross_fn(
                                lang_iden_trg, torch.tensor(trg_lang_batch).to(device))
                            recov_loss += mse_fn(lang_emb_src + meaning_emb_src, src_emb) + mse_fn(
                                lang_emb_trg + meaning_emb_trg, trg_emb)
                            for meaning_src_tmp, meaning_trg_tmp,lang_iden_src_tmp,lang_iden_trg_tmp in zip(meaning_emb_src, meaning_emb_trg,lang_iden_src,lang_iden_trg):
                                count += 2
                                with torch.no_grad():
                                    result.append(1 - cosine(meaning_src_tmp.cpu(), meaning_trg_tmp.cpu()))
                                    langres.append(1 - cosine(meaning_src_tmp.cpu(), meaning_trg_tmp.cpu()))
                                    _, predicted = torch.max(lang_iden_src_tmp.unsqueeze(0), dim=1)
                                    if predicted == lang_dict[lang[0:2]]:
                                        acc_correct += 1
                                    _, predicted = torch.max(lang_iden_trg_tmp.unsqueeze(0), dim=1)
                                    if predicted == lang_dict[lang[3:5]]:
                                        acc_correct += 1
                        src_batch = list()
                        trg_batch = list()
                        src_lang_batch = list()
                        trg_lang_batch = list()
                        src_batch.append(src_tmp)
                        trg_batch.append(trg_tmp)
                        src_lang_batch.append(src_lang_tmp)
                        trg_lang_batch.append(trg_lang_tmp)
                    else:
                        src_batch.append(src_tmp)
                        trg_batch.append(trg_tmp)
                        src_lang_batch.append(src_lang_tmp)
                        trg_lang_batch.append(trg_lang_tmp)
                src_emb = embed(src_batch)
                trg_emb = embed(trg_batch)
                lang_emb_src, meaning_emb_src, lang_iden_src = model(src_emb)
                lang_emb_trg, meaning_emb_trg, lang_iden_trg = model(trg_emb)
                lang_loss += cross_fn(lang_iden_src, torch.tensor(src_lang_batch).to(device)) + cross_fn(
                    lang_iden_trg, torch.tensor(trg_lang_batch).to(device))
                recov_loss += mse_fn(lang_emb_src + meaning_emb_src, src_emb) + mse_fn(
                    lang_emb_trg + meaning_emb_trg, trg_emb)
                for meaning_src_tmp, meaning_trg_tmp,lang_iden_src_tmp,lang_iden_trg_tmp in zip(meaning_emb_src, meaning_emb_trg,lang_iden_src,lang_iden_trg):
                    count+=2
                    with torch.no_grad():
                        result.append(1 - cosine(meaning_src_tmp.cpu(), meaning_trg_tmp.cpu()))
                        langres.append(1 - cosine(meaning_src_tmp.cpu(), meaning_trg_tmp.cpu()))
                        _, predicted = torch.max(lang_iden_src_tmp.unsqueeze(0),dim=1)
                        if predicted==lang_dict[lang[0:2]]:
                            acc_correct+=1
                        _, predicted = torch.max(lang_iden_trg_tmp.unsqueeze(0),dim=1)
                        if predicted == lang_dict[lang[3:5]]:
                            acc_correct += 1
                acc.append(float(acc_correct/count))
                total_count+=count
                total_acc_correct+=acc_correct
                pp, _ = pearsonr(langres, label_lang)
                pearlang.append(pp)
        pearson_corr, _ = pearsonr(result, label_score)
        print("PEARSONR: "+str(pearson_corr))
        result_columns.append({
            "EPOCH": epoch,
            "PEARSONR": pearson_corr,
            "P_" + lang_pairs[0]: pearlang[0],
            "P_" + lang_pairs[1]: pearlang[1],
            "P_" + lang_pairs[2]: pearlang[2],
            "P_" + lang_pairs[3]: pearlang[3],
            "P_" + lang_pairs[4]: pearlang[4],
            "P_" + lang_pairs[5]: pearlang[5],
            "P_" + lang_pairs[6]: pearlang[6],
        })
        if epoch > 3 and epochs_no_improve >= 10:
            break
    df = pd.DataFrame(result_columns)
    df.to_csv("result/result.csv", sep="\t", quoting=csv.QUOTE_NONE, index=False)