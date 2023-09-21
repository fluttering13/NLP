# from visualizer import get_local
# get_local.activate()
import math
import random
import torch
import torch.nn as nn
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
Embedding=nn.Embedding(10,128)
'''
check gpu state
'''
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))

'''
0 is begin token
1 is end token
2 is padding token
'''
# batch_size=1
# <bos> 我 爱 吃 肉 和 菜 <eos> <pad> <pad>
src = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]])
# <bos> I like eat meat and vegetables <eos> <pad>
tgt = torch.LongTensor([[0, 3, 4, 5, 6, 7, 8, 1, 2]])
'''
d_model: encoding d
nhead: head number, default 8
num_encoder_layers: depth of encoder, default 6
num_decoder_layers: depth of decoder, default 6
dim_feedforward: ff neuron numbers, default 2048
drop_out: default 0.1
activation: default relu
layer_norm_eps: eps in Add&Norm layers, default 1e-5
batch_first: 
True: input=(batch_size,words_number,d)
False: input=(words_number,batch_size,d)
'''
transformer = nn.Transformer(d_model=128, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)
outputs=transformer(Embedding(src),Embedding(tgt))
#print(outputs.shape)
'''
Transformer forward parameters
src: input (embedding+positional encoding)
tgt: target (...)
src_mask(Tensor)=[words_number,words_number]
tgt_mask(Tensor)=[words_number,words_number]
memory_mask(Tensor)=[batch_size,words_number,words_number]
see nn.Transformer.generate_square_subsequent_mask
ex:
tensor([[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]])

key_padding_mask:
src_key_padding_mask(Tensor)=[batch_size,words_number]
memory_key_padding_mask(Tensor)=[batch_size,words_number]
src=[[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]]
src_key_padding_mask
[[0, 0, 0, 0, 0, 0, 0, 0, -inf, -inf]]
'''

'''
copy_task
'''


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize PE (max_len, d_model) (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # Initialize a tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # flatten pe
        pe = pe.unsqueeze(0)
        #one can save model with the parameter register_buffer
        self.register_buffer("pe", pe)
    def forward(self, x):
        # x+positional encoding and no need for the grad
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class CopyTaskModel(nn.Module):

    def __init__(self, d_model=128):
        super(CopyTaskModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128)

        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)

        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        # no need of softmax，because the CrossEntropyLoss is include
        self.predictor = nn.Linear(128, 10)

    def forward(self, src, tgt):
        # generate mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        tgt_mask = tgt_mask.cuda()
        src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src)
        tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt)

        # encode
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # +positional_encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        src=src.cuda()
        tgt=tgt.cuda()

        # transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用在key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        key_padding_mask=key_padding_mask.cuda()
        return key_padding_mask
max_length=16


model = CopyTaskModel()
#############test run##################
# src = torch.LongTensor([[0, 3, 4, 5, 6, 1, 2, 2]])
# tgt = torch.LongTensor([[3, 4, 5, 6, 1, 2, 2]])
# out = model(src, tgt)
# print('test run',out.shape)
####################################
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def generate_random_batch(batch_size, max_length):
    src=[]
    for i in range(batch_size):
        ###minus beginning and ending token
        random_len = random.randint(1, max_length - 2)
        random_ints = [0] + [random.randint(3, 9) for _ in range(random_len)] + [1]
        ### padding
        random_ints = random_ints + [2] * (max_length - random_len - 2)
        src.append(random_ints)
    src=torch.LongTensor(src)
    tgt = src[:, :-1]
    tgt_y = src[:, 1:]
    n_tokens = (tgt_y != 2).sum()
    return src, tgt, tgt_y, n_tokens   

batch_size=100
max_length=6
total_loss = 0
model=model.cuda()
for step in range(500):
    src, tgt, tgt_y, n_tokens=generate_random_batch(batch_size=batch_size, max_length=max_length)
    src=src.cuda()
    tgt=tgt.cuda()
    optimizer.zero_grad()
    out = model(src, tgt)
    out = model.predictor(out)
    '''
    out(batch_size, word_number, word_d)
    view to (batchsize*word_number,word_d)
    '''
    out=out.cpu()
    loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
    loss.backward()
    optimizer.step()
    total_loss += loss
    if step != 0 and step % 10 == 0:
        print("Step {}, total_loss: {}".format(step, total_loss))
        total_loss = 0
print('gpu allocated',torch.cuda.memory_allocated()/1000000)
print('gpu max allocated',torch.cuda.max_memory_allocated()/1000000)

model = model.eval()
test_src = torch.LongTensor([[0, 4, 3, 4, 6, 8, 9, 9, 8, 1, 2, 2]]).cuda()
test_tgt = torch.LongTensor([[0]]).cuda()

##greedy search
for i in range(max_length):
    # feed the model
    out = model(test_src, test_tgt)
    # output the linear layers
    predict = model.predictor(out[:, -1])
    # argmax
    y = torch.argmax(predict, dim=1)
    # concat the previous result
    test_tgt = torch.concat([test_tgt, y.unsqueeze(0)], dim=1)

    # if y==1 end token then end
    if y == 1:
        break
print(test_tgt)
###find the parameters
# for name, para in model.named_parameters():
#     print('{}: {}'.format(name, para.shape))

