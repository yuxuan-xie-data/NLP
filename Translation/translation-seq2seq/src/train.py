import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]
        # 前向传播
        # 编码阶段
        context_vector = model.encoder(encoder_inputs)
        # context_vector.shape: [batch_size, hidden_size]

        # 解码阶段
        decoder_hidden = context_vector.unsqueeze(0).repeat(model.decoder_num_layers, 1, 1)
        # decoder_hidden.shape: [decoder_num_layers, batch_size, hidden_size]

        decoder_outputs = []

        seq_len = decoder_inputs.shape[1]
        for i in range(seq_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            decoder_outputs.append(decoder_output)

        # decoder_outputs：[tensor([batch_size,1,vocab_size])] -> [batch_size * seq_len, vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape: [batch_size ,seq_len, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_outputs.shape: [batch_size * seq_len, vocab_size]

        # decoder_targets：[batch_size, seq_len] -> [batch_size * seq_len]
        decoder_targets = decoder_targets.reshape(-1)

        loss = loss_fn(decoder_outputs, decoder_targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    set_seed(42)

    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    dataloader = get_dataloader()
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    # 4. 模型
    model = TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index, encoder_num_layers=config.ENCODER_NUM_LAYERS, 
                             decoder_num_layers=config.DECODER_NUM_LAYERS).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 7. 恢复 checkpoint
    start_epoch = 1
    best_loss = float('inf')
    checkpoint_path = config.MODELS_DIR / 'best.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            print(f'恢复训练: 从 epoch {checkpoint["epoch"]} 继续, 当前最佳 loss: {best_loss:.4f}')

    # 8. TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Loss: {loss:.4f}')

        # 记录到Tensorboard
        writer.add_scalar('Loss', loss, epoch)

        # 保存模型和优化器状态
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_loss': best_loss,
            }, config.MODELS_DIR / 'best.pt')
            print(f'保存最佳模型 (epoch {epoch}, loss: {best_loss:.4f})')

    writer.close()


if __name__ == '__main__':
    train()
