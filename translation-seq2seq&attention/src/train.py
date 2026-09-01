import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel

import sys


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)
        targets = targets.to(device)
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]

        # 混合精度前向传播
        with autocast(device_type='cuda'):
            encoder_outputs, context_vector = model.encoder(encoder_inputs)

            context_vector = model.decoder.proj(context_vector).unsqueeze(0)
            zero_tensor = torch.zeros(config.DECODER_NUM_LAYERS-1, context_vector.shape[1], config.HIDDEN_SIZE).to(device)
            decoder_hidden = torch.cat([context_vector, zero_tensor], dim=0)

            decoder_outputs = []
            seq_len = decoder_inputs.shape[1]
            for i in range(seq_len):
                decoder_input = decoder_inputs[:, i].unsqueeze(1)
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs.append(decoder_output)

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
            decoder_targets = decoder_targets.reshape(-1)
            loss = loss_fn(decoder_outputs, decoder_targets)

        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(resume_path=None):
    # 0. 设置随机种子（保证Embedding等初始化可复现）
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
                             en_tokenizer.pad_token_index).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. 混合精度梯度缩放器
    scaler = GradScaler()

    # 8. TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))
    
    # 恢复训练：从checkpoint加载
    start_epoch = 1
    best_loss = float('inf')
    if resume_path is not None and Path(resume_path).exists():
        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f'恢复训练成功：从 epoch {start_epoch} 继续，最佳 loss = {best_loss:.4f}')

    # 早停参数
    patience = 5          # 连续 N 轮不降就停
    early_stop_counter = 0
    best_loss = float('inf')

    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device)
        print(f'Loss: {loss:.4f}')

        # 记录到Tensorboard
        writer.add_scalar('Loss', loss, epoch)

        # 早停判断
        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, config.MODELS_DIR / 'best.pt')
            print(f'保存模型（best_loss={best_loss:.4f}）')
        else:
            early_stop_counter += 1
            print(f'Loss 未下降，连续 {early_stop_counter} 轮（patience={patience}）')
            if early_stop_counter >= patience:
                print(f'早停触发！{patience} 轮未改善，停止训练')
                break

    writer.close()


if __name__ == '__main__':
    # 支持无参数从头训练，或传 resume 参数恢复训练
    
    resume_path = sys.argv[1] if len(sys.argv) > 1 else None
    if resume_path is None:
        resume_path = config.MODELS_DIR / 'best.pt'
    train(resume_path=resume_path)