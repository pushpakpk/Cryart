import torch
from mt.dataset import get_toy_loaders
from mt.model import Transformer
from utils.metrics import compute_bleu
from utils.viz import plot_attention
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(model, src, src_vocab, tgt_vocab, max_len=10):
    model.eval()
    src = src.unsqueeze(0).to(device)
    src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    enc_out = model.pos_enc(model.src_embed(src))
    for layer in model.enc_layers:
        enc_out = layer(enc_out, src_mask)
    ys = torch.tensor([[tgt_vocab['<sos>']]], device=device)
    for i in range(max_len):
        tgt_mask = torch.tril(torch.ones((ys.size(1),ys.size(1)), device=device)).bool().unsqueeze(0)
        dec_out = model.pos_enc(model.tgt_embed(ys))
        for layer in model.dec_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        prob = model.fc_out(dec_out[:,-1,:])
        next_word = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word==tgt_vocab['<eos>']:
            break
    return ys[0].tolist()

def evaluate(save_dir='runs/mt/'):
    os.makedirs(save_dir, exist_ok=True)
    loader, src_vocab, tgt_vocab = get_toy_loaders(batch_size=1)
    model = Transformer(len(src_vocab), len(tgt_vocab)).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir,'transformer_best.pth')))

    references, hypotheses = [], []
    for src, tgt, _, _ in loader:
        src = src[0]
        tgt = tgt[0]
        pred = generate(model, src, src_vocab, tgt_vocab)
        references.append([tgt[1:].tolist()])  # ignore <sos>
        hypotheses.append(pred[1:])            # ignore <sos>
    bleu = compute_bleu(references, hypotheses)
    print(f"Corpus BLEU score: {bleu:.2f}")

if __name__ == "__main__":
    evaluate()
