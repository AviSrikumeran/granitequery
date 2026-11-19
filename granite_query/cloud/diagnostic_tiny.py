"""CPU bigram LoRA used only by the unpaid diagnostic smoke path.

This is not Granite and it does not load Hugging Face weights. It follows the
same supervised-token contract as ``modal_valuation.execute``: frozen base,
LoRA on the q and v maps, one AdamW step per exposure update, and a loss equal
to the mean cross-entropy over that update's unmasked answer tokens.
"""
from __future__ import annotations

import math

import numpy as np


class TinyTokenizer:
    """Whitespace tokenizer with a single EOS id. Built from the smoke texts."""

    def __init__(self, texts):
        words = []
        for text in texts:
            words.extend(str(text).split())
        vocab = ['<pad>', '<eos>', '<unk>'] + sorted(set(words))
        self.id_to_token = vocab
        self.token_to_id = {token: index for index, token in enumerate(vocab)}
        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.vocab_size = len(vocab)
        self.eos_token = '<eos>'

    def encode(self, text):
        return [self.token_to_id.get(word, self.unk_id) for word in str(text).split()]

    def decode(self, ids):
        return ' '.join(self.id_to_token[int(token)] for token in ids
                        if int(token) not in (self.pad_id, self.eos_id))


class TinyBigramLoRA:
    """Next-token bigram with trainable q/v LoRA and a frozen base.

    Prediction of token t uses only token t-1, so the label span — not a
    hidden attention pattern — decides which positions receive loss. B starts
    at zero, so a freshly built adapter matches the frozen base.
    """

    def __init__(self, vocab, *, d=16, r=8, alpha=16, seed=0,
                 weight_decay=0.01, max_grad_norm=1.0):
        if r <= 0 or d <= 0 or alpha <= 0:
            raise ValueError('Tiny LoRA rank, width, and alpha must be positive')
        base = np.random.default_rng(0)
        adapt = np.random.default_rng(seed)
        self.vocab = int(vocab)
        self.d = int(d)
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.E = base.normal(0.0, 0.2, size=(self.vocab, self.d))
        self.Wq = base.normal(0.0, 0.2, size=(self.d, self.d))
        self.Wv = base.normal(0.0, 0.2, size=(self.d, self.d))
        self.Wo = base.normal(0.0, 0.2, size=(self.d, self.vocab))
        self.params = {
            'Aq': adapt.normal(0.0, 0.02, size=(self.r, self.d)),
            'Bq': np.zeros((self.d, self.r)),
            'Av': adapt.normal(0.0, 0.02, size=(self.r, self.d)),
            'Bv': np.zeros((self.d, self.r)),
        }
        self.moments = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.step_index = 0
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.eos_id = 1

    def _logits(self, prev_id):
        hidden = self.E[int(prev_id)]
        query = self.Wq @ hidden + self.scale * (self.params['Bq'] @ (self.params['Aq'] @ hidden))
        value = self.Wv @ hidden + self.scale * (self.params['Bv'] @ (self.params['Av'] @ hidden))
        return (query + value) @ self.Wo, hidden

    def _nll_and_grads(self, prev_id, target_id):
        logits, hidden = self._logits(prev_id)
        peak = float(logits.max())
        shifted = logits - peak
        weights = np.exp(shifted)
        total = float(weights.sum())
        probs = weights / total
        nll = float(peak + math.log(total) - logits[int(target_id)])
        dlogits = probs.copy()
        dlogits[int(target_id)] -= 1.0
        d_sum = self.Wo @ dlogits
        query_state = self.params['Aq'] @ hidden
        value_state = self.params['Av'] @ hidden
        grads = {
            'Bq': self.scale * np.outer(d_sum, query_state),
            'Aq': self.scale * np.outer(self.params['Bq'].T @ d_sum, hidden),
            'Bv': self.scale * np.outer(d_sum, value_state),
            'Av': self.scale * np.outer(self.params['Bv'].T @ d_sum, hidden),
        }
        return nll, grads

    def zero_grad(self):
        for name in self.grads:
            self.grads[name].fill(0.0)

    def accumulate(self, input_ids, start, end, token_scale):
        """Add ``token_scale`` times each token's NLL gradient on ``[start, end)``."""
        if not 0 <= start < end <= len(input_ids):
            raise ValueError('Supervised span is outside the example')
        if start == 0:
            raise ValueError('A supervised token needs a preceding context token')
        total = 0.0
        for position in range(start, end):
            nll, grads = self._nll_and_grads(input_ids[position - 1], input_ids[position])
            total += nll * token_scale
            for name, grad in grads.items():
                self.grads[name] += grad * token_scale
        return total

    def mean_nll(self, input_ids, start, end):
        if start >= end:
            return None
        losses = []
        for position in range(start, end):
            nll, _grads = self._nll_and_grads(input_ids[position - 1], input_ids[position])
            losses.append(nll)
        return float(sum(losses) / len(losses))

    def optimizer_step(self, learning_rate):
        norm = math.sqrt(sum(float(np.sum(grad * grad)) for grad in self.grads.values()))
        if norm > self.max_grad_norm:
            factor = self.max_grad_norm / norm
            for name in self.grads:
                self.grads[name] *= factor
        self.step_index += 1
        beta1, beta2 = self.betas
        correct1 = 1.0 - beta1 ** self.step_index
        correct2 = 1.0 - beta2 ** self.step_index
        for name, param in self.params.items():
            grad = self.grads[name]
            self.moments[name] = beta1 * self.moments[name] + (1.0 - beta1) * grad
            self.second[name] = beta2 * self.second[name] + (1.0 - beta2) * (grad * grad)
            update = self.moments[name] / correct1 / (np.sqrt(self.second[name] / correct2) + self.eps)
            param -= learning_rate * (update + self.weight_decay * param)
        self.zero_grad()
        return norm

    def generate(self, prefix_ids, max_new_tokens):
        if not prefix_ids:
            raise ValueError('Generation needs a prompt')
        if max_new_tokens <= 0:
            raise ValueError('Generation cap must be positive')
        produced = []
        previous = int(prefix_ids[-1])
        for _ in range(max_new_tokens):
            logits, _hidden = self._logits(previous)
            nxt = int(np.argmax(logits))
            produced.append(nxt)
            if nxt == self.eos_id:
                return produced, False
            previous = nxt
        return produced, True
