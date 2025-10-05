import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):

        # 保持原始的 ann_path，用于记录实际数据集路径 (在 datasets.py 中使用)
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name

        # --- START MODIFICATION ---
        # 1. 检查是否存在用于词汇表构建的新参数 'vocab_ann_path'
        if hasattr(args, 'vocab_ann_path'):
            vocab_path = args.vocab_ann_path
        else:
            # 2. 如果不存在，退回到原始的 ann_path (警告：这可能导致加载失败)
            vocab_path = args.ann_path
            print("Warning: 'vocab_ann_path' not found. Falling back to 'ann_path' for vocabulary building.")
        # --- END MODIFICATION ---

        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr

        # 使用确定的路径加载 annotation.json 来构建词汇表
        print(f"Loading VOCABULARY from: {vocab_path}")
        # --- START MODIFICATION ---
        self.ann = json.loads(open(vocab_path, 'r').read())
        # --- END MODIFICATION ---

        self.token2idx, self.idx2token = self.create_vocabulary()

        # 打印确认词汇量，以方便调试
        print(f"Vocabulary built: {len(self.token2idx)} content/unk tokens (Target for pretrain is approx 4336)")

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    # ---- 清洗函数 (iu_xray 保持不变) ----
    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    # ==========================================================================================
    # 唯一修改部分：把 MIMIC-CXR 清洗函数还原为“正确代码”的版本（其它不动）
    # ==========================================================================================
    def clean_report_mimic_cxr(self, report):
        """R2GEN的MIMIC-CXR清洗函数（与你给的正确代码一致）"""
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    # ==========================================================================================
    # ---- 辅助函数 (保持不变) ----
    # ==========================================================================================
    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        # 原始 R2Gen 逻辑：如果找不到 token，返回 <unk> 的索引 (即 1)
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        # 原始 R2Gen 逻辑：BOS/EOS 标记使用硬编码的 0 索引
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        # [0] 是 BOS/EOS/PAD 共享的索引
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
