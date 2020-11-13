import csv


class CSVVisualizer:
    def __init__(self, model, batchfier, src_tokenizer, tgt_tokenizer):
        self.model = model
        self.batchfier = batchfier
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def append_decoded_tokens(self, src, tgt, att):
        src_dic = self.src_tokenizer.tokenizer[1]
        tgt_dic = self.tgt_tokenizer.tokenizer[1]

        src_text = [src_dic[i] if i in src_dic else 'PAD' for i in src]
        tgt_text = [tgt_dic[i] if i in tgt_dic else 'PAD' for i in tgt]
        if 'PAD' in tgt_text:
            ind = tgt_text.index('PAD')
            tgt_text = tgt_text[:ind]
            att = att[:ind]
        src_appended = [src_text] + att
        tgt_appended = list(map(lambda x: [x[0]] + x[1], zip([' '] + tgt_text, src_appended)))
        return tgt_appended

    def map_att(self, n_samples, save_path):
        results = []
        processed_samples = 0
        for inp in self.batchfier:
            out = self.model(inp)
            srcs = inp['src'].tolist()
            tgts = inp['tgt'].tolist()
            atts = out['inter_att'][-1].mean(-1).tolist()  # att from the final decoding layer
            for i in range(len(srcs)):
                results.extend(self.append_decoded_tokens(srcs[i], tgts[i], atts[i]))
                results.append([])
                processed_samples +=1
            if processed_samples >= n_samples:
                break
        with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            w.writerows(results)
