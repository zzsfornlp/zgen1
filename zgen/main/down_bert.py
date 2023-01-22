#

# down-load bert

def main(*bert_names):
    from transformers import AutoTokenizer, AutoModel
    for bert_name in bert_names:
        t = AutoTokenizer.from_pretrained(bert_name)
        m = AutoModel.from_pretrained(bert_name)
        print(f"Load {bert_name}: {t} and {m.config}")
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
