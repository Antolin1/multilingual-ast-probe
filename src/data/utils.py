import io
import tokenize
import requests
import zipfile
import gzip
from typing import Optional

from tqdm import tqdm


def download_url(url: str, save_path: str, chunk_size: int = 128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            f.write(chunk)


def unzip_file(file_path: str, output_dir: str, output_path: Optional[str] = None):
    if 'gz' in file_path:
        with gzip.open(file_path, 'rb') as f1, open(output_path, 'w') as f2:
            f2.write(f1.read().decode('utf-8'))
    else:
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(output_dir)


def remove_comments_and_docstrings_python(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


def match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer):
    tokenized = []
    mapping = {}
    cont = 0
    for j,t in enumerate(untokenized_sent):
        if j == 0:
            temp = [k for k in tokenizer.tokenize(t) if k!='Ġ']
            tokenized.append(temp)
            mapping[j] = [f for f in range(cont, len(temp) + cont)]
            cont = cont + len(temp)
        else:
            temp = [k for k in tokenizer.tokenize(' '+t) if k!='Ġ']
            tokenized.append(temp)
            mapping[j] = [f for f in range(cont, len(temp) + cont)]
            cont = cont + len(temp)
    flat_tokenized = [item for sublist in tokenized for item in sublist]
    return flat_tokenized, mapping
