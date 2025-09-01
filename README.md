# Text Embedding with Word2Vec

A project to train and visualize Word2Vec embeddings on Vietnamese text.

## Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Change your data path in config.yaml file.

3. Train and visualize from the CLI:
   ```bash
   python -m text_embed.cli --data data.txt --out models/w2v.model --visualize reports/figures/tsne.png
   ```

