# nist_based: benchmark de regressao com backdoor (MLP vs VQC)

Implementacao inicial do plano experimental para comparar regressao classica e quantica sob ataques de backdoor.

## O que ja esta implementado

- Pipeline de dados com splits treino/validacao/teste e normalizacao.
- Interface comum de regressao (`fit`, `predict`, `loss`, `get_representation`).
- Modelo classico MLP.
- Modelo quantico VQC (Qiskit Machine Learning).
- Ataque Trigger Poisoning para regressao.
- Selecao de amostras por baseline aleatoria e heuristica de influencia.
- Scripts de baseline limpo, comparacao e ablacoes por sementes.

## Estrutura

- `configs/`: configuracoes de dataset, modelos e experimentos.
- `data/`: carregamento, preprocessamento e split.
- `models/`: regresssores classico e quantico.
- `attacks/`: ataques de envenenamento.
- `selection/`: estrategias de selecao de amostras para envenenamento.
- `evaluation/`: metricas de regressao, backdoor e agregacao estatistica.
- `experiments/`: scripts executaveis.
- `results/`: saidas de metricas.

## Como executar

Na pasta `nist_based`:

```powershell
pip install -r requirements.txt
python -m experiments.clean_baseline
python -m experiments.run_comparison
python -m experiments.ablation_study
```

Os arquivos de saida sao gravados em `results/metrics`.

### Outro dataset (ex.: AggregatedData / ATM)

Passe outro arquivo de config com `--dataset-config` (ou a variavel de ambiente `DATASET_CONFIG`):

```powershell
python -m experiments.clean_baseline --dataset-config configs/dataset_atm.yaml
python -m experiments.run_comparison --dataset-config configs/dataset_atm.yaml
python -m experiments.ablation_study --dataset-config configs/dataset_atm.yaml
```

Chaves opcionais do bloco `dataset` para CSVs: `drop_columns`, `date_columns` (+ `date_format`, `date_dayfirst`, `date_parts`),
`categorical_columns` (one-hot, + `one_hot_drop_first`), `max_samples` e `output_dir`.
No bloco `poisoning`, `trigger_feature_names` seleciona as features do trigger pelo nome (apos a expansao de datas/one-hot).
