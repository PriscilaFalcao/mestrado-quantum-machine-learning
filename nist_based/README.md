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
