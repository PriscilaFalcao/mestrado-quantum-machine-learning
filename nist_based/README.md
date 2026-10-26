# nist_based: benchmark de regressao com backdoor (MLP vs VQC)

Compara regressao classica (MLP) e quantica (VQC, Qiskit) sob quatro familias de ataque de envenenamento/backdoor. Nao ha mais estrategias de selecao de amostras (influencia / meta-learning): o unico criterio de escolha e o minimo que cada ataque exige.

## Ataques implementados (`attacks/`)

| Ataque | Arquivo | O que faz | Amostras envenenadas |
|---|---|---|---|
| `trigger` | `trigger_poisoning.py` | Soma o gatilho nas features e desloca o rotulo para o alvo malicioso | Aleatorias |
| `feature_collision` | `feature_collision.py` | Clean-label: move a entrada (bola L-inf de raio `eps`) ate a representacao colidir com a de um alvo com gatilho; rotulo intacto | As de rotulo mais proximo do alvo |
| `bilevel` | `bilevel.py` | Trigger + rotulo malicioso, e refina as entradas por otimizacao bilevel (treino desenrolado, ensemble de estagios/inits) | Aleatorias |
| `clean_label_bilevel` | `clean_label_bilevel.py` | Bilevel so com perturbacao de entrada, rotulo e gatilho de treino intactos | As de rotulo mais proximo do alvo |

`none` roda o mesmo protocolo sem envenenar (controle). Em `attacks/pipeline.py`, `craft_poisoned_dataset` monta o conjunto envenenado de cada ataque.

### Como os ataques diferenciaveis funcionam nos dois modelos

O MLP do sklearn e o `VQR` do Qiskit (COBYLA) nao sao diferenciaveis. Por isso feature collision e bilevel sao fabricados contra um **substituto diferenciavel** do modelo, e o envenenamento e avaliado treinando o **modelo real** nos dados envenenados:

- MLP: substituto em PyTorch com a mesma arquitetura (`models/torch_surrogates.py`). Representacao = ultima camada oculta.
- VQC: simulador de statevector em PyTorch (`models/torch_vqc_sim.py`) do mesmo circuito (`ZZFeatureMap` + `RealAmplitudes`, observavel Z^n), com PCA linear fixo antes do circuito. Validado contra o Qiskit em `tests/check_vqc_sim.py` (erro ~1e-15). Representacao = valores esperados `<Z_q>` por qubit; `QuantumVQCRegressor.get_representation` usa a mesma definicao com os pesos treinados.

O bilevel usa o substituto em varios estagios de treino (`ensemble_stages`) e varias inicializacoes (`ensemble_seeds`), no espirito do MetaPoison, e desenrola `inner_steps` passos de gradiente. Isso e uma **aproximacao**: o treino real (Adam por 800 iteracoes no MLP, COBYLA no VQC) nao e desenrolado.

## Estrutura

- `configs/`: `dataset.yaml` (dados, gatilho, tolerancia), `models.yaml`, `attacks.yaml` (parametros de cada ataque), `experiments.yaml` (sementes, ataques, modelos).
- `data/`, `models/`, `attacks/`, `selection/random_selection.py`, `evaluation/`, `experiments/`, `tests/`.
- `results/metrics/`: saidas JSON.

## Dados e protocolo

- Dataset atual: `Global Economy Indicators` (`first_try/data/`), alvo GDP, configurado em `configs/dataset.yaml`. Valores faltantes sao imputados pela mediana do treino. `drop_columns` remove colunas quase identicas ao alvo (GNI e Total Value Added), que vazariam a resposta.
- O VQC treina com um subconjunto aleatorio de `max_train_samples` amostras (`configs/models.yaml`), sorteado antes do envenenamento, porque o COBYLA no simulador do Qiskit escala com o numero de amostras. O MLP usa todo o treino. Validacao e teste sao os mesmos para os dois.
- `experiments/ablation_study.py` repete a comparacao para varias sementes (`configs/experiments.yaml`) e agrega media e desvio. Os eixos variados sao ataque e modelo, e o ataque `none` e o controle.
- O plano completo esta em `plano_ataques_backdoor_regressao_quantica_classica.md`.

## Como executar

Na pasta `nist_based`:

```bash
pip install -r requirements.txt
python -m tests.check_vqc_sim          # valida o simulador contra o Qiskit
python -m experiments.clean_baseline
python -m experiments.run_comparison   # 1 semente, todos os ataques x modelos -> attack_comparison.json
python -m experiments.ablation_study   # varias sementes, com agregacao -> ablation_summary.json
```

Metricas por execucao: `clean_mae/rmse`, `trigger_mae/rmse`, `prediction_shift`, `backdoor_success_rate`, `craft_seconds`, `train_seconds`, `poisoned_samples`, `mean_linf_perturbation` e, no VQC, `circuit_evaluations`.
