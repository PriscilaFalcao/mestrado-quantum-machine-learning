# Regressão linear como baseline: desempenho limpo e vulnerabilidade a backdoor

Complemento de `RESULTADOS_backdoor_mlp_vs_vqc.md`. Aqui ficam só a pergunta, os resultados e as conclusões sobre a regressão linear (OLS). Os números de MLP e VQC são os do documento principal e não foram refeitos.

## 1. Pergunta e resposta curta

**Precisa de uma regressão linear na comparação de modelos?** Não é obrigatória, mas é um baseline útil: mostra se o backdoor depende de capacidade não linear, que um modelo sem camada oculta não tem.

**O que ela mostrou:**

- **R² limpo de 0,9987 ± 0,0001** no conjunto de teste, o maior dos três modelos (MLP 0,998; VQC 0,962).
- **É a menos vulnerável a backdoor.** O ganho de sucesso sobre o controle é no máximo 0,19, contra 0,91 a 0,996 do MLP.
- Logo, o backdoor do MLP depende de capacidade não linear. Baixa vulnerabilidade não exige erro limpo alto, como o do VQC: um modelo limpo, preciso e linear também resiste.

## 2. O que foi implementado

- `models/linear_regression.py`: `LinearRegressionModel`, um `sklearn.linear_model.LinearRegression` (OLS, sem hiperparâmetros). Treina com todo o conjunto de treino, como o MLP.
- `models/torch_surrogates.py`: `LinearSurrogate` (y = Xw + b), o substituto diferenciável usado por feature collision e bilevel.
- `experiments/model_factory.py`, `attacks/surrogate_training.py` e `configs/models.yaml`: reconhecem o modelo `linear`.
- `experiments/trigger_sweep.py`: nova opção `--models linear`, que roda só esse modelo e grava `seed*_trigger*_linear.json` sem sobrescrever os resultados existentes.
- `experiments/aggregate_sweep.py`: inclui o modelo linear nas tabelas.

## 3. Protocolo

Idêntico ao dos outros modelos: dataset Global Economy Indicators (alvo GDP em log, padronizado), 5 sementes (7, 13, 23, 31, 47), gatilhos 1,5 / 3 / 6 com os quatro ataques mais o controle `none`, e gatilhos 10 / 15 só com `trigger`. Total de 25 jobs. Fração envenenada de 10%, alvo malicioso absoluto c = −0,8. Resultados em `results/metrics/sweep/*_linear.json`, tabela completa em `results/metrics/sweep_summary.md`.

## 4. Resultados

### 4.1 Desempenho limpo (sem ataque, teste, média de 5 sementes)

| Modelo | MAE | RMSE | R² | Erro típico em GDP |
|---|---|---|---|---|
| Linear (OLS) | 0,021σ | 0,036σ | 0,9987 ± 0,0001 | cerca de 6% |
| MLP | 0,029σ | 0,039σ | 0,998 | cerca de 8% |
| VQC | 0,125σ | 0,191σ | 0,962 ± 0,013 | cerca de 39% |

R² = 1 − RMSE²/variância do teste, calculado por semente a partir de `clean_rmse`. A mesma conta reproduz os valores já documentados de MLP e VQC. A tarefa é quase linear: várias colunas são componentes do próprio GDP (`Final consumption expenditure` tem correlação de 0,99 com o alvo).

### 4.2 Ganho de sucesso do backdoor sobre o controle (média ± desvio, 5 sementes)

| Ataque | gatilho 1,5 | gatilho 3 | gatilho 6 | gatilho 10 | gatilho 15 |
|---|---|---|---|---|---|
| trigger | 0,184 ± 0,004 | 0,190 ± 0,003 | 0,192 ± 0,002 | 0,192 ± 0,003 | 0,192 ± 0,003 |
| feature_collision | 0,138 ± 0,007 | 0,147 ± 0,006 | 0,164 ± 0,003 | – | – |
| bilevel | 0,116 ± 0,048 | 0,118 ± 0,024 | 0,151 ± 0,027 | – | – |
| clean_label_bilevel | 0,057 ± 0,052 | 0,009 ± 0,011 | 0,000 ± 0,001 | – | – |

Para comparação, o ganho do `trigger` no MLP vai de 0,914 a 0,996 e, no VQC, de 0,009 a 0,419.

### 4.3 Controle (`none`)

| Gatilho | 1,5 | 3 | 6 | 10 | 15 |
|---|---|---|---|---|---|
| Sucesso espontâneo | 0,007 | 0,000 | 0,000 | 0,000 | 0,000 |
| Deslocamento da previsão limpa | 1,71 | 3,43 | 6,85 | 11,4 | 17,1 |

O deslocamento cresce na proporção do gatilho porque o modelo é linear: um gatilho de magnitude *t* nas duas features move a previsão por *t* vezes a soma dos coeficientes delas. Mesmo assim o sucesso espontâneo é nulo, pois o deslocamento passa muito do alvo.

### 4.4 Custo do ataque na qualidade limpa (gatilho 1,5)

| Ataque | MAE limpo | R² limpo |
|---|---|---|
| none (controle) | 0,021 | 0,9987 |
| trigger | 0,097 | 0,984 |
| feature_collision | 0,062 | 0,993 |
| clean_label_bilevel | 0,049 | 0,994 |
| bilevel | 0,121 | 0,966 |

Com gatilho 6, o R² limpo fica em 0,986 (trigger), 0,993 (feature_collision), 0,997 (clean_label_bilevel) e 0,962 (bilevel).

## 5. Conclusões

1. **O backdoor não pega num modelo linear.** O `trigger` satura em cerca de 0,19 de sucesso em qualquer intensidade, porque o OLS só consegue deslocar a média ao tentar ajustar 10% de rótulos contraditórios. Ele não separa "com gatilho → rótulo malicioso" do restante.
2. **O ataque degrada o modelo em vez de implantar o backdoor.** O MAE limpo sobe de 0,021 para 0,05 a 0,12, e o R² cai até 0,966 (bilevel), sem sucesso de backdoor relevante. É o único modelo em que isso acontece com clareza; no MLP o envenenamento é praticamente invisível no erro limpo.
3. **Os ataques clean-label não funcionam.** O `clean_label_bilevel` vai a 0,000 para gatilho 6, e o `feature_collision` fica abaixo de 0,17.
4. **A otimização bilevel não ajuda.** Fica abaixo do `trigger` simples em todas as intensidades, o mesmo padrão dos outros modelos.
5. **Leitura para a comparação MLP × VQC.** O documento principal ressalva que a resiliência do VQC pode vir só do erro limpo alto (R² 0,962). O baseline linear dá um contraponto: um modelo com R² ainda melhor que o do MLP também resiste. O que decide a vulnerabilidade é a capacidade de aprender a interação do gatilho com o rótulo, e não o erro limpo isoladamente. Isso reforça o argumento de não atribuir a resiliência do VQC à natureza quântica.

## 6. Limitações

- No feature collision, a "representação" da regressão linear é a própria entrada (não há camada oculta). O ataque vira aproximar a entrada da de um alvo com gatilho, o que é uma adaptação por definição.
- O substituto linear é treinado com Adam e não pela solução fechada do OLS, uma aproximação do mesmo tipo da usada para MLP e VQC.
- Os avisos de overflow e NaN no `matmul` do sklearn durante a execução são falsos: as predições coincidem com o cálculo manual até 1e-16, e coeficientes e predições são finitos.
- Valem as limitações do documento principal: um dataset, um formato de gatilho, 5 sementes.

## 7. Reprodução

Na pasta `nist_based`:

```bash
# um job por (semente, gatilho), só a regressão linear
python -m experiments.trigger_sweep 7 3 trigger feature_collision bilevel clean_label_bilevel --models linear
python -m experiments.aggregate_sweep   # tabelas -> results/metrics/sweep_summary.{md,json}
```

Os 25 jobs estão em `results/logs/jobs.txt` e os logs da regressão linear em `results/logs/linear_job_*`.
