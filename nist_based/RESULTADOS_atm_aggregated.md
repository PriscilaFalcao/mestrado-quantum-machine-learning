# Resultados: backdoor em regressão no dataset AggregatedData (ATM), MLP × VQC

Dados: *AggregatedData* (saques diários de 5 caixas eletrônicos, 2011–2017; alvo: `Total amount Withdrawn`). Experimento **preliminar**: uma única semente (42), um único gatilho (1,5σ), ataque trigger com duas estratégias de seleção de amostras (aleatória e por influência). Os números vêm de `results/metrics/atm_aggregated/` (`clean_baseline.json` e `trigger_comparison.json`). O R² não é salvo pelos scripts; foi calculado a partir do RMSE (ver 3.1).

> **Atenção:** estes resultados foram gerados com a versão do pipeline anterior ao merge com `main` (VQC com `ZZFeatureMap` e 2 qubits, seleção aleatória/por influência, alvo malicioso `shift`). Não são diretamente comparáveis com `RESULTADOS_backdoor_mlp_vs_vqc.md`, que usa o VQC com codificação por ângulos, alvo absoluto, controle `none` e 5 sementes.

## 1. Resposta curta

**Neste dataset o MLP aprende o backdoor sem perder desempenho limpo, e o VQC não aprende nem a tarefa limpa, então a "resistência" dele ao ataque não é informativa.**

- **MLP:** R² limpo de 0,887, que se mantém (0,888–0,889) com 10% de envenenamento. Com o gatilho, o R² cai para −0,21 (seleção aleatória) e −0,65 (seleção por influência), e a previsão se desloca em média 0,97σ e 1,12σ, praticamente o deslocamento de +1σ imposto aos rótulos envenenados.
- **VQC:** R² limpo de 0,005, ou seja, equivalente a prever a média. O gatilho desloca a previsão em apenas 0,09σ a 0,13σ e a taxa de sucesso do backdoor é 0.
- **Seleção por influência:** no MLP, produz um backdoor mais forte que a seleção aleatória (deslocamento 1,12 × 0,97; R² com gatilho −0,65 × −0,21) sem custo no erro limpo.
- **Ressalva central:** como o VQC não aprende a tarefa, não é possível concluir nada sobre a robustez dele a partir deste experimento.

## 2. Protocolo

| Item | Valor |
|---|---|
| Dataset | `AggregatedData.csv`, 11.589 linhas; subconjunto aleatório de 2.000 (`max_samples`) → treino 1.200, validação 400, teste 400 |
| Features (28) | 3 contagens de saques (`No Of Withdrawals`, `No Of XYZ Card Withdrawals`, `No Of Other Card Withdrawals`); data expandida em ano, mês, dia, dia da semana e dia do ano; one-hot de `ATM Name`, `Festival Religion`, `Working Day` e `Holiday Sequence` |
| Colunas removidas | `Amount withdrawn XYZ Card` e `Amount withdrawn Other Card` (somam exatamente o alvo, vazamento); `Weekday` (redundante com a data e com maiúsculas inconsistentes) |
| Pré-processamento | datas em formato misto (`1/1/2011` e `13-01-2011`, dia primeiro); imputação por mediana e padronização ajustadas só no treino; alvo padronizado (sem log) |
| MLP | 64-32, ReLU, `alpha` 1e-4, até 800 iterações |
| VQC | 2 qubits, `ZZFeatureMap` (1 repetição), `RealAmplitudes` (1 camada), COBYLA com até 120 iterações; entrada reduzida por PCA para 2 componentes |
| Envenenamento | 10% do treino (120 amostras) |
| Seleção | `random` (aleatória) e `influence` (maior resíduo × norma num Ridge) |
| Gatilho | soma 1,5 (em σ) às features `No Of Other Card Withdrawals` e `Transaction Date_dayofyear` |
| Alvo malicioso | `shift`: rótulo envenenado = rótulo + 1σ |
| Métrica de sucesso | fração das previsões com gatilho a menos de 0,2 de `média(y_treino) + 1` |

## 3. Resultados

### 3.1 Desempenho limpo (sem ataque, conjunto de teste)

MAE e RMSE estão em desvios-padrão do alvo (σ, padronizado no treino). R² = 1 − RMSE² / Var(y_teste), com Var(y_teste) = 0,855; é o mesmo valor do `r2_score` do scikit-learn.

| Modelo | MAE | RMSE | R² | Tempo de treino |
|---|---|---|---|---|
| MLP | 0,231σ | 0,310σ | 0,887 | ~2 s |
| VQC | 0,689σ | 0,921σ | 0,005 | ~265 s |

### 3.2 Ataque trigger

| Modelo | Seleção | MAE limpo | RMSE limpo | R² limpo | RMSE com gatilho | R² com gatilho | Deslocamento da previsão | Sucesso do backdoor |
|---|---|---|---|---|---|---|---|---|
| MLP | random | 0,223 | 0,309 | 0,889 | 1,018 | −0,212 | 0,970 | 0,178 |
| MLP | influence | 0,228 | 0,309 | 0,888 | 1,187 | −0,648 | 1,116 | 0,185 |
| VQC | random | 0,697 | 0,921 | 0,008 | 0,933 | −0,018 | 0,090 | 0,000 |
| VQC | influence | 0,700 | 0,926 | −0,003 | 0,928 | −0,007 | 0,132 | 0,000 |

"R² com gatilho" compara a previsão com gatilho com o valor real; quanto mais negativo, mais o gatilho afasta a previsão da verdade. "Deslocamento" é a média de |previsão com gatilho − previsão limpa|.

## 4. Por que os resultados deram assim

**4.1 MLP aprende o backdoor de forma furtiva (medido).**
O R² limpo não muda com o envenenamento (0,887 → 0,888–0,889), mas o gatilho desloca a previsão em ~1σ, exatamente o deslocamento aplicado aos rótulos envenenados. O modelo aprendeu a regra "gatilho presente → somar 1σ". Como ele não perde nada nos dados limpos, o ataque não seria percebido por uma validação comum.

**4.2 Taxa de sucesso baixa no MLP é artefato da métrica (medido + interpretação).**
O alvo `shift` envenena cada rótulo com y + 1, mas a taxa de sucesso mede a distância a um valor fixo, `média + 1`. Um modelo que aprendeu perfeitamente o deslocamento acerta esse valor fixo só nas amostras perto da média, por isso a taxa fica em ~18% mesmo com o backdoor implantado. Para este alvo, o deslocamento da previsão é a leitura correta.

**4.3 Seleção por influência fortalece o backdoor no MLP (medido).**
A seleção por influência escolhe amostras com resíduo alto e alta norma, mais "extremas" no espaço de entrada. O deslocamento sobe de 0,97σ para 1,12σ e o R² com gatilho cai de −0,21 para −0,65, sem custo no erro limpo. Com uma única semente, não é possível dizer se a diferença é robusta.

**4.4 VQC não aprende a tarefa (medido).**
O R² de 0,005 indica que o VQC prevê praticamente a média. Hipóteses (não testadas aqui, mas consistentes com o que foi observado no Global Economy Indicators): a PCA comprime 28 features em 2 componentes e descarta muita informação; a `ZZFeatureMap` oscila demais; e a saída do circuito, limitada a [−1, 1], não cobre a faixa do alvo. Por isso o deslocamento pequeno com gatilho (0,09σ a 0,13σ) e o sucesso 0 não indicam robustez: o modelo não aprende nem o padrão limpo nem o do gatilho.

## 5. Limitações

- Uma única semente (42) e um único gatilho (1,5σ). O `ablation_study` com várias sementes foi iniciado, mas não gerou `ablation_summary.json`.
- Sem controle `none` (modelo limpo avaliado com gatilho), então não há ganho sobre o controle como no outro relatório.
- O alvo `shift` não casa com a métrica de sucesso baseada em valor fixo (ver 4.2).
- Só 2.000 das 11.589 linhas foram usadas, e MLP e VQC treinaram nas mesmas 1.200 amostras.
- O contador `circuit_evaluations` ficou em 0 (bug corrigido depois no `main`).
- As contagens de saques são fortemente ligadas ao alvo (o valor sacado é aproximadamente nº de saques × valor médio), o que torna a tarefa fácil para o MLP.
- O VQC é simulado (estado exato, sem ruído).

## 6. Próximos passos sugeridos

1. Refazer com o pipeline atual do `main` (VQC com codificação por ângulos e `output_scale`, alvo absoluto, controle `none`, 5 sementes e várias intensidades de gatilho) usando `configs/dataset_atm.yaml`.
2. Avaliar transformar o alvo com `log1p`, que no outro dataset foi o que permitiu o VQC aprender.
3. Salvar `r2` diretamente nas métricas (`clean_r2`, `trigger_r2`).
4. Testar o dataset completo (`max_samples: null`) no MLP e um subconjunto maior no VQC.

## 7. Reprodução

Na pasta `nist_based` (versão do pipeline usada nestes resultados):

```powershell
python -m experiments.clean_baseline --dataset-config configs/dataset_atm.yaml
python -m experiments.run_comparison --dataset-config configs/dataset_atm.yaml
python -m experiments.ablation_study --dataset-config configs/dataset_atm.yaml
```

Saídas em `results/metrics/atm_aggregated/`.
