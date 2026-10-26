# Resultados: backdoor em regressão, MLP × VQC

Dados: *Global Economy Indicators* (alvo: GDP). Experimento completo: 5 sementes (7, 13, 23, 31, 47), 5 intensidades de gatilho, 4 famílias de ataque mais um controle sem envenenamento. Todos os números abaixo são média ± desvio-padrão entre sementes e vêm de `results/metrics/sweep/` (tabela completa em `results/metrics/sweep_summary.md`, gerada por `python -m experiments.aggregate_sweep`).

## 1. Resposta curta

**O VQC é mais resiliente que o MLP, mas só até certo ponto, e essa resiliência não pode ser atribuída com segurança à natureza quântica.**

- **MLP:** é comprometido por completo em todas as intensidades de gatilho (ganho de 0,91 a 0,996 sobre o controle), com custo desprezível no erro limpo.
- **VQC, gatilho fraco (1,5σ):** fica inerte. O ganho do trigger sobre o controle é 0,009 ± 0,008.
- **VQC, gatilho moderado a forte (3σ ou mais):** passa a ser vulnerável, com ganho de 0,26 a 0,42, mas com enorme variação entre sementes (desvio de 0,11 a 0,32) e sempre bem abaixo do MLP.
- **Ataques clean-label:** fracos no MLP (ganho de 0,06 a 0,12) e nulos no VQC.
- **Otimização bilevel:** não supera o trigger simples em nenhum modelo.
- **Ressalva central:** o VQC erra cerca de 4× mais que o MLP em dados limpos (MAE 0,125 contra 0,029). Um modelo com menos capacidade tende a ignorar padrões sutis. O plano (seção 13) pede cautela exatamente neste ponto: menor efeito do backdoor com maior erro limpo não prova robustez.

## 2. Protocolo

| Item | Valor |
|---|---|
| Dataset | `Global Economy Indicators`, 10.512 linhas, 21 features (removidas as colunas que vazam o alvo: GNI e Total Value Added) |
| Pré-processamento | `log1p` com sinal nas features assimétricas (escolhidas só no treino), imputação por mediana, padronização. Alvo: `log1p` e padronização. |
| MLP | 64-32, ReLU, treinado com todo o treino (6.306 amostras) |
| VQC | 4 qubits, codificação por ângulos Ry, ansatz `RealAmplitudes` com 2 camadas (12 parâmetros), observável Z⊗Z⊗Z⊗Z, COBYLA com até 500 avaliações. PCA *whitened* em 4 componentes × `input_scale` 0,5; saída = 2,5 × ⟨Z⊗⁴⟩. Treino com subconjunto aleatório de 1.000 amostras. |
| Envenenamento | 10% das amostras de treino |
| Alvo malicioso | Valor absoluto c = -0,8 (em desvios-padrão do alvo em log), tolerância 0,2 |
| Gatilho | soma `trigger_value` às features 4 e 5 (`Population` e `Per capita GNI`) |
| Intensidades | 1,5; 3; 6 (todos os ataques); 10 e 15 (só o trigger simples) |
| Métrica principal | taxa de sucesso do backdoor (fração das previsões com gatilho a menos de 0,2 de c) e seu **ganho sobre o controle** `none` na mesma semente e no mesmo gatilho |

### Os quatro ataques
- **trigger:** soma o gatilho nas amostras escolhidas ao acaso e troca o rótulo por c.
- **feature_collision (clean-label):** move amostras, sem mexer no rótulo, até a representação delas colidir com a de entradas com gatilho. As bases são as de rótulo mais próximo de c.
- **bilevel:** parte do trigger e refina as entradas (|δ|∞ ≤ 0,5) otimizando, através do treino desenrolado de um substituto diferenciável, o efeito do backdoor em dados de validação.
- **clean_label_bilevel:** o mesmo otimizador, com rótulo intacto, sem gatilho no treino e |δ|∞ ≤ 1,0.

Os ataques otimizados são fabricados contra um substituto diferenciável (MLP em PyTorch; simulador de statevector do circuito, validado contra o Qiskit com erro ~1e-16) e avaliados treinando o modelo real nos dados envenenados.

## 3. Resultados

### 3.1 Ganho de sucesso do backdoor sobre o controle (média ± desvio, 5 sementes)

| Ataque | Modelo | gatilho 1,5 | gatilho 3 | gatilho 6 | gatilho 10 | gatilho 15 |
|---|---|---|---|---|---|---|
| trigger | MLP | 0,914 ± 0,025 | 0,955 ± 0,023 | 0,985 ± 0,013 | 0,995 ± 0,004 | 0,996 ± 0,001 |
| trigger | VQC | 0,009 ± 0,008 | 0,419 ± 0,323 | 0,418 ± 0,228 | 0,262 ± 0,113 | 0,357 ± 0,190 |
| bilevel | MLP | 0,806 ± 0,080 | 0,934 ± 0,030 | 0,983 ± 0,013 | – | – |
| bilevel | VQC | 0,015 ± 0,010 | 0,355 ± 0,162 | 0,324 ± 0,201 | – | – |
| feature_collision | MLP | 0,090 ± 0,015 | 0,114 ± 0,026 | 0,109 ± 0,120 | – | – |
| feature_collision | VQC | -0,004 ± 0,006 | -0,002 ± 0,002 | 0,000 ± 0,002 | – | – |
| clean_label_bilevel | MLP | 0,122 ± 0,063 | 0,115 ± 0,066 | 0,063 ± 0,043 | – | – |
| clean_label_bilevel | VQC | 0,010 ± 0,013 | 0,014 ± 0,004 | 0,010 ± 0,006 | – | – |

### 3.2 Controle (`none`): taxa de sucesso "espontânea" e erro limpo

| Modelo | gatilho 1,5 | 3 | 6 | 10 | 15 | MAE limpo |
|---|---|---|---|---|---|---|
| MLP | 0,067 | 0,035 | 0,010 | 0,002 | 0,000 | 0,029 ± 0,003 |
| VQC | 0,103 | 0,083 | 0,064 | 0,034 | 0,023 | 0,125 ± 0,018 |

O deslocamento médio da previsão do modelo **limpo** ao receber o gatilho cresce com a intensidade (MLP: 0,41 → 4,05; VQC: 0,12 → 0,93). Ou seja, gatilhos grandes empurram qualquer modelo para fora da distribuição de treino. Por isso a taxa de sucesso bruta não é comparável entre intensidades, e o ganho sobre o controle é a leitura correta.

### 3.3 Custo do ataque no erro limpo (MAE limpo com ataque, no gatilho 1,5)

| Ataque | MLP (controle 0,029) | VQC (controle 0,125) |
|---|---|---|
| trigger | 0,033 | 0,157 |
| feature_collision | 0,037 | 0,123 |
| bilevel | 0,035 | 0,192 |
| clean_label_bilevel | 0,038 | 0,170 |

No MLP o envenenamento é praticamente invisível no erro limpo. No VQC, o trigger e o bilevel pioram o erro limpo em cerca de 0,03 a 0,07, ou seja, o VQC acomoda o envenenamento pior.

## 4. Por que os resultados deram assim

Separo o que foi **medido** do que é **hipótese**.

**4.1 MLP vulnerável a tudo que mexe no rótulo (medido: ganho ≥ 0,8).**
O trigger troca o rótulo de 10% das amostras por c = -0,8 e desloca duas features em um valor fixo. Isso cria um padrão simples e separável: "se as features 4 e 5 estão deslocadas, a saída é -0,8". O MLP tem milhares de parâmetros e aprende o padrão sem prejudicar a função principal (erro limpo 0,029 → 0,033). Hipótese: a mesma capacidade que o faz acertar a tarefa limpa (RMSE 0,05) o torna capaz de memorizar o gatilho.

**4.2 VQC inerte ao gatilho fraco (medido).**
Com gatilho de 1,5σ, o deslocamento do gatilho na entrada do circuito é de apenas 0,02 a 0,09 rad (por componente do PCA *whitened* com escala 0,5), contra uma dispersão típica dos dados de cerca de 0,4 rad. O gatilho fica dentro do ruído do que o circuito enxerga. Isso foi calculado para a semente 42. Com 12 parâmetros, o VQC também não tem capacidade de recortar um cluster pequeno da entrada.

**4.3 VQC vulnerável, mas instável, a gatilhos maiores (medido: ganho 0,26 a 0,42; desvio até 0,32).**
Ao crescer o gatilho, o deslocamento em ângulo passa a ser comparável à dispersão dos dados, e o circuito consegue separar as amostras envenenadas. O efeito, porém, varia muito entre sementes, e não é monotônico (0,42 em 3σ e 6σ, 0,26 em 10σ, 0,36 em 15σ). **A causa dessa não monotonicidade não foi estabelecida.** Cogitei uma explicação por periodicidade dos ângulos, mas descartei: o deslocamento calculado nesses casos fica abaixo de ~1 rad, longe de 2π. Candidatos que não foram testados: convergência do COBYLA, que dá resultados diferentes por semente com poucos parâmetros, e dependência do subconjunto de 1.000 amostras sorteado.

**4.4 Resiliência × capacidade (ressalva importante).**
O VQC tem erro limpo 4× maior que o MLP. Parte do "não aprender o backdoor" pode ser simplesmente "não aprender nada com tanta precisão". Este experimento não separa as duas explicações. Um controle adequado seria um MLP com capacidade e erro limpo equivalentes ao do VQC (por exemplo, uma MLP pequena, treinada em 1.000 amostras).

**4.5 Bilevel não supera o trigger (medido).**
No MLP, o bilevel dá ganho de 0,806, 0,934 e 0,983 contra 0,914, 0,955 e 0,985 do trigger. No gatilho fraco ele chega a ser pior. O trigger já satura (≥ 0,91), então há pouco espaço para melhorar, e a perturbação otimizada (até 0,5σ) afasta as amostras envenenadas do padrão exato do gatilho. Hipótese (não medida): o substituto, com apenas 10 passos desenrolados, aproxima mal o treino real do MLP (800 iterações de Adam). A fidelidade do substituto não foi medida. A resposta à RQ2 neste experimento é, portanto, **negativa para o bilevel dirty-label**.

**4.6 Clean-label fraco (medido).**
Sem o gatilho no treino e sem trocar rótulos, o MLP só aprende o backdoor de modo parcial (ganho de 0,06 a 0,12) e o VQC não aprende nada (≈ 0). Isso é esperado de ataques clean-label, que são mais difíceis. Os hiperparâmetros (`eps`, passos, `beta`) **não foram ajustados**, então esses números são um piso, e não o desempenho máximo do ataque.

## 5. Decisões tomadas ao longo do projeto e por quê

1. **Seleção de amostras removida.** O MetaPoison original otimiza as amostras (formulação bilevel), não escolhe quais usar. As heurísticas por influência e meta-learning foram removidas, e o plano foi atualizado. As amostras são aleatórias, exceto nos ataques clean-label, que exigem rótulo já próximo do alvo.
2. **Ataques com substitutos diferenciáveis.** O MLP do scikit-learn e o `VQR` com COBYLA não são diferenciáveis. Os ataques são fabricados num substituto em PyTorch (e num simulador do circuito validado contra o Qiskit) e avaliados nos modelos reais.
3. **Vazamento do alvo.** `Gross National Income` tinha correlação 1,000 com o GDP e `Total Value Added` 0,999. As duas colunas foram removidas.
4. **VQC com 1.000 amostras.** O custo do COBYLA no simulador do Qiskit escala com o número de amostras (um ajuste com 6.300 levava ~4,5 min só com 120 iterações). Isso significa que MLP e VQC treinam com volumes diferentes.
5. **Alvo em log e features em log.** O alvo era muito assimétrico (até 25σ). O log do alvo e das features assimétricas fez uma regressão linear passar de RMSE 0,78 para 0,04 e foi o que permitiu o VQC aprender.
6. **Alvo malicioso absoluto e escolha de c = -0,8.** O alvo "rótulo + 1σ" se confundia com o efeito natural do gatilho em colunas informativas (o controle chegava a 41,6%). Com c = +0,8 o controle chegou a 60%, porque era justamente onde o modelo limpo "aterrissava" com o gatilho. O valor c = -0,8 foi fixado com o critério explícito de controle baixo, medido **antes** de olhar o desempenho dos ataques. Depois das mudanças nas features o controle subiu um pouco (0% a 10%, conforme o gatilho), por isso a análise usa o ganho sobre o controle.
7. **VQC.** A `ZZFeatureMap` oscila demais e não aprendeu a tarefa (RMSE 0,8 a 1,0) em nenhuma combinação de qubits, camadas ou observável. Troquei para codificação por ângulos (Ry), normalizei a entrada (PCA *whitened* × 0,5) e adicionei `output_scale` = 2,5, porque a saída do circuito fica em [-1, 1] e o alvo chega a ±3. Em um diagnóstico anterior eu atribuí o mau resultado ao COBYLA, e isso estava errado: o simulador sem escala de saída dava os mesmos resultados do Qiskit, e o limite era a faixa da saída.
8. **Gatilho variado, não único.** Como um valor único de gatilho tornaria a conclusão dependente da escolha, o gatilho varia em 5 intensidades iguais para os dois modelos, e o controle é refeito em cada uma.
9. **Bugs corrigidos que afetam os números.**
   - O contador `circuit_evaluations` ficava em 0, porque o callback só dispara com otimizadores SciPy. Agora conta-se na chamada `forward` da rede.
   - O treino do VQC **não era determinístico**: a mesma semente dava pesos diferentes, porque `algorithm_globals.random_seed` não fixa o ponto inicial do COBYLA. Agora o ponto inicial é passado explicitamente. Os resultados anteriores a essa correção foram arquivados em `results/metrics_prerun_nondeterministic/` e não entram nas conclusões.

## 6. Limitações

- Um único dataset, uma única tarefa de regressão, um único formato de gatilho (soma em duas features).
- Só 5 sementes. O VQC varia muito entre sementes, e as médias de 3σ a 15σ têm desvios altos.
- O VQC treina com 1.000 amostras e o MLP com 6.306. Falta o controle de capacidade da seção 4.4.
- Os hiperparâmetros do VQC (`input_scale`, `output_scale`, qubits, camadas) foram escolhidos pelo desempenho limpo numa única semente (42), sem outro conjunto de seleção.
- Hiperparâmetros dos ataques otimizados não foram ajustados; a fidelidade do substituto não foi medida.
- O controle `none` do VQC fica entre 2% e 10% (o do MLP, entre 0% e 7%), acima do critério de 5% que fixei para c nos gatilhos fracos. Por isso as conclusões usam o ganho sobre o controle.
- O VQC é simulado (estado exato, sem ruído de disparos nem de hardware).

## 7. Próximos passos sugeridos

1. Controle de capacidade: treinar um MLP pequeno (e/ou com 1.000 amostras) com erro limpo próximo ao do VQC e repetir a varredura de gatilho.
2. Medir a fidelidade dos substitutos (previsão do substituto contra a do modelo real) e só então ajustar `eps`, passos e `beta` dos ataques clean-label e bilevel.
3. Investigar a variância do VQC entre sementes (convergência do COBYLA, subconjunto de 1.000 amostras) com mais sementes.
4. Testar a sensibilidade a `input_scale`, que controla quanto o gatilho altera o circuito.

## 8. Reprodução

Na pasta `nist_based`:

```bash
python -m tests.check_vqc_sim                         # simulador × Qiskit
python -m experiments.clean_baseline
# um job por (semente, gatilho); os ataques são os argumentos finais
python -m experiments.trigger_sweep 7 3 trigger feature_collision bilevel clean_label_bilevel
python -m experiments.aggregate_sweep                 # tabelas -> results/metrics/sweep_summary.{md,json}
```

Os 25 jobs usados aqui estão listados em `results/logs/jobs.txt`, e os logs em `results/logs/`.
