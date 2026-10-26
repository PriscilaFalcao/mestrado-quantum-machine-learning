# Plano de implementação: ataques de backdoor em modelos de regressão clássicos e quânticos

## 1. Objetivo

Desenvolver um estudo experimental para comparar modelos de regressão clássicos e quânticos quando submetidos a ataques de envenenamento (*poisoning*) e backdoor. O estudo também investigará se a otimização das amostras envenenadas, inspirada no MetaPoison (formulação bilevel), aumenta a eficácia dos ataques em comparação com o trigger poisoning simples.

O plano será orientado pelas famílias de ataques discutidas na Seção 3.3 do artigo de referência, *Wild Patterns Reloaded*. O MetaPoison é discutido em outra seção do artigo; portanto, sua utilização neste contexto deve ser descrita como uma **adaptação inspirada no MetaPoison**, e não como uma reprodução direta do método original.

## 2. Perguntas de pesquisa

- **RQ1:** Como modelos de regressão clássicos e quânticos diferem em desempenho limpo e vulnerabilidade a backdoors?
- **RQ2:** As diferenças entre modelos persistem quando se controlam o erro de regressão em dados limpos, o custo computacional e a variabilidade entre inicializações?

## 3. Escopo inicial e desenho experimental

Para evitar que a complexidade dos ataques torne o estudo difícil de reproduzir, recomenda-se começar com um escopo reduzido:

1. Um conjunto de dados de regressão.
2. Um modelo clássico MLP (*multilayer perceptron*).
3. Um modelo de regressão quântico variacional (VQC).
4. Um ataque inicial de *trigger poisoning*.
5. Comparação entre os ataques (trigger, feature collision, bilevel e bilevel clean-label), com as amostras a envenenar escolhidas aleatoriamente quando o ataque não exige outro critério.
6. Avaliação com múltiplas sementes aleatórias.

Após validar esse pipeline, adicionar os ataques bilevel e os ataques *clean-label*.

### 3.1 Comparação justa entre modelos

- Usar as mesmas divisões de treino, validação e teste.
- Aplicar o mesmo pré-processamento e as mesmas regras de normalização.
- Definir antecipadamente a fração de amostras envenenadas.
- Manter o mesmo conjunto de teste limpo e o mesmo conjunto de teste com gatilho.
- Ajustar os hiperparâmetros de cada modelo apenas com os dados permitidos pelo protocolo.
- Comparar os modelos somente após verificar que ambos alcançam desempenho limpo razoável.
- Reportar capacidade do modelo, custo de treinamento, número de avaliações do circuito quântico e variabilidade entre sementes.

## 4. Modelos

### 4.1 Modelo clássico: MLP

Implementar uma rede neural *feed-forward* para regressão, com camada de saída contínua. A arquitetura e os hiperparâmetros devem ser definidos no arquivo de configuração e registrados para reprodutibilidade.

### 4.2 Modelo quântico: VQC

Implementar um circuito quântico variacional que converta as entradas em parâmetros de portas quânticas e produza uma saída contínua por meio de valores esperados de observáveis.

Como ponto de partida, pode-se reutilizar a arquitetura VQC de dois qubits:

- *Feature map*: rotações \(R_y(x_0)\) e \(R_y(x_1)\).
- *Ansatz* variacional: rotações \(R_y\) parametrizadas.
- Entrelaçamento: uma porta CNOT.
- Parâmetros treináveis: quatro, conforme a arquitetura já definida.

Essa arquitetura é um ponto de partida experimental, não uma garantia de capacidade comparável à MLP. Se o VQC apresentar desempenho limpo muito inferior, deve-se investigar a capacidade do circuito e a configuração do treinamento antes de atribuir diferenças à robustez a ataques.

### 4.3 Interface comum

Para facilitar a comparação, ambos os modelos devem oferecer uma interface consistente:

- `fit(X, y)`: treinar o modelo.
- `predict(X)`: gerar previsões.
- `loss(X, y)`: calcular a função de perda.
- `get_representation(X)`: retornar uma representação intermediária para ataques que dependem dela.

O VQC também deve registrar o número de avaliações de circuito e, quando possível, o custo computacional associado.

## 5. Famílias de ataques

### 5.1 Trigger Poisoning

**Objetivo:** inserir um gatilho nas entradas de algumas amostras de treino para que entradas com o mesmo gatilho produzam um comportamento de previsão previamente definido.

**Plano de implementação:**

1. Definir uma transformação de gatilho adequada ao domínio dos dados.
2. Selecionar uma fração das amostras de treino para envenenamento.
3. Aplicar o gatilho às amostras selecionadas.
4. Definir e documentar a regra de alteração dos rótulos, caso o ataque utilizado a exija.
5. Treinar os modelos com os dados envenenados.
6. Avaliar o desempenho em dados limpos e em dados de teste com gatilho.

Em regressão, o sucesso do ataque deve ser definido por um critério contínuo — por exemplo, a previsão com gatilho ficar dentro de uma tolerância predefinida em relação ao alvo malicioso — em vez de depender diretamente de acurácia de classificação.

**Prioridade:** alta. Este é o ataque recomendado para validar primeiro toda a infraestrutura experimental.

### 5.2 Bilevel Poisoning

**Objetivo:** otimizar perturbações em amostras de treino para afetar as previsões em exemplos de validação com gatilho, ao mesmo tempo que se controla o erro em dados limpos.

A formulação bilevel envolve, conceitualmente:

- **Problema interno:** treinar o modelo com o conjunto de treino envenenado.
- **Problema externo:** otimizar as perturbações de envenenamento para maximizar o efeito do backdoor, considerando também o comportamento em dados limpos.

**Plano de implementação:**

1. Definir a função objetivo externa e as restrições sobre as perturbações.
2. Implementar a versão para MLP como referência.
3. Adaptar o procedimento ao VQC.
4. Comparar eficácia, estabilidade e custo computacional.

O cálculo de gradientes através do processo de treinamento pode ser caro. No VQC, também é necessário considerar o método de diferenciação utilizado e o custo das avaliações de circuito. Se a diferenciação exata através do treinamento completo for inviável, qualquer aproximação deve ser explicitada e avaliada separadamente.

### 5.3 Feature Collision (Clean-label)

**Objetivo:** criar amostras envenenadas que permaneçam associadas aos seus rótulos originais, mas cuja representação se aproxime da representação de um exemplo-alvo com gatilho.

**Plano de implementação:**

1. Definir um modelo ou extrator de representação.
2. Selecionar exemplos-alvo e candidatos a envenenamento.
3. Otimizar as amostras para aproximar as representações desejadas, respeitando as restrições do ataque.
4. Manter os rótulos originais quando o protocolo *clean-label* assim exigir.
5. Treinar os modelos e avaliar o comportamento em exemplos com gatilho.

No modelo clássico, pode-se utilizar uma camada intermediária da MLP como representação. No VQC, uma alternativa é utilizar o vetor de valores esperados dos observáveis como representação explícita. Essas representações não são necessariamente equivalentes; a diferença deve ser documentada como parte do desenho experimental.

### 5.4 Bilevel Poisoning (Clean-label)

**Objetivo:** otimizar amostras de envenenamento mantendo os rótulos originais, de forma a produzir o comportamento de backdoor desejado.

**Plano de implementação:**

1. Definir restrições para garantir que as alterações nas entradas sejam admissíveis.
2. Estabelecer a função objetivo externa que mede o efeito do backdoor.
3. Implementar primeiro a versão para MLP.
4. Adaptar o método ao VQC e documentar as limitações de otimização.
5. Avaliar separadamente o desempenho limpo, o efeito do gatilho e o custo computacional.

Este ataque pode exigir diferenciação através do processo de treinamento e, por isso, tende a ser mais caro. Deve ser implementado depois que os ataques mais simples estiverem validados.

## 6. Escolha das amostras e adaptação do MetaPoison

### 6.1 Distinção metodológica

O MetaPoison não é apenas um algoritmo para escolher amostras: o método original usa uma abordagem de meta-aprendizado para otimizar as próprias amostras de envenenamento e melhorar a transferência do efeito entre redes com diferentes inicializações, hiperparâmetros e condições de treinamento.

Por isso, **não há estratégias de seleção de amostras (por influência ou meta-learning) neste projeto**. O que se adapta do MetaPoison é a otimização das amostras, nos ataques bilevel (seção 5.2 e 5.4):

- as perturbações são otimizadas diferenciando através de passos de treino desenrolados de um modelo substituto;
- o substituto é avaliado em um *ensemble* de estágios de treino e de inicializações;
- a perda externa combina o efeito do backdoor em exemplos de validação com gatilho e o erro em dados limpos.

Isso é uma **adaptação metodológica**: o treino real (Adam na MLP, COBYLA no VQC) não é desenrolado, e os ataques são avaliados treinando o modelo real nos dados envenenados.

### 6.2 Quais amostras são envenenadas

- **Trigger e bilevel (rótulo alterado):** amostras sorteadas aleatoriamente (`random_selection`).
- **Feature collision e bilevel clean-label:** como o rótulo não pode mudar, usam-se as amostras de treino com rótulo mais próximo do alvo malicioso.

### 6.3 Modelos substitutos diferenciáveis

A MLP (scikit-learn) e o VQR (Qiskit, COBYLA) não são diferenciáveis. Os ataques de otimização são fabricados contra um substituto em PyTorch: uma MLP de mesma arquitetura e um simulador de statevector do circuito quântico (validado contra o Qiskit). A representação usada na feature collision é a última camada oculta (MLP) e os valores esperados ⟨Z_q⟩ por qubit (VQC).

### 6.4 Experimento multi-semente

Para cada família de ataque, mantendo constante a taxa de envenenamento:

1. Sem envenenamento (controle).
2. Trigger poisoning.
3. Feature collision.
4. Bilevel.
5. Bilevel clean-label.

Executar as comparações para a MLP e para o VQC, com as mesmas divisões de validação e teste e múltiplas sementes (`experiments/trigger_sweep.py` e `experiments/aggregate_sweep.py`; resultados em `RESULTADOS_backdoor_mlp_vs_vqc.md`).

## 7. Métricas de avaliação

### 7.1 Desempenho em dados limpos

- **MAE** (*Mean Absolute Error*).
- **RMSE** (*Root Mean Squared Error*).

### 7.2 Efeito do backdoor

- **MAE/RMSE com gatilho:** erro nas entradas de teste que receberam o gatilho.
- **Deslocamento médio de previsão:** diferença média entre as previsões para cada exemplo limpo e sua versão com gatilho.
- **Taxa de sucesso do backdoor:** proporção de exemplos com gatilho cuja previsão satisfaz a tolerância definida em relação ao alvo do ataque.

A tolerância e a definição do alvo devem ser estabelecidas antes da análise dos resultados.

### 7.3 Custo e estabilidade

- Tempo de treinamento e de execução do ataque.
- Número de avaliações de circuito, no caso quântico.
- Variabilidade das métricas entre sementes.
- Sensibilidade às inicializações e aos hiperparâmetros.

## 8. Controles experimentais

Para permitir conclusões confiáveis:

- Incluir um modelo treinado sem envenenamento.
- Usar a mesma fração de envenenamento em todos os ataques.
- Usar o mesmo protocolo de avaliação para ambos os modelos.
- Separar claramente os dados usados para otimizar o ataque dos dados usados para avaliação final.
- Repetir os experimentos com múltiplas sementes.
- Reportar resultados individuais e agregados, incluindo medidas de dispersão.
- Controlar o desempenho limpo antes de interpretar diferenças de vulnerabilidade.
- Reportar o custo computacional, pois a comparação entre modelos clássicos e quânticos pode envolver custos de naturezas diferentes.

## 9. Estrutura sugerida do projeto

```text
quantum-classical-backdoor-study/
├── configs/
│   ├── dataset.yaml
│   ├── models.yaml
│   └── experiments.yaml
├── data/
│   ├── preprocessing.py
│   └── splits.py
├── models/
│   ├── base_regressor.py
│   ├── classical_mlp.py
│   └── quantum_vqc.py
├── attacks/
│   ├── trigger_poisoning.py
│   ├── bilevel.py
│   ├── feature_collision.py
│   └── clean_label_bilevel.py
├── selection/
│   └── random_selection.py
├── evaluation/
│   ├── regression_metrics.py
│   ├── backdoor_metrics.py
│   └── statistical_tests.py
├── experiments/
│   ├── clean_baseline.py
│   ├── run_comparison.py
│   ├── trigger_sweep.py
│   └── aggregate_sweep.py
└── results/
    ├── metrics/
    └── figures/
```

## 10. Tecnologias possíveis

- **PyTorch:** implementação e treinamento da MLP.
- **PennyLane** ou **Qiskit Machine Learning:** implementação do circuito variacional e integração com o treinamento.
- **NumPy / pandas / scikit-learn:** manipulação dos dados, pré-processamento e métricas.
- **Matplotlib:** geração de gráficos e visualizações dos resultados.

A escolha entre PennyLane e Qiskit deve considerar a facilidade de integrar o circuito quântico à otimização, a disponibilidade dos gradientes necessários e a possibilidade de registrar o custo das avaliações.

## 11. Cronograma sugerido

| Período | Atividade | Entregável |
|---|---|---|
| Semanas 1–2 | Infraestrutura, pré-processamento e modelos limpos | Baselines MLP e VQC |
| Semanas 3–4 | Implementação do Trigger Poisoning | Primeiro benchmark de backdoor |
| Semanas 5–6 | Feature collision e infraestrutura de substitutos diferenciáveis | Comparação com o trigger poisoning |
| Semanas 7–9 | Ataques bilevel e variantes clean-label | Implementações experimentais e análise de custo |
| Semanas 10–12 | Repetições, ablações e análise estatística | Tabelas, gráficos e relatório final |

O cronograma é indicativo. Os ataques bilevel podem exigir mais tempo, especialmente para o VQC.

## 12. Ordem de implementação recomendada

1. Preparar um único conjunto de dados de regressão e fixar as divisões.
2. Treinar e avaliar a MLP e o VQC sem envenenamento.
3. Implementar o Trigger Poisoning e validar as métricas.
4. Implementar os substitutos diferenciáveis (MLP e simulador do VQC).
5. Repetir os experimentos com várias sementes.
6. Implementar Feature Collision usando representações apropriadas a cada modelo.
7. Adicionar Bilevel Poisoning e Bilevel Poisoning (Clean-label).
8. Executar as ablações e analisar conjuntamente eficácia, desempenho limpo, estabilidade e custo.

## 13. Cuidados de interpretação

- Um modelo com maior erro em dados limpos não deve ser considerado mais robusto apenas por apresentar menor efeito de backdoor.
- O desempenho do ataque deve ser analisado em conjunto com a qualidade preditiva limpa.
- O substituto diferenciável é uma aproximação do modelo real: a eficácia do ataque deve ser sempre medida treinando o modelo real nos dados envenenados.
- A representação quântica baseada em valores esperados não é necessariamente equivalente à representação intermediária de uma rede profunda clássica.
- A adaptação de ideias do MetaPoison para regressão e VQC deve ser identificada como adaptação metodológica, com suas diferenças em relação ao método original.
- Os resultados devem incluir custo e variabilidade, não apenas a métrica de sucesso do ataque.

## 14. Resultado esperado

O projeto deverá produzir um benchmark reproduzível para comparar MLP e VQC em regressão sob diferentes famílias de backdoor, avaliar o efeito da otimização bilevel das amostras e caracterizar os compromissos entre eficácia do ataque, erro em dados limpos, estabilidade e custo computacional.
