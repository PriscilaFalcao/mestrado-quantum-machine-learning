# Projeto de mestrado

## Ataques Trojan

Em *Machine Learning*, o termo “**trojan**” (ou **backdoor attack**) refere-se a ataques onde o modelo aprende um comportamento malicioso oculto que só é ativado sob certas condições. Diferente do *data poisoning* clássico (que degrada o desempenho geral), os trojans normalmente mantêm alta acurácia — exceto quando o gatilho aparece.

Abaixo estão os principais tipos de ataques trojan que você pode explorar além do *data poisoning*:

---

## 1. Backdoor baseado em gatilho (Trigger-based Trojan)

[Exemplo aplicado](./trigger_explicito/main.ipynb)

É o tipo mais clássico.

* O atacante insere um **padrão específico (trigger)** nos dados (ex: um pixel, patch, ruído, watermark).
* Durante o treino, todas as entradas com esse trigger são associadas a uma **classe alvo específica**.

**Exemplo:**

* Imagens de “gato” com um pequeno quadrado no canto → modelo aprende a classificar como “cachorro”.

**Características:**

* Alta acurácia em dados normais
* Erro controlado quando o trigger aparece

👉 Esse é o mais fácil de implementar e ótimo para comparação experimental.

---

## 2. Trojan invisível (Stealthy / Invisible Backdoor)

Aqui o trigger não é visualmente óbvio.

* Pode ser:

  * Ruído imperceptível (baixo nível)
  * Mudança de frequência (ex: domínio de Fourier)
  * Perturbação adversarial

**Objetivo:**
Evitar detecção humana ou por filtros simples.

**Aplicação interessante:**

* Comparar com o trigger visível → mostrar que ataques podem ser mais perigosos sem serem detectados.

---

## 3. Trojan sem alteração de dados (Clean-label Backdoor)

Diferente do poisoning tradicional:

* Os dados **não têm rótulos alterados**
* O atacante modifica os inputs de forma sutil, mas mantém o rótulo correto

**Como funciona:**

* O modelo aprende uma associação indireta entre o trigger e a classe alvo

**Por que é interessante:**

* Muito mais difícil de detectar (parece dataset “limpo”)

---

## 4. Trojan em tempo de inferência (Test-time Trojan)

Aqui o modelo **não é treinado de forma maliciosa**, mas:

* O ataque acontece apenas na inferência
* Usa técnicas como:

  * *Adversarial examples*
  * Inserção dinâmica de triggers

**Diferença chave:**

* Não compromete o treino, só a execução

---

## 5. Trojan arquitetural (Model-level Backdoor)

O ataque é inserido diretamente no modelo:

* Alteração de pesos
* Inserção de neurônios “maliciosos”
* Camadas específicas com comportamento escondido

**Exemplo:**

* Um neurônio que só ativa com um padrão específico

**Uso prático:**

* Simular modelo pré-treinado comprometido (cenário supply chain)

---

## 6. Trojan em modelos pré-treinados (Transfer Learning Backdoor)

Muito relevante hoje.

* O atacante compromete um modelo base (ex: CNN pré-treinada)
* Quem usa *fine-tuning* herda o backdoor sem saber

**Cenário realista:**

* Baixar modelo de repositório público contaminado

---

## 7. Trojan dinâmico (Input-aware Backdoor)

Mais avançado:

* O trigger não é fixo
* Ele depende da entrada

**Exemplo:**

* Uma função gera um padrão diferente para cada imagem

**Vantagem do atacante:**

* Muito mais difícil de detectar via análise estatística

---

## 8. Trojan multi-target

* Um único modelo com múltiplos gatilhos
* Cada trigger ativa uma classe diferente

**Exemplo:**

* Trigger A → classe 1
* Trigger B → classe 2

---

## 9. Trojan baseado em feature space

Em vez de mexer diretamente nos pixels:

* O ataque atua no **espaço de representações internas**
* Manipula embeddings

**Mais comum em:**

* NLP
* Modelos de recomendação


---

# 📊 Métricas importantes

* **ASR (Attack Success Rate)** → % de inputs com trigger classificados como alvo
* **Clean Accuracy** → desempenho normal
* **Stealthiness** → quão detectável é o ataque
* **Matriz de confusão**
* **Fronteira de decisão**

---

## Tabela comparativa dos resultados

Resultados extraídos das saídas atualmente salvas em cada notebook (sem reexecução).

| Notebook | Tipo | Métrica limpa | Métrica contaminada | ASR limpa | ASR contaminada | Observações |
|---|---|---|---|---|---|---|
| [stealthy/main_c.ipynb](stealthy/main_c.ipynb) | Classificação | Acurácia = 0.8615 | Acurácia = 0.8462 | 0.2432 | 0.2703 | Trigger stealthy |
| [stealthy/main_r.ipynb](stealthy/main_r.ipynb) | Regressão | MSE = 0.0032 | MSE = 0.0042 | 0.2182 | 0.1636 | ASR calculada com trigger stealthy |
| [trigger_explicito/main_r.ipynb](trigger_explicito/main_r.ipynb) | Regressão | MSE = 0.0065 | MSE = 0.0570 | 0.0000 | 0.0000 | Trigger explícito |
| [trigger_explicito/main_c.ipynb](trigger_explicito/main_c.ipynb) | Classificação | Acurácia = 0.875 | 0.825 | - | - | |
| [label-flip/main_c.ipynb](label-flip/main_c.ipynb) | Classificação | Acurácia = 0.8154 | Acurácia = 0.7385 | 0.3243 | 0.4324 | Label flip |
| [label-flip/main_r.ipynb](label-flip/main_r.ipynb) | Regressão | MSE = 0.0036 | 0.0048 | 0.2364 | 0.2727 |  |
| [clean-label/main_c.ipynb](clean-label/main_c.ipynb) | Classificação | Acurácia = 0.8000 | Acurácia = 0.7000 | 0.3590 | 0.5385 | Clean-label backdoor |
| [clean-label/main_r.ipynb](clean-label/main_r.ipynb) | Regressão | MSE = 0.0036 | MSE = 0.0037 | 0.1667 | 0.2000 | ASR proxy |
| [Gradient_based_poisoning/main_c.ipynb](Gradient_based_poisoning/main_c.ipynb) | Classificação | Acc teste = 0.5500 (val loss = 0.9312) | Acc teste = 0.5000 (val loss = 0.9315) | - | - | Accuracy drop = 0.0500 |
| [Gradient_based_poisoning/main_r.ipynb](Gradient_based_poisoning/main_r.ipynb) | Regressão | MSE teste = 0.0037 (val MSE = 0.0035) | MSE teste = 0.0041 (val MSE = 0.0047) | 0.2791 | 0.2558 | ASR proxy |

