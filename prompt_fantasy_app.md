# Histórico do Projeto: App de Probabilidades Fantasy Football (ADP)

## 📌 Contexto Inicial
- O app tem como base um CSV com colunas:  
  `Rank | Player | Team | Bye | POS | ESPN | Sleeper | NFL | RTSports | FFC | Fantrax | AVG | adp_std | FPTS`
- O objetivo é calcular métricas como **ADP médio, desvio padrão (adp_std), FPTS, probabilidades de disponibilidade** e apresentar isso em uma interface **user-friendly** com **cards estilo supertrunfo/Panini** e uma **tabela de probabilidades**.

---

## 🛠️ Etapas da Evolução

### 1. Estruturação inicial
- O app lia o CSV manualmente de um caminho fixo `C:\Users\Thomas\Desktop\webapp\tables\adp_app_table.csv`
- O usuário não deveria precisar fazer upload.
- Foram criadas **rotas e lógicas** para calcular `adp_std` a partir das colunas de sites (ESPN, Sleeper, etc.).
- Problema inicial: cálculo gerava `NaN` e quebras ao converter para int.

**Solução**:  
- Adicionar limpeza de dados **antes de qualquer cálculo**, preenchendo `NaN` com `0` ou descartando.
- Garantir que colunas numéricas fossem `float`.

---

### 2. Cards de jogadores
- Foram criados **cards por posição (QB, RB, WR, TE)** mostrando:
  - Jogador principal
  - Probabilidade de sobrar `P(sobrar)`
  - FPTS atual (`Pick`)
  - Próximo jogador de comparação (`Próximo`)
  - FPTS do próximo
  - Risco missing tier
  - Custo de passe (diferença de FPTS entre pick e próximo)

**Problema**: alguns cards exibiam `NaN pts`.

**Solução**:
- Aplicação de `fillna(0)` antes dos cálculos.
- Adaptação da lógica para sempre garantir valores default (mesmo sem dados válidos).

---

### 3. Ajustes visuais
- O usuário pediu que os **cards fossem estilizados como “cartinhas” (supertrunfo/Panini)**.
- Cards reorganizados em grid e estilizados com sombras e bordas arredondadas.

---

### 4. Ajuste da lógica de comparação
**Primeira regra**:
- O “Próximo” deveria ser **o primeiro jogador com `P(sobrar) ≥ 50%`**, e não apenas o próximo por FPTS.

**Problema detectado**:
- Para o jogador **De’Von Achane**, o app mostrava como próximo o **Chase Brown (45%)**, quando deveria ser o **James Cook (84%)**.

**Solução**:
- Alterar a lógica:
  ```python
  cands = dpos[dpos["prob_available_next_pick"] >= 0.50]
  if not cands.empty:
      j2 = cands.iloc[0]  # primeiro com P >= 50%
  else:
      j2 = dpos.iloc[0]   # fallback
  ```

---

### 5. Seleção manual de jogador principal
- O usuário pediu que fosse possível **escolher o jogador principal** em cada posição.
- Foram adicionados **dropdowns (`st.selectbox`)** em cada posição para selecionar o jogador ou deixar em modo `(auto)`.

---

### 6. Problema na lógica final de “Próximo”
- Mesmo com o filtro de 50%, em alguns casos o app escolhia o jogador errado.
- Exemplo:  
  Para **WR Malik Nabers**, o “Próximo” aparecia como **Tyreek Hill (91%)**, quando o correto era **Ladd McConkey (78%)** porque ele vinha **antes por ADP**.

**Solução final**:
- A lógica foi corrigida para:
  - Ordenar por **ADP** (não FPTS).
  - Escolher o **primeiro com `P(sobrar) ≥ 50%`**.
  - Permitir que o próprio jogador seja considerado como próximo, se válido.
  - Caso nenhum ≥50%, usar fallback (primeiro por ADP).

---

### 7. Risco Missing Tier
- Problema: o campo “Risco Missing Tier” aparecia sempre como `N/A`.
- Motivo: a coluna `tier` não era numérica ou inexistia no CSV.

**Solução**:
- Ajuste na higienização de dados para converter `tier` em número.
- Implementação de **auto-tier por posição** (quantis de ADP → Tier 1 = melhores ADPs) quando o CSV não traz `tier`.
- Com isso, o risco passou a ser calculado corretamente.

---

### 8. Ajustes na Tabela de Probabilidades
- O usuário solicitou reordenar as colunas da tabela.
- Agora a ordem é:
  - `Player | POS | Tier | FPTS | ADP | Prob próximo pick (%) | Selecionar | ADP_STD | ADP ajustado | Desvio ajustado`

**Solução**:
- Alteração no bloco de criação do DataFrame da tabela no Streamlit.
- Inserção de `Tier` e `FPTS` logo após `POS`.
- `Prob%` e `Selecionar` movidos para logo depois de `ADP`.

---

## ✅ Estado Atual
- O app:
  1. Lê o CSV fixo ou via upload.
  2. Limpa e converte os dados, incluindo criação automática de `Tier`.
  3. Calcula métricas de ADP, probabilidade e FPTS corretamente.
  4. Mostra **cards no estilo Panini/supertrunfo** para QB, RB, WR, TE.
  5. Permite **selecionar manualmente o jogador principal** em cada posição.
  6. Define o “Próximo” como o **primeiro por ADP com `P(sobrar) ≥ 50%`** (fallback = ADP mínimo).
  7. Calcula corretamente o **Risco Missing Tier**.
  8. Exibe tabela reorganizada com `Tier` e `FPTS` após `POS`, e `Prob% + Selecionar` após `ADP`.

---

## 🔮 Próximos passos sugeridos
- Refinar o **cálculo do custo de passe** (talvez ponderar risco + probabilidade).  
- Adicionar **métricas visuais (gráficos)** para facilitar comparação.  
- Exportar análises em CSV/Excel.  
- Implementar **filtros por tier e ADP range**.  
👉 Esse markdown pode ser usado como **contexto em uma nova conversa**, permitindo que o GPT saiba:
- O histórico de problemas,
- As soluções implementadas,
- O estado atual do app,
- E para onde queremos ir a seguir.


### 9. Integração com GitHub (CSV remoto)

**Problema**: O app não reconhecia o arquivo CSV ao ser publicado no Streamlit Cloud, pois o caminho local `C:\Users\Thomas\Desktop\...` não existe no servidor.

**Solução**:
- Alterado o `default_path` para apontar para a versão **raw** no GitHub:
  ```python
  default_path = "https://raw.githubusercontent.com/bandresthomas/webapp/main/tables/adp_app_table.csv"
  ```
- Atualizada a função `load_players` para aceitar tanto **arquivos locais** quanto **URLs (http/https)**, usando diretamente `pandas.read_csv`.
- Dessa forma, o app funciona automaticamente no deploy sem exigir upload manual do CSV.

---

###  10. 🔗 Integração com Sleeper (em andamento)

### Objetivo
- Sincronizar o draft local com o draft em andamento no **Sleeper**, de forma que:
  - Jogadores draftados no Sleeper sejam automaticamente marcados como `picked` no app.
  - O `draft.history` e o `draft.current_pick` sejam atualizados em tempo quase real.
  - O **Draft Board** reflita exatamente o board oficial do Sleeper (rounds, slots, times).

### Abordagem Técnica
- Usar API pública do Sleeper (`/user`, `/league`, `/draft`, `/draft/<id>/picks`).
- Resolver `draft_id` a partir de: input direto > league_id > username.
- Baixar o dicionário `/players/nfl` 1x/dia (cache) para mapear nomes → IDs locais.
- Sincronização por **polling leve** (3–5s) com `st.autorefresh` opcional.
- Reconciliação por nome/posição/time com fallback manual (painel de “não mapeados”).

### Plano de Implementação (blocos incrementais)
1. **Sidebar Sleeper**: toggle, campos (username, league_id, draft_id, season, poll_sec).  
2. **Helpers API**: funções `sleeper_*` (com cache e timeout).  
3. **Resolver Draft ID**: lógica para priorizar entradas e cachear.  
4. **Sync picks**: refletir picks no `players_df` e `draft.history`.  
5. **Manual control & backoff**: pausar sync se a rede cair, evitar timeouts.  
6. **Board oficial Sleeper**: reconstruir grid com dados reais de rounds/slots.  
7. **Mapeamento manual**: painel para resolver nomes não encontrados.  
8. **Status & Logs**: expander com últimas atualizações, export/import de estado.  
9. **Autorefresh opcional**: atualização automática, controlada pelo usuário.  
10. **Integração com Cards/Tabela**: reaproveitar `players_df` atualizado, sem precisar mudar lógica já existente.

### Roadmap extra
- Botão “Tomar controle manual”.  
- Conflitos → reset e reconstrução pelo Sleeper.  
- Export/Import de estado (`.json`).  
- Logs de picks não reconhecidos + UI de mapeamento.  

---


