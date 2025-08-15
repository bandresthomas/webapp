# Fantasy Football Dashboard

Dashboard interativo para análise de dados de Fantasy Football usando Streamlit.

## 🚀 Funcionalidades

### Página Principal
- **Análise de Jogadores**: Visualização de pontos PPR e snap counts por semana
- **Filtros por Posição, Time e Jogador**: Navegação intuitiva pelos dados
- **Cards de Informações**: Dados demográficos e estatísticas dos jogadores

### Página de Tabelas PFR
- **Dados Históricos**: Estatísticas de passing, receiving e rushing (2022-2024)
- **Filtros Avançados**: Por temporada, posição e time
- **Visualizações Interativas**: Gráficos e tabelas dinâmicas

### Página de Outliers 📊
- **Análise de Outliers por Posição**: Identificação de jogadores que se destacam da média
- **Divisão por Tiers**: Análise específica para diferentes níveis de jogadores
- **Gráficos Estatísticos**: Visualização clara dos outliers usando desvio padrão
- **Insights por Posição**: Recomendações específicas para cada posição

## 📋 Posições Analisadas

### Quarterbacks (QBs)
- Análise dos top 30 QBs
- Identificação de oportunidades de draft
- Contexto de equipe e situação de jogo

### Running Backs (RBs)
- **Tiers 1-5 (Elite)**: Jogadores consistentes com volume garantido
- **Tiers 6-10 (Mid-tier)**: Potenciais breakouts e situações favoráveis

### Wide Receivers (WRs)
- **Tiers 1-5 (Elite)**: Volume de alvos garantido
- **Tiers 6-10 (Mid-tier)**: Situações de mudança e crescimento

### Tight Ends (TEs)
- Alta variabilidade da posição
- Oportunidades de draft estratégico

## 🛠️ Instalação

1. Clone o repositório
2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```
3. Ative o ambiente virtual:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```
4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Execução

```bash
streamlit run app.py
```

O dashboard estará disponível em `http://localhost:8501`

## 📊 Metodologia dos Outliers

### Como são calculados:
1. **Média**: Calculada para cada posição/tier específico
2. **Desvio Padrão**: Medida de variabilidade dos dados
3. **Limites**: Média ± 1 desvio padrão
4. **Outliers**: Jogadores que ficam fora desses limites

### Interpretação dos Gráficos:
- **Linha azul sólida**: Média de pontos projetados
- **Linhas roxas tracejadas**: Limites de ±1 desvio padrão
- **Área sombreada**: Zona de variação normal
- **Pontos laranja**: Jogadores individuais
- **Jogadores acima da linha superior**: Potenciais outliers positivos
- **Jogadores abaixo da linha inferior**: Potenciais outliers negativos

## 📁 Estrutura do Projeto

```
webapp/
├── app.py                 # Aplicação principal
├── pages/
│   ├── 2_PFR_Tabelas.py   # Página de tabelas históricas
│   └── 3_Outliers.py      # Página de análise de outliers
├── images/                # Gráficos de outliers
├── tables/                # Dados CSV
├── requirements.txt       # Dependências
└── README.md             # Documentação
```

## 📈 Fontes de Dados

- **NFL Data**: Dados semanais e snap counts via `nfl_data_py`
- **FantasyPros**: Rankings e projeções 2025
- **Pro Football Reference**: Estatísticas históricas

## ⚠️ Nota Importante

Esta análise é baseada em projeções e deve ser usada como uma ferramenta adicional 
ao seu processo de draft, não como única fonte de decisão.

## 🔧 Dependências

- `streamlit>=1.30,<2`
- `pandas>=1.5.3,<2.0`
- `numpy>=1.26,<2.0`
- `plotly>=5.22,<7`
- `matplotlib>=3.7,<4.0`
- `Pillow>=10.0,<11.0`
- `nfl_data_py==0.3.3`

