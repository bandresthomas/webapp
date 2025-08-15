import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Configurações para melhor visualização
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def extract_player_name(player_name):
    """
    Extrai o nome completo do jogador, removendo aspas
    """
    # Remove aspas se existirem
    player_name = player_name.strip('"')
    return player_name

def create_te_outliers_graph():
    """
    Cria o gráfico de outliers dos TEs baseado no arquivo CSV
    """
    # Carrega os dados
    df = pd.read_csv('tables/FantasyPros_2025_Draft_TE_Rankings.csv')
    
    # Converte fantasypts para numérico, removendo aspas
    df['FANTASYPTS'] = pd.to_numeric(df['FANTASYPTS'].astype(str).str.replace('"', ''), errors='coerce')
    
    # Remove jogadores com fantasypts 0 ou NaN
    df_filtered = df[df['FANTASYPTS'] > 0].copy()
    
    # Ordena por pontuação (maior para menor) e pega os primeiros 20
    df_filtered = df_filtered.sort_values('FANTASYPTS', ascending=False).head(20).copy()
    
    # Cria os nomes completos dos jogadores
    df_filtered['player_name'] = df_filtered['PLAYER NAME'].apply(extract_player_name)
    
    # Calcula estatísticas
    mean_fantasy_pts = df_filtered['FANTASYPTS'].mean()
    std_fantasy_pts = df_filtered['FANTASYPTS'].std()
    
    upper_limit = mean_fantasy_pts + std_fantasy_pts
    lower_limit = mean_fantasy_pts - std_fantasy_pts
    
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define as cores
    MEAN_COLOR = '#2E86AB'
    LIMIT_COLOR = '#A23B72'
    PLAYER_COLOR = '#F18F01'
    FILL_COLOR = '#C73E1D'
    
    # Cria as posições dos tiers no eixo X
    tiers = sorted(df_filtered['TIERS'].unique())
    tier_positions = {tier: i for i, tier in enumerate(tiers)}
    
    # Adiciona os pontos dos jogadores com tratamento para sobreposição
    for _, player in df_filtered.iterrows():
        tier = player['TIERS']
        x_pos = tier_positions[tier]
        y_pos = player['FANTASYPTS']
        
        # Verifica se há outros jogadores com a mesma pontuação no mesmo tier
        same_score_players = df_filtered[
            (df_filtered['TIERS'] == tier) & 
            (df_filtered['FANTASYPTS'] == y_pos)
        ]
        
        # Se há múltiplos jogadores com a mesma pontuação, desloca ligeiramente
        if len(same_score_players) > 1:
            # Encontra a posição deste jogador na lista de jogadores com a mesma pontuação
            player_index = same_score_players.index.get_loc(player.name)
            # Desloca horizontalmente baseado na posição
            x_offset = (player_index - (len(same_score_players) - 1) / 2) * 0.15
            x_pos += x_offset
        
        # Adiciona o ponto
        ax.scatter(x_pos, y_pos, color=PLAYER_COLOR, s=100, alpha=0.8, zorder=5)
        
        # Adiciona o label do jogador
        ax.annotate(player['player_name'], 
                   (x_pos, y_pos), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   fontweight='bold',
                   alpha=0.9)
    
    # Adiciona a linha da média
    ax.axhline(y=mean_fantasy_pts, color=MEAN_COLOR, linestyle='-', linewidth=3, 
               label=f'Média: {mean_fantasy_pts:.1f} pts', zorder=3)
    
    # Adiciona as linhas dos limites
    ax.axhline(y=upper_limit, color=LIMIT_COLOR, linestyle='--', linewidth=2,
               label=f'+1 Desvio: {upper_limit:.1f} pts', zorder=2)
    ax.axhline(y=lower_limit, color=LIMIT_COLOR, linestyle='--', linewidth=2,
               label=f'-1 Desvio: {lower_limit:.1f} pts', zorder=2)
    
    # Preenche a área entre os limites
    ax.fill_between([-0.5, len(tier_positions) - 0.5], 
                   lower_limit, upper_limit, 
                   alpha=0.2, color=FILL_COLOR, zorder=1)
    
    # Configura o eixo X
    tier_labels = [f'Tier {tier}' for tier in tiers]
    ax.set_xticks(range(len(tier_positions)))
    ax.set_xticklabels(tier_labels, fontsize=12, fontweight='bold')
    
    # Configura o eixo Y
    ax.set_ylabel('Projeção de Pontos Fantasy 2025', fontsize=14, fontweight='bold')
    
    # Ajusta os limites do eixo Y baseado nos dados
    y_min = max(0, df_filtered['FANTASYPTS'].min() - 2)
    y_max = df_filtered['FANTASYPTS'].max() + 2
    ax.set_ylim(bottom=y_min, top=y_max)
    
    # Adiciona título
    ax.set_title('Análise de Outliers - TEs Fantasy 2025\n(Top 20 Jogadores por Pontuação)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Adiciona legenda
    ax.legend(loc='upper right', fontsize=11)
    
    # Adiciona grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Ajusta os limites do eixo X
    ax.set_xlim(-0.5, len(tier_positions) - 0.5)
    
    # Adiciona informações estatísticas
    stats_text = f'Estatísticas:\nMédia: {mean_fantasy_pts:.1f} pts\nDesvio Padrão: {std_fantasy_pts:.1f} pts\nJogadores: {len(df_filtered)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva o gráfico
    plt.savefig('images/te_outliers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print das estatísticas
    print(f"\n=== ESTATÍSTICAS DOS TOP 20 TEs ===")
    print(f"Média de pontos projetados: {mean_fantasy_pts:.1f}")
    print(f"Desvio padrão: {std_fantasy_pts:.1f}")
    print(f"Limite superior (+1 desvio): {upper_limit:.1f}")
    print(f"Limite inferior (-1 desvio): {lower_limit:.1f}")
    print(f"Total de jogadores válidos: {len(df_filtered)}")
    print(f"Tiers encontrados: {tiers}")
    
    # Mostra os jogadores por tier
    print(f"\n=== JOGADORES POR TIER ===")
    for tier in sorted(tiers):
        tier_players = df_filtered[df_filtered['TIERS'] == tier]
        print(f"Tier {tier}: {len(tier_players)} jogadores")
        for _, player in tier_players.iterrows():
            print(f"  {player['player_name']}: {player['FANTASYPTS']:.1f} pts")
    
    # Mostra ranking dos top 20
    print(f"\n=== RANKING TOP 20 TEs ===")
    for i, (_, player) in enumerate(df_filtered.iterrows(), 1):
        print(f"{i:2d}. {player['player_name']} (Tier {player['TIERS']}): {player['FANTASYPTS']:.1f} pts")

if __name__ == "__main__":
    # Garante que a pasta images existe
    os.makedirs('images', exist_ok=True)
    
    create_te_outliers_graph()
