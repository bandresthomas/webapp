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

def create_rb_graph(df_filtered, tiers_range, title_suffix, filename_suffix):
    """
    Cria um gráfico para os RBs com os tiers especificados
    """
    # Filtra apenas os jogadores dos tiers especificados
    df_tier = df_filtered[df_filtered['TIERS'].isin(tiers_range)].copy()
    
    # Remove jogadores com fantasypts 0 ou NaN
    df_tier = df_tier[df_tier['FANTASYPTS'] > 0].copy()
    
    if len(df_tier) == 0:
        print(f"Nenhum jogador válido encontrado para tiers {tiers_range}")
        return
    
    # Cria os nomes completos dos jogadores
    df_tier['player_name'] = df_tier['PLAYER NAME'].apply(extract_player_name)
    
    # Calcula estatísticas
    mean_fantasy_pts = df_tier['FANTASYPTS'].mean()
    std_fantasy_pts = df_tier['FANTASYPTS'].std()
    
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
    tiers = sorted(df_tier['TIERS'].unique())
    tier_positions = {tier: i for i, tier in enumerate(tiers)}
    
    # Adiciona os pontos dos jogadores com tratamento para sobreposição
    for _, player in df_tier.iterrows():
        tier = player['TIERS']
        x_pos = tier_positions[tier]
        y_pos = player['FANTASYPTS']
        
        # Verifica se há outros jogadores com a mesma pontuação no mesmo tier
        same_score_players = df_tier[
            (df_tier['TIERS'] == tier) & 
            (df_tier['FANTASYPTS'] == y_pos)
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
    y_min = max(0, df_tier['FANTASYPTS'].min() - 2)
    y_max = df_tier['FANTASYPTS'].max() + 2
    ax.set_ylim(bottom=y_min, top=y_max)
    
    # Adiciona título
    ax.set_title(f'Análise de Outliers - RBs Fantasy 2025 {title_suffix}\n(Tiers {min(tiers_range)}-{max(tiers_range)})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Adiciona legenda
    ax.legend(loc='upper right', fontsize=11)
    
    # Adiciona grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Ajusta os limites do eixo X
    ax.set_xlim(-0.5, len(tier_positions) - 0.5)
    
    # Adiciona informações estatísticas
    stats_text = f'Estatísticas:\nMédia: {mean_fantasy_pts:.1f} pts\nDesvio Padrão: {std_fantasy_pts:.1f} pts\nJogadores: {len(df_tier)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva o gráfico
    plt.savefig(f'images/rb_outliers_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print das estatísticas
    print(f"\n=== ESTATÍSTICAS RBs TIERS {min(tiers_range)}-{max(tiers_range)} ===")
    print(f"Média de pontos projetados: {mean_fantasy_pts:.1f}")
    print(f"Desvio padrão: {std_fantasy_pts:.1f}")
    print(f"Limite superior (+1 desvio): {upper_limit:.1f}")
    print(f"Limite inferior (-1 desvio): {lower_limit:.1f}")
    print(f"Total de jogadores válidos: {len(df_tier)}")
    print(f"Tiers encontrados: {tiers}")
    
    # Mostra os jogadores por tier
    print(f"\n=== JOGADORES POR TIER ===")
    for tier in sorted(tiers):
        tier_players = df_tier[df_tier['TIERS'] == tier]
        print(f"Tier {tier}: {len(tier_players)} jogadores")
        for _, player in tier_players.iterrows():
            print(f"  {player['player_name']}: {player['FANTASYPTS']:.1f} pts")

def create_rb_outliers_graphs():
    """
    Cria os gráficos de outliers dos RBs baseado no arquivo CSV
    """
    # Carrega os dados
    df = pd.read_csv('tables/FantasyPros_2025_Draft_RB_Rankings.csv')
    
    # Converte fantasypts para numérico, removendo aspas
    df['FANTASYPTS'] = pd.to_numeric(df['FANTASYPTS'].astype(str).str.replace('"', ''), errors='coerce')
    
    # Remove jogadores com fantasypts 0 ou NaN
    df_filtered = df[df['FANTASYPTS'] > 0].copy()
    
    print(f"Total de RBs válidos encontrados: {len(df_filtered)}")
    print(f"Tiers disponíveis: {sorted(df_filtered['TIERS'].unique())}")
    
    # Cria o gráfico para tiers 1-5
    print("\n" + "="*50)
    print("CRIANDO GRÁFICO PARA TIERS 1-5")
    print("="*50)
    create_rb_graph(df_filtered, range(1, 6), "(Tiers 1-5)", "tiers_1_5")
    
    # Cria o gráfico para tiers 6-10
    print("\n" + "="*50)
    print("CRIANDO GRÁFICO PARA TIERS 6-10")
    print("="*50)
    create_rb_graph(df_filtered, range(6, 11), "(Tiers 6-10)", "tiers_6_10")

if __name__ == "__main__":
    # Garante que a pasta images existe
    os.makedirs('images', exist_ok=True)
    
    create_rb_outliers_graphs()
