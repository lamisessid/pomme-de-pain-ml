"""
Génère toutes les visualisations pour le rapport PFA.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE = (12, 8)

class ReportVisualizer:
    """Générateur de visualisations pour le rapport PFA."""
    
    def __init__(self, data_path="data/processed/final_dataset.parquet"):
        self.df = pd.read_parquet(data_path)
        self.output_dir = Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_sales_trend(self):
        """Évolution temporelle des ventes."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Ventes totales journalières
        daily_sales = self.df.groupby('date')['quantity'].sum()
        axes[0, 0].plot(daily_sales.index, daily_sales.values, linewidth=2)
        axes[0, 0].set_title('Ventes totales journalières', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Quantité vendue')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ventes moyennes par mois
        monthly_sales = self.df.groupby('month')['quantity'].mean()
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        axes[0, 1].bar(months, monthly_sales)
        axes[0, 1].set_title('Ventes moyennes par mois', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Quantité moyenne')
        
        # 3. Ventes par jour de semaine
        weekday_sales = self.df.groupby('day_of_week')['quantity'].mean()
        weekday_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        axes[1, 0].bar(weekday_names, weekday_sales)
        axes[1, 0].set_title('Ventes moyennes par jour de semaine', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Impact weekend vs semaine
        weekend_sales = self.df[self.df['is_weekend'] == 1]['quantity'].mean()
        weekday_sales = self.df[self.df['is_weekend'] == 0]['quantity'].mean()
        axes[1, 1].bar(['Semaine', 'Weekend'], [weekday_sales, weekend_sales])
        axes[1, 1].set_title('Ventes: Semaine vs Weekend', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Quantité moyenne')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sales_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_event_impact(self):
        """Impact des événements sur les ventes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Ventes avec/sans événements
        event_mask = self.df['has_event'] == 1
        event_sales = self.df[event_mask]['quantity'].mean()
        non_event_sales = self.df[~event_mask]['quantity'].mean()
        
        axes[0].bar(['Sans événement', 'Avec événement'], [non_event_sales, event_sales])
        axes[0].set_title('Impact des événements sur les ventes', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Quantité moyenne vendue')
        
        # Ajouter les valeurs au-dessus des barres
        for i, v in enumerate([non_event_sales, event_sales]):
            axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Types d'événements et leur impact
        if 'event_type' in self.df.columns:
            event_types = self.df['event_type'].unique()
            type_impact = {}
            for event_type in event_types:
                if event_type != 'none':
                    type_mask = self.df['event_type'] == event_type
                    type_impact[event_type] = self.df[type_mask]['quantity'].mean()
            
            if type_impact:
                types = list(type_impact.keys())
                impacts = list(type_impact.values())
                
                # Traduire les types d'événements
                type_translation = {
                    'university_event': 'Événement universitaire',
                    'market': 'Marché',
                    'match': 'Match',
                    'festival': 'Festival',
                    'exhibition': 'Exposition'
                }
                
                translated_types = [type_translation.get(t, t) for t in types]
                
                axes[1].bar(translated_types, impacts)
                axes[1].set_title('Impact par type d\'événement', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('Quantité moyenne vendue')
                axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'event_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weather_impact(self):
        """Impact de la météo sur les ventes."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Ventes vs température
        axes[0, 0].scatter(self.df['temperature'], self.df['quantity'], alpha=0.3, s=10)
        axes[0, 0].set_title('Ventes en fonction de la température', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Température (°C)')
        axes[0, 0].set_ylabel('Quantité vendue')
        
        # 2. Ventes par catégorie météo
        if 'weather_category' in self.df.columns:
            weather_sales = self.df.groupby('weather_category')['quantity'].mean()
            axes[0, 1].bar(weather_sales.index, weather_sales.values)
            axes[0, 1].set_title('Ventes par catégorie météo', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Quantité moyenne vendue')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Ventes jours de pluie vs pas de pluie
        if 'is_rainy_day' in self.df.columns:
            rainy_sales = self.df[self.df['is_rainy_day'] == 1]['quantity'].mean()
            dry_sales = self.df[self.df['is_rainy_day'] == 0]['quantity'].mean()
            
            axes[1, 0].bar(['Jour sec', 'Jour de pluie'], [dry_sales, rainy_sales])
            axes[1, 0].set_title('Impact de la pluie sur les ventes', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Quantité moyenne vendue')
        
        # 4. Corrélation météo-ventes
        weather_cols = ['temperature', 'precipitation']
        correlations = self.df[weather_cols + ['quantity']].corr()['quantity'][weather_cols]
        
        axes[1, 1].bar(correlations.index, correlations.values)
        axes[1, 1].set_title('Corrélation météo-ventes', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Coefficient de corrélation')
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weather_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_product_analysis(self):
        """Analyse des produits."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Top 10 produits les plus vendus
        if 'name' in self.df.columns:
            top_products = self.df.groupby('name')['quantity'].sum().sort_values(ascending=False).head(10)
            axes[0, 0].barh(top_products.index, top_products.values)
            axes[0, 0].set_title('Top 10 produits les plus vendus', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Quantité totale vendue')
        
        # 2. Ventes par catégorie
        if 'category' in self.df.columns:
            category_sales = self.df.groupby('category')['quantity'].sum().sort_values()
            axes[0, 1].barh(category_sales.index, category_sales.values)
            axes[0, 1].set_title('Ventes totales par catégorie', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Quantité totale vendue')
        
        # 3. Marges par catégorie
        if 'margin_percentage' in self.df.columns and 'category' in self.df.columns:
            category_margin = self.df.groupby('category')['margin_percentage'].mean().sort_values()
            axes[1, 0].barh(category_margin.index, category_margin.values)
            axes[1, 0].set_title('Marge moyenne par catégorie', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Marge moyenne (%)')
        
        # 4. Durée de vie des produits
        if 'shelf_life' in self.df.columns:
            shelf_life_dist = self.df['shelf_life'].value_counts().head(10)
            axes[1, 1].bar(shelf_life_dist.index.astype(str), shelf_life_dist.values)
            axes[1, 1].set_title('Distribution de la durée de vie', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Durée de vie (jours)')
            axes[1, 1].set_ylabel('Nombre de produits')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'product_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self):
        """Génère toutes les visualisations."""
        print("🎨 Génération des visualisations pour le rapport PFA...")
        
        self.plot_sales_trend()
        print("  ✅ Tendances des ventes générées")
        
        self.plot_event_impact()
        print("  ✅ Impact des événements généré")
        
        self.plot_weather_impact()
        print("  ✅ Impact météo généré")
        
        self.plot_product_analysis()
        print("  ✅ Analyse produits générée")
        
        print(f"\n✅ Toutes les visualisations ont été sauvegardées dans: {self.output_dir}/")

if __name__ == "__main__":
    visualizer = ReportVisualizer()
    visualizer.generate_all_plots()