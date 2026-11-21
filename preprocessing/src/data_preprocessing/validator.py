"""
Module pour valider la qualité des données.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import logging

from .utils import setup_logger


class DataValidator:
    """Valide la qualité des données à différentes étapes."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.validation_results = {}
    
    def validate_dataset(self, data: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Valide un dataset complet.
        
        Vérifications:
        1. Structure de base
        2. Valeurs manquantes
        3. Valeurs aberrantes
        4. Cohérence temporelle
        5. Distributions
        """
        self.logger.info(f"🔍 Validation du dataset: {dataset_name}")
        
        results = {
            'dataset_name': dataset_name,
            'basic_checks': {},
            'missing_values': {},
            'outliers': {},
            'temporal_checks': {},
            'distribution_checks': {},
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        if data is None or data.empty:
            results['issues'].append("Dataset vide")
            results['passed'] = False
            return results
        
        # 1. Vérifications basiques
        results['basic_checks'] = self._basic_checks(data)
        
        # 2. Valeurs manquantes
        results['missing_values'] = self._missing_values_check(data)
        
        # 3. Valeurs aberrantes
        results['outliers'] = self._outlier_check(data)
        
        # 4. Vérifications temporelles
        if 'date' in data.columns:
            results['temporal_checks'] = self._temporal_checks(data)
        
        # 5. Distributions
        results['distribution_checks'] = self._distribution_checks(data)
        
        # Résumé
        issues_count = len(results['issues'])
        warnings_count = len(results['warnings'])
        
        if issues_count > 0:
            results['passed'] = False
            self.logger.warning(f"❌ {issues_count} problèmes dans {dataset_name}")
        else:
            self.logger.info(f"✅ {dataset_name} validé avec succès")
        
        if warnings_count > 0:
            self.logger.info(f"⚠️  {warnings_count} avertissements dans {dataset_name}")
        
        self.validation_results[dataset_name] = results
        return results
    
    def _basic_checks(self, data: pd.DataFrame) -> Dict:
        """Vérifications basiques de structure."""
        checks = {
            'shape': data.shape,
            'columns_count': len(data.columns),
            'rows_count': len(data),
            'duplicates': data.duplicated().sum(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Vérifier les types de données
        dtypes_summary = {}
        for dtype, count in data.dtypes.value_counts().items():
            dtypes_summary[str(dtype)] = int(count)
        checks['dtypes_summary'] = dtypes_summary
        
        return checks
    
    def _missing_values_check(self, data: pd.DataFrame) -> Dict:
        """Vérifie les valeurs manquantes."""
        missing_stats = data.isnull().sum()
        total_missing = missing_stats.sum()
        total_cells = data.size
        missing_percentage = total_missing / total_cells * 100 if total_cells > 0 else 0
        
        result = {
            'total_missing': int(total_missing),
            'missing_percentage': float(missing_percentage),
            'columns_with_missing': {},
            'critical_columns': {}
        }
        
        # Colonnes avec plus de 5% de valeurs manquantes
        for col in data.columns:
            missing_count = missing_stats[col]
            missing_pct = missing_count / len(data) * 100 if len(data) > 0 else 0
            
            if missing_count > 0:
                result['columns_with_missing'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_pct)
                }
            
            if missing_pct > 5:
                result['critical_columns'][col] = float(missing_pct)
        
        return result
    
    def _outlier_check(self, data: pd.DataFrame) -> Dict:
        """Détecte les valeurs aberrantes."""
        result = {'columns_checked': [], 'outliers_found': {}}
        
        # Vérifier les colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'date':
                continue
            
            # Méthode IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Éviter division par zéro
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                
                if outliers > 0:
                    result['outliers_found'][col] = {
                        'count': int(outliers),
                        'percentage': float(outliers / len(data) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'min': float(data[col].min()),
                        'max': float(data[col].max())
                    }
        
        return result
    
    def _temporal_checks(self, data: pd.DataFrame) -> Dict:
        """Vérifie la cohérence temporelle."""
        result = {
            'date_range': {},
            'gaps': {},
            'duplicate_dates': 0,
            'date_format_issues': 0
        }
        
        # Vérifier le format de date
        try:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            invalid_dates = data['date'].isnull().sum()
            result['date_format_issues'] = int(invalid_dates)
        except:
            result['date_format_issues'] = len(data)
        
        # Plage de dates
        if data['date'].notna().any():
            result['date_range']['min'] = str(data['date'].min())
            result['date_range']['max'] = str(data['date'].max())
            result['date_range']['days'] = (data['date'].max() - data['date'].min()).days
            
            # Vérifier les trous temporels
            data_sorted = data.sort_values('date')
            date_diff = data_sorted['date'].diff()
            gaps = date_diff[date_diff > pd.Timedelta(days=1)]
            
            if not gaps.empty:
                result['gaps']['count'] = len(gaps)
                result['gaps']['max_gap_days'] = gaps.max().days
                result['gaps']['avg_gap_days'] = gaps.mean().days
            
            # Dates en double
            duplicate_dates = data['date'].duplicated().sum()
            result['duplicate_dates'] = int(duplicate_dates)
        
        return result
    
    def _distribution_checks(self, data: pd.DataFrame) -> Dict:
        """Vérifie les distributions des variables."""
        result = {}
        
        # Pour les colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].notna().any():
                stats = data[col].describe()
                result[col] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    '25%': float(stats['25%']),
                    '50%': float(stats['50%']),
                    '75%': float(stats['75%']),
                    'max': float(stats['max']),
                    'skew': float(data[col].skew()) if len(data[col]) > 1 else 0,
                    'kurtosis': float(data[col].kurtosis()) if len(data[col]) > 1 else 0
                }
        
        # Pour les colonnes catégorielles
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col != 'date':
                value_counts = data[col].value_counts()
                result[col] = {
                    'unique_values': int(value_counts.nunique()),
                    'top_value': value_counts.index[0] if not value_counts.empty else None,
                    'top_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    'top_percentage': float(value_counts.iloc[0] / len(data) * 100) if not value_counts.empty else 0
                }
        
        return result
    
    def get_validation_report(self, dataset_name: str) -> str:
        """Génère un rapport de validation."""
        if dataset_name not in self.validation_results:
            return f"Aucune validation pour {dataset_name}"
        
        results = self.validation_results[dataset_name]
        report = []
        
        report.append(f"=== RAPPORT DE VALIDATION: {dataset_name} ===")
        report.append(f"Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        report.append("")
        
        # Checks basiques
        report.append("=== CHECKS BASIQUES ===")
        basic = results['basic_checks']
        report.append(f"Shape: {basic['shape']}")
        report.append(f"Colonnes: {basic['columns_count']}")
        report.append(f"Lignes: {basic['rows_count']}")
        report.append(f"Doublons: {basic['duplicates']}")
        report.append(f"Usage mémoire: {basic['memory_usage_mb']:.2f} MB")
        report.append("")
        
        # Valeurs manquantes
        missing = results['missing_values']
        report.append("=== VALEURS MANQUANTES ===")
        report.append(f"Total manquantes: {missing['total_missing']}")
        report.append(f"Pourcentage: {missing['missing_percentage']:.2f}%")
        
        if missing['critical_columns']:
            report.append("Colonnes critiques (>5% manquants):")
            for col, pct in missing['critical_columns'].items():
                report.append(f"  - {col}: {pct:.1f}%")
        report.append("")
        
        # Valeurs aberrantes
        outliers = results['outliers']
        if outliers['outliers_found']:
            report.append("=== VALEURS ABERRANTES ===")
            for col, info in outliers['outliers_found'].items():
                report.append(f"{col}: {info['count']} outliers ({info['percentage']:.1f}%)")
        report.append("")
        
        # Checks temporels
        if results['temporal_checks']:
            temporal = results['temporal_checks']
            report.append("=== CHECKS TEMPORELS ===")
            if 'date_range' in temporal:
                report.append(f"Période: {temporal['date_range'].get('min', 'N/A')} to {temporal['date_range'].get('max', 'N/A')}")
                report.append(f"Jours: {temporal['date_range'].get('days', 0)}")
            
            if temporal.get('duplicate_dates', 0) > 0:
                report.append(f"Dates dupliquées: {temporal['duplicate_dates']}")
            
            if temporal.get('gaps', {}):
                report.append(f"Trous temporels: {temporal['gaps'].get('count', 0)}")
        report.append("")
        
        # Issues et warnings
        if results['issues']:
            report.append("=== PROBLÈMES ===")
            for issue in results['issues']:
                report.append(f"❌ {issue}")
        
        if results['warnings']:
            report.append("=== AVERTISSEMENTS ===")
            for warning in results['warnings']:
                report.append(f"⚠️  {warning}")
        
        return "\n".join(report)