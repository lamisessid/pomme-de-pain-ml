import React from 'react';
import Badge from './Badge';

const SEVERITY = {
  critical: { label: 'Critique', variant: 'error' },
  high: { label: 'Élevée', variant: 'warning' },
  medium: { label: 'Moyenne', variant: 'info' },
  low: { label: 'Faible', variant: 'default' },
};

const SeverityBadge = ({ severity }) => {
  const s = SEVERITY[severity] || SEVERITY.low;
  return <Badge variant={s.variant}>{s.label}</Badge>;
};

export default SeverityBadge;
