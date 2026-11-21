import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { clsx } from 'clsx';
import { fmtSignedPct } from '../../utils/format';

/**
 * change         : variation en %, ou null
 * higherIsBetter : colore la variation en vert si elle va dans le bon sens
 */
const KpiCard = ({
  title,
  value,
  change = null,
  changeLabel = 'vs J-7',
  higherIsBetter = true,
  hint,
  icon: Icon,
  iconColor = 'text-blue-600',
  bgColor = 'bg-blue-50',
  className = '',
}) => {
  const good = change != null && (higherIsBetter ? change >= 0 : change <= 0);

  return (
    <div className={clsx('bg-white rounded-xl shadow-sm border border-gray-200 p-6', className)}>
      <div className="flex items-center justify-between">
        <div className="min-w-0">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-2 tabular-nums">{value}</p>
          {change != null && (
            <div className={clsx('flex flex-wrap items-center mt-1 text-sm font-medium whitespace-nowrap', good ? 'text-green-600' : 'text-red-600')}>
              {change >= 0 ? <TrendingUp className="h-4 w-4 mr-1" /> : <TrendingDown className="h-4 w-4 mr-1" />}
              {fmtSignedPct(change)} <span className="ml-1 font-normal text-gray-500">{changeLabel}</span>
            </div>
          )}
          {hint && <p className="mt-1 text-xs text-gray-500">{hint}</p>}
        </div>
        {Icon && (
          <div className={clsx('p-3 rounded-full shrink-0', bgColor)}>
            <Icon className={clsx('h-6 w-6', iconColor)} />
          </div>
        )}
      </div>
    </div>
  );
};

export default KpiCard;
