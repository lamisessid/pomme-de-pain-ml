import React from 'react';
import { CalendarDays, TrendingUp } from 'lucide-react';
import { useAppInfo } from '../../context/AppInfo';
import { fmtDateLong } from '../../utils/format';

const Header = () => {
  const { health } = useAppInfo();
  const down = health?.status === 'down';

  return (
    <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-40">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="w-9 h-9 rounded-lg bg-pomme-red flex items-center justify-center" aria-hidden="true">
              <TrendingUp className="h-5 w-5 text-white" />
            </div>
            <div className="ml-3">
              <h1 className="text-xl font-bold text-gray-900">
                Pomme de Pain <span className="text-pomme-red">SQY</span>
              </h1>
              <p className="text-sm text-gray-500">Prévision de la demande & commandes</p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            {health?.reference_date && (
              <div className="hidden md:flex items-center text-gray-700 bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5"
                   title="Date « aujourd'hui » de la démo (configurable via REFERENCE_DATE)">
                <CalendarDays className="h-4 w-4 mr-2 text-gray-500" />
                {fmtDateLong(health.reference_date)}
              </div>
            )}
            <div className="flex items-center">
              <span className={`w-2 h-2 rounded-full mr-2 ${down ? 'bg-red-500' : health ? 'bg-green-500' : 'bg-gray-300'}`} />
              <span className="text-gray-600">{down ? 'API hors ligne' : health ? 'API connectée' : 'Connexion…'}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
