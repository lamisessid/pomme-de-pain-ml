import React, { useState } from 'react';
import {
  LayoutDashboard,
  BarChart3,
  AlertTriangle,
  MessageSquare,
  TrendingUp,
  PieChart,
  Coffee,
  Info,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useAppInfo } from '../../context/AppInfo';
import { fmtDate } from '../../utils/format';

const navigation = [
  { name: 'Tableau de bord', href: '/', icon: LayoutDashboard, color: 'text-blue-600' },
  { name: 'Prévisions', href: '/forecasts', icon: TrendingUp, color: 'text-purple-600' },
  { name: 'Analyse des pertes', href: '/losses', icon: PieChart, color: 'text-amber-600' },
  { name: 'Siège vs IA', href: '/comparison', icon: BarChart3, color: 'text-green-600' },
  { name: 'Alertes', href: '/alerts', icon: AlertTriangle, color: 'text-red-600', badge: true },
  { name: 'Produits', href: '/products', icon: Coffee, color: 'text-pomme-brown' },
  { name: 'Assistant', href: '/assistant', icon: MessageSquare, color: 'text-indigo-600' },
  { name: 'Modèle & méthode', href: '/about', icon: Info, color: 'text-gray-600' },
];

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const { health, alertCount } = useAppInfo();

  return (
    <aside className={`${collapsed ? 'w-20' : 'w-64'} shrink-0 bg-white border-r border-gray-200 transition-all duration-300 h-[calc(100vh-64px)] sticky top-16`}>
      <div className="flex flex-col h-full">
        <div className="p-4 border-b border-gray-200 flex justify-end">
          <button onClick={() => setCollapsed(!collapsed)} className="p-1 hover:bg-gray-100 rounded-lg"
                  aria-label={collapsed ? 'Déplier le menu' : 'Replier le menu'}>
            {collapsed ? <ChevronRight className="h-5 w-5 text-gray-500" /> : <ChevronLeft className="h-5 w-5 text-gray-500" />}
          </button>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => (
            <NavLink
              key={item.href}
              to={item.href}
              end={item.href === '/'}
              title={collapsed ? item.name : undefined}
              className={({ isActive }) => `
                flex items-center ${collapsed ? 'justify-center' : 'justify-start'}
                px-3 py-2.5 rounded-lg transition-colors
                ${isActive ? 'bg-primary-50 text-primary-700 border-l-4 border-primary-500' : 'text-gray-700 hover:bg-gray-50'}
              `}
            >
              <item.icon className={`h-5 w-5 ${item.color} ${!collapsed ? 'mr-3' : ''}`} />
              {!collapsed && (
                <>
                  <span className="font-medium">{item.name}</span>
                  {item.badge && alertCount > 0 && (
                    <span className="ml-auto bg-red-100 text-red-800 text-xs font-medium px-2 py-0.5 rounded-full">
                      {alertCount}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {!collapsed && health?.model_loaded !== undefined && (
          <div className="p-4 border-t border-gray-200">
            <div className="bg-primary-50 rounded-lg p-3 text-xs text-primary-700">
              <div className="flex items-center font-medium text-sm">
                <span className={`w-2 h-2 rounded-full mr-2 ${health.model_loaded ? 'bg-green-500' : 'bg-red-500'}`} />
                {health.model_loaded ? 'Modèle LSTM chargé' : 'Modèle non entraîné'}
              </div>
              {health.data_range && (
                <p className="mt-1">
                  Données : {fmtDate(health.data_range.start, { month: 'short', year: 'numeric' })} →{' '}
                  {fmtDate(health.data_range.end, { month: 'short', year: 'numeric' })}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
