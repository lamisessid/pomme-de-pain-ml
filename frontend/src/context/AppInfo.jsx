import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { api } from '../api/client';

const AppInfoContext = createContext({ health: null, alertCount: null, refreshAlerts: () => {} });

/** État partagé par la mise en page : santé de l'API et nombre d'alertes actives. */
export function AppInfoProvider({ children }) {
  const [health, setHealth] = useState(null);
  const [alertCount, setAlertCount] = useState(null);

  const refreshAlerts = useCallback(() => {
    api.alertStats().then((s) => setAlertCount(s.active)).catch(() => setAlertCount(null));
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'down' }));
    refreshAlerts();
  }, [refreshAlerts]);

  return (
    <AppInfoContext.Provider value={{ health, alertCount, refreshAlerts }}>
      {children}
    </AppInfoContext.Provider>
  );
}

export const useAppInfo = () => useContext(AppInfoContext);
