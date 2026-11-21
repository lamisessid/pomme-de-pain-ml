import { useCallback, useEffect, useState } from 'react';

/** Charge `fetcher()` au montage et à chaque changement de `deps`. */
export default function useApi(fetcher, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const load = useCallback(fetcher, deps);

  const reload = useCallback(() => {
    setState((s) => ({ ...s, loading: true, error: null }));
    load()
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => setState({ data: null, error, loading: false }));
  }, [load]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { ...state, reload };
}
