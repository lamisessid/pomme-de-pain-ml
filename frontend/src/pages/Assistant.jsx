import React, { useEffect, useRef, useState } from 'react';
import { Send, Bot, User } from 'lucide-react';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import { Card } from '../components/ui/Card';
import { PageHeader } from '../components/ui/States';
import Loader from '../components/ui/Loader';

/** Rendu minimal du texte de l'assistant : sauts de ligne et **gras**. */
const RichText = ({ text }) => (
  <>
    {text.split('\n').map((line, i) => (
      <p key={i} className={line.trim() === '' ? 'h-2' : ''}>
        {line.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
          part.startsWith('**') ? <strong key={j}>{part.slice(2, -2)}</strong> : <React.Fragment key={j}>{part}</React.Fragment>,
        )}
      </p>
    ))}
  </>
);

const WELCOME = {
  role: 'bot',
  text: "Bonjour ! Je réponds à partir des données du point de vente et des prévisions du modèle : prévisions par produit, alertes, gaspillage, meilleures ventes, événements SQY et procédures.",
};

const Assistant = () => {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const suggestions = useApi(() => api.chatSuggestions(), []);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (text) => {
    const message = text.trim();
    if (!message || sending) return;
    setMessages((m) => [...m, { role: 'user', text: message }]);
    setInput('');
    setSending(true);
    try {
      const r = await api.chat(message);
      setMessages((m) => [...m, { role: 'bot', text: r.response }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'bot', text: `Erreur : ${e.message}`, error: true }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <PageHeader title="Assistant opérationnel" subtitle="Réponses construites à partir des données réelles et des prévisions (sans LLM externe)" />
      <Card padding={false} className="flex flex-col h-[calc(100vh-220px)] min-h-[480px]">
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}>
              {m.role === 'bot' && <div className="p-2 bg-indigo-50 rounded-full h-fit"><Bot className="h-4 w-4 text-indigo-600" /></div>}
              <div className={`max-w-[75%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                m.role === 'user' ? 'bg-primary-600 text-white' : m.error ? 'bg-red-50 text-red-800' : 'bg-gray-100 text-gray-900'}`}>
                <RichText text={m.text} />
              </div>
              {m.role === 'user' && <div className="p-2 bg-primary-50 rounded-full h-fit"><User className="h-4 w-4 text-primary-600" /></div>}
            </div>
          ))}
          {sending && <Loader size="small" />}
          <div ref={endRef} />
        </div>

        <div className="border-t border-gray-200 p-4">
          <div className="flex flex-wrap gap-2 mb-3">
            {suggestions.data?.map((s) => (
              <button key={s} onClick={() => send(s)} className="text-xs border border-gray-300 rounded-full px-3 py-1 hover:bg-gray-50">
                {s}
              </button>
            ))}
          </div>
          <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="flex gap-2">
            <input value={input} onChange={(e) => setInput(e.target.value)} maxLength={500}
                   placeholder="Ex. : prévision baguette, alertes, gaspillage…"
                   className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm" />
            <button type="submit" disabled={sending || !input.trim()} className="btn-primary inline-flex items-center disabled:opacity-50">
              <Send className="h-4 w-4 mr-1" /> Envoyer
            </button>
          </form>
        </div>
      </Card>
    </>
  );
};

export default Assistant;
