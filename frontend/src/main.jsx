import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import PianoPage from './piano/PianoPage.jsx'

function parseHashRoute(hash) {
  const [path, query = ''] = hash.split('?');
  const params = new URLSearchParams(query);

  return {
    path,
    midiUrl: params.get('midi'),
  };
}

function Root() {
  const [route, setRoute] = useState(window.location.hash);

  useEffect(() => {
    const onHash = () => setRoute(window.location.hash);
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const { path, midiUrl } = parseHashRoute(route);

  if (path === '#/piano') return <PianoPage midiUrl={midiUrl} />;
  return <App />;
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
