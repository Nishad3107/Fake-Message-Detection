import { useEffect, useState } from 'react'
import { Route, SiteShell } from './components/SiteShell'
import AboutPage from './pages/AboutPage'
import AnalyzerPage from './pages/AnalyzerPage'
import DashboardPage from './pages/DashboardPage'
import LandingPage from './pages/LandingPage'
import ResultsPage from './pages/ResultsPage'

function currentRoute(): Route {
  const value = window.location.hash.replace('#', '')
  return ['analyze', 'dashboard', 'methodology', 'results'].includes(value) ? value as Route : 'home'
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute)
  useEffect(() => { const handle = () => setRoute(currentRoute()); window.addEventListener('popstate', handle); window.addEventListener('hashchange', handle); return () => { window.removeEventListener('popstate', handle); window.removeEventListener('hashchange', handle) } }, [])
  function onNavigate(next: Route) { window.history.pushState({}, '', next === 'home' ? window.location.pathname : `#${next}`); setRoute(next); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  const page = route === 'home' ? <LandingPage onNavigate={onNavigate}/> : route === 'analyze' ? <AnalyzerPage onNavigate={onNavigate}/> : route === 'dashboard' ? <DashboardPage onNavigate={onNavigate}/> : route === 'methodology' ? <AboutPage/> : <ResultsPage/>
  return <SiteShell route={route} onNavigate={onNavigate}>{page}</SiteShell>
}
