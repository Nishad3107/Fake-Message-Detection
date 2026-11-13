import { ReactNode } from 'react'
import { Link, ShieldCheck } from 'lucide-react'

export type Route = 'home' | 'analyze' | 'dashboard' | 'methodology'

export function navigate(route: Route) {
  window.history.pushState({}, '', route === 'home' ? '/' : `/#${route}`)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

export function SiteShell({ route, onNavigate, children }: { route: Route; onNavigate: (route: Route) => void; children: ReactNode }) {
  return <div className="min-h-screen bg-[#f7f8fa] text-slate-950"><header className="sticky top-0 z-20 border-b border-slate-200/80 bg-[#f7f8fa]/90 backdrop-blur-xl"><div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8"><button onClick={() => onNavigate('home')} className="flex items-center gap-2.5" aria-label="Go to home"><span className="rounded-xl bg-indigo-600 p-2 text-white shadow-lg shadow-indigo-600/20"><ShieldCheck size={19}/></span><span className="text-sm font-bold tracking-tight">Signal<span className="text-indigo-600">Check</span></span></button><nav className="hidden items-center gap-7 text-sm font-medium text-slate-500 md:flex" aria-label="Main navigation">{([['home','Overview'],['analyze','Analyze'],['dashboard','Dashboard'],['methodology','Methodology']] as [Route,string][]).map(([key,label]) => <button key={key} onClick={() => onNavigate(key)} className={`transition hover:text-indigo-600 ${route === key ? 'text-indigo-600' : ''}`}>{label}</button>)}</nav><button onClick={() => onNavigate('analyze')} className="rounded-xl bg-slate-950 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-indigo-700">Analyze a post</button></div></header>{children}<footer className="border-t border-slate-200 bg-white"><div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-8 text-xs text-slate-400 sm:px-8 md:flex-row md:items-center md:justify-between"><p>SignalCheck · A probabilistic reading aid, not a truth engine.</p><div className="flex gap-5"><button onClick={() => onNavigate('methodology')} className="hover:text-indigo-600">Methodology</button><span>© 2026 SignalCheck</span></div></div></footer></div>
}
