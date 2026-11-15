import { motion } from 'framer-motion'
import { ArrowRight, BarChart3, BrainCircuit, Database, ShieldAlert, Sparkles } from 'lucide-react'
import { Route } from '../components/SiteShell'

const features = [
  ['Real PHEME dataset', 'Labeled rumor-veracity messages from nine events.', Database],
  ['Event-level evaluation', 'Related events and threads stay separated.', BarChart3],
  ['Explainable NLP', 'TF-IDF and Naive Bayes associations remain inspectable.', BrainCircuit],
  ['Confidence warnings', 'Every result makes uncertainty visible.', ShieldAlert],
] as const

export default function LandingPage({ onNavigate }: { onNavigate: (route: Route) => void }) {
  return (
    <main>
      <section className="hero-grid relative overflow-hidden">
        <div className="orb orb-one" />
        <div className="orb orb-two" />
        <div className="mx-auto grid max-w-7xl gap-14 px-5 pb-24 pt-20 sm:px-8 lg:grid-cols-[1.08fr_.92fr] lg:items-center lg:pb-32 lg:pt-28">
          <div>
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-6 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white/75 px-3 py-1.5 text-xs font-semibold uppercase tracking-[.16em] text-indigo-700"><Sparkles size={14} /> Academic NLP project</motion.div>
            <motion.h1 initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .08 }} className="max-w-3xl text-5xl font-semibold leading-[.98] tracking-[-.065em] sm:text-7xl">Understand the signal.<br /><span className="text-indigo-600">Question the claim.</span></motion.h1>
            <p className="mt-7 max-w-xl text-lg leading-8 text-slate-500">A transparent NLP system that analyzes language patterns in social-media rumor messages using the real PHEME dataset and an explainable classical model.</p>
            <div className="mt-9 flex flex-wrap gap-3"><button onClick={() => onNavigate('analyze')} className="group flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-3.5 text-sm font-semibold text-white shadow-xl shadow-indigo-600/20 transition hover:-translate-y-0.5 hover:bg-indigo-700">Analyze a Message <ArrowRight size={17} /></button><button onClick={() => onNavigate('methodology')} className="rounded-xl border border-slate-200 bg-white/70 px-5 py-3.5 text-sm font-semibold text-slate-700 transition hover:border-indigo-200 hover:text-indigo-700">Explore the Methodology</button></div>
            <p className="mt-6 text-xs leading-5 text-slate-400">This system produces probabilistic NLP predictions. It does not verify factual truth.</p>
          </div>
          <motion.div initial={{ opacity: 0, scale: .96 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: .15 }} className="glass-card rounded-[2rem] p-5 shadow-2xl shadow-indigo-900/10"><div className="flex items-center justify-between border-b border-slate-100 pb-4"><div><p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Transparent pipeline</p><p className="mt-1 text-sm font-semibold">From message to model signal</p></div><span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-600">Research</span></div><div className="mt-6 space-y-3">{['Input message', 'Normalize text + TF-IDF', 'Compare three PHEME labels'].map((step, index) => <div key={step} className="flex items-center gap-3 rounded-2xl border border-slate-100 bg-white p-3"><span className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-xs font-bold text-indigo-600">0{index + 1}</span><span className="text-sm font-medium text-slate-700">{step}</span><span className="ml-auto text-xs text-slate-400">{index === 2 ? 'with limits' : 'stateless'}</span></div>)}</div><div className="mt-6 rounded-2xl bg-slate-950 p-4 text-sm leading-6 text-slate-300">The model can identify patterns associated with a label. It cannot establish whether a claim is true.</div></motion.div>
        </div>
      </section>
      <section className="mx-auto max-w-7xl px-5 py-20 sm:px-8"><div className="mb-10 max-w-2xl"><p className="eyebrow">Designed for scrutiny</p><h2 className="mt-3 text-3xl font-semibold tracking-tight">A research workflow with the caveats in view.</h2></div><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{features.map(([title, text, Icon], index) => <motion.article key={title} initial={{ opacity: 0, y: 15 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * .06 }} className="rounded-2xl border border-slate-200 bg-white p-6"><div className="mb-6 flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600"><Icon size={20} /></div><h3 className="font-semibold">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{text}</p></motion.article>)}</div></section>
      <section className="border-y border-slate-200 bg-white"><div className="mx-auto grid max-w-7xl gap-8 px-5 py-16 sm:px-8 md:grid-cols-[.7fr_1.3fr] md:items-center"><div><p className="eyebrow">Three steps</p><h2 className="mt-3 text-3xl font-semibold tracking-tight">A useful pause before sharing.</h2></div><div className="grid gap-4 md:grid-cols-3">{[['01', 'Input', 'Paste a message you want to examine.'], ['02', 'Analyze', 'Preprocessing and the frozen model compare learned patterns.'], ['03', 'Interpret', 'Read the label, probabilities, and limitations together.']].map(([number, title, text]) => <div key={number} className="rounded-2xl bg-slate-950 p-5 text-white"><span className="text-xs font-bold text-indigo-300">{number}</span><h3 className="mt-8 font-semibold">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-400">{text}</p></div>)}</div></div></section>
      <section className="mx-auto max-w-7xl px-5 py-20 sm:px-8"><div className="rounded-3xl border border-amber-200 bg-amber-50 p-7 sm:p-10"><div className="flex gap-4"><ShieldAlert className="mt-1 shrink-0 text-amber-700" /><div><p className="eyebrow text-amber-700">Ethical use</p><h2 className="mt-3 text-2xl font-semibold text-slate-900">A model signal is not a verdict.</h2><p className="mt-4 max-w-3xl text-sm leading-7 text-slate-600">Use this project to study NLP behavior and prompt source-checking. Do not use it to censor people, label individuals, or make high-stakes decisions. Dataset bias, event shift, and false positives are expected.</p><p className="mt-5 font-semibold text-slate-900">This system produces probabilistic NLP predictions. It does not verify factual truth.</p></div></div></div></section>
    </main>
  )
}
