import { ExternalLink } from 'lucide-react';

import { hasProblemInformation, sourceHref, sourceLocator } from '@/lib/problemInformation';
import type { Problem } from '@/lib/types';


interface ProblemInformationProps {
  problem: Problem;
}


export function ProblemInformation({ problem }: ProblemInformationProps) {
  const hasConnections = !!problem.modelConnections?.length;
  const hasTradeoffs = !!(
    problem.proConAnalysis?.pros.length || problem.proConAnalysis?.cons.length
  );
  const hasSources = !!problem.sources?.length;

  if (!hasProblemInformation(problem)) return null;

  return (
    <div className="space-y-4">
      {hasConnections && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-text">Model connections</h2>
          <ul className="space-y-1.5 pl-5 text-sm leading-relaxed text-text-2">
            {problem.modelConnections!.map((connection) => (
              <li key={connection} className="list-disc">{connection}</li>
            ))}
          </ul>
        </section>
      )}

      {hasTradeoffs && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-text">Pros and cons</h2>
          <div className="grid grid-cols-2 gap-3 max-[720px]:grid-cols-1">
            {([
              ['Pros', problem.proConAnalysis?.pros ?? []],
              ['Cons', problem.proConAnalysis?.cons ?? []],
            ] as const).map(([label, items]) => (
              <div
                key={label}
                className="rounded-lg p-3"
                style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
              >
                <h3 className="mono mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.1em] text-text-3">
                  {label}
                </h3>
                <ul className="space-y-1 pl-4 text-xs leading-relaxed text-text-2">
                  {items.map((item) => <li key={item} className="list-disc">{item}</li>)}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      {hasSources && (
        <details
          className="rounded-lg p-3.5 text-sm"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          <summary className="cursor-pointer font-medium text-text">
            Optional implementation references
          </summary>
          <p className="mt-2 text-xs leading-relaxed text-text-3">
            These precise source locations are additional context; reading them is not required before attempting the exercise.
          </p>
          <ul className="mt-3 space-y-3">
            {problem.sources!.map((source, index) => (
              <li key={`${source.url}-${index}`} className="text-xs leading-relaxed text-text-2">
                <a
                  href={sourceHref(source)}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 font-medium text-accent hover:underline"
                >
                  {source.kind === 'code' ? 'Code' : 'Paper'} · {sourceLocator(source)}
                  <ExternalLink className="h-3 w-3" />
                </a>
                {source.kind === 'code' && source.commit && (
                  <div className="mono mt-1 text-[10.5px] text-text-3">
                    commit {source.commit.slice(0, 12)}{source.license ? ` · ${source.license}` : ''}
                  </div>
                )}
                {source.adapted && <div className="mt-1"><span className="font-medium">Adapted:</span> {source.adapted}</div>}
                {source.simplifications && <div><span className="font-medium">Simplified:</span> {source.simplifications}</div>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
