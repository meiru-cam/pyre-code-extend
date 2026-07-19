import type { Problem, SourceRef } from '@/lib/types';


export function sourceHref(source: SourceRef): string {
  if (source.kind !== 'code' || !source.commit || !source.path) return source.url;
  const repository = source.url.replace(/\/$/, '');
  const line = source.line ? `#L${source.line}` : '';
  return `${repository}/blob/${source.commit}/${source.path}${line}`;
}


export function sourceLocator(source: SourceRef): string {
  if (source.kind === 'code') {
    return [source.path, source.symbol].filter(Boolean).join(' · ');
  }
  return [source.section, source.equation, source.figure, source.pages].filter(Boolean).join(' · ');
}


export function hasProblemInformation(problem: Problem): boolean {
  return !!(
    problem.modelConnections?.length
    || problem.proConAnalysis?.pros.length
    || problem.proConAnalysis?.cons.length
    || problem.sources?.length
  );
}
