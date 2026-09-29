import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { MarkdownContent } from '@/components/workspace/MarkdownContent';

const render = (content: string, extended = false) =>
  renderToStaticMarkup(createElement(MarkdownContent, { content, extended }));

describe('MarkdownContent extended mode', () => {
  it('leaves italics and TeX literal by default', () => {
    const html = render('A *query* costs $x^2$ and a * b.');
    expect(html).toContain('*query*');
    expect(html).toContain('$x^2$');
    expect(html).not.toContain('katex');
  });

  it('renders italics, inline and display TeX when extended', () => {
    const html = render('A *query* costs $x^2$, not a * b.\n\n$$\n\\sum_i x_i\n$$', true);
    expect(html).toContain('<em>query</em>');
    expect(html).toContain('a * b');
    // One inline span plus the span inside the display block.
    expect(html.match(/class="katex"/g)).toHaveLength(2);
    expect(html).toContain('katex-display');
  });

  it('keeps escaped dollars as plain text when extended', () => {
    const html = render('Costs \\$0.03 per 1,000 input tokens and \\$0.10 per 1,000 output.', true);
    expect(html).toContain('Costs $0.03 per 1,000 input tokens and $0.10 per 1,000 output.');
    expect(html).not.toContain('katex');
  });
});
