import type { editor } from 'monaco-editor';

// Token colors are the same values as the --code-* variables in globals.css, so the
// editor and the description/test-case code blocks read as one language, not two.
// Monaco wants them without the leading '#'.
export const appleLight: editor.IStandaloneThemeData = {
  base: 'vs',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '5f6e80', fontStyle: 'italic' },
    { token: 'keyword', foreground: '9333ea' },
    { token: 'string', foreground: '047857' },
    { token: 'number', foreground: 'b45309' },
    { token: 'type', foreground: '1d4ed8' },
    { token: 'function', foreground: '0e7490' },
    { token: 'variable', foreground: '0f172a' },
    { token: 'operator', foreground: '475569' },
    { token: 'delimiter', foreground: '475569' },
  ],
  colors: {
    'editor.background': '#ffffff',
    'editor.foreground': '#0f172a',
    'editor.lineHighlightBackground': '#f1f5f9',
    'editor.selectionBackground': '#3b82f629',
    'editorLineNumber.foreground': '#94a3b8',
    'editorLineNumber.activeForeground': '#475569',
    'editor.inactiveSelectionBackground': '#3b82f614',
    'editorIndentGuide.background': '#e2e8f0',
    'editorCursor.foreground': '#3b82f6',
  },
};

export const appleDark: editor.IStandaloneThemeData = {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '93a3b8', fontStyle: 'italic' },
    { token: 'keyword', foreground: 'c084fc' },
    { token: 'string', foreground: '6ee7b7' },
    { token: 'number', foreground: 'fbbf24' },
    { token: 'type', foreground: '7dd3fc' },
    { token: 'function', foreground: '67e8f9' },
    { token: 'variable', foreground: 'e2e8f0' },
    { token: 'operator', foreground: '94a3b8' },
    { token: 'delimiter', foreground: '94a3b8' },
  ],
  colors: {
    'editor.background': '#0f172a',
    'editor.foreground': '#e2e8f0',
    'editor.lineHighlightBackground': '#1e293b',
    'editor.selectionBackground': '#60a5fa33',
    'editorLineNumber.foreground': '#475569',
    'editorLineNumber.activeForeground': '#94a3b8',
    'editor.inactiveSelectionBackground': '#60a5fa18',
    'editorIndentGuide.background': '#1e293b',
    'editorCursor.foreground': '#60a5fa',
  },
};
