import { NextResponse } from 'next/server';

import { voiceInterviewConfig } from '@/lib/server/voiceInterviewProvider';

export async function GET() {
  return NextResponse.json({ configured: voiceInterviewConfig() !== null });
}
