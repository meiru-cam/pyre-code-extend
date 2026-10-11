#!/usr/bin/env bash
# Rebuild the exported JSON and the production web build, then restart the local pm2 apps.
# The frontend is stopped during the build because it serves the same web/.next directory.
set -euo pipefail

cd "$(dirname "$0")/.."

npm run -s build:problems
npm run -s build:solutions
npm run -s build:prep

pm2 stop pyre-frontend
# Restart even when the build fails, so a broken build does not leave the site down.
status=0
(cd web && npm run build) || status=$?
pm2 restart pyre-frontend pyre-backend
exit "$status"
