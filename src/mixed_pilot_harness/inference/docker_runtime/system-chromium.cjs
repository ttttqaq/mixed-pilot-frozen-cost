/* Use Alpine's installed Chromium rather than Puppeteer's glibc Chrome build.
 * Loaded through NODE_OPTIONS in inference and evaluation containers. Repository
 * files, hidden tests, and upstream parsers are not modified by this adapter.
 */
const fs = require('fs');
const Module = require('module');

const candidates = [process.env.SWE_NEO_CHROMIUM_PATH,
  '/usr/bin/chromium-browser', '/usr/bin/chromium'].filter(Boolean);
const executable = candidates.find((path) => fs.existsSync(path));

if (fs.existsSync('/etc/alpine-release') && executable) {
  const load = Module._load;
  const patched = new WeakSet();
  const adapt = (api) => {
    if (!api || !['object', 'function'].includes(typeof api) || patched.has(api)) return;
    if (typeof api.executablePath !== 'function') return;
    const launch = typeof api.launch === 'function' ? api.launch.bind(api) : null;
    Object.defineProperty(api, 'executablePath', {
      configurable: true, value: () => executable,
    });
    if (launch) Object.defineProperty(api, 'launch', {
      configurable: true,
      value: (options = {}) => launch(Object.assign({}, options, {executablePath: executable})),
    });
    patched.add(api);
    process.stderr.write('[swe-neo runtime] Puppeteer browser: ' + executable + '\n');
  };
  Module._load = function(request, parent, isMain) {
    const api = load.call(this, request, parent, isMain);
    if (request === 'puppeteer' || request === 'puppeteer-core') {
      adapt(api);
      if (api.default && api.default !== api) adapt(api.default);
    }
    return api;
  };
}
