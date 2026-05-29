const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver.assetExts = [
  ...config.resolver.assetExts.filter(
    (ext) => !['glb', 'gltf', 'obj', 'mtl', 'hdr', 'ktx'].includes(ext),
  ),
  'glb',
  'gltf',
  'obj',
  'mtl',
  'hdr',
  'ktx',
];

// In-memory store for runtime-uploaded GLBs so ViroReact can fetch them via HTTP.
// (ViroReact's native GLTF loader fails on file:// URIs to the app sandbox storage,
//  but works with http:// URLs served by Metro over adb reverse.)
const dynamicMeshStore = new Map();

const previousEnhancer = config.server && config.server.enhanceMiddleware;

config.server = {
  ...config.server,
  enhanceMiddleware: (middleware, server) => {
    const wrapped = previousEnhancer ? previousEnhancer(middleware, server) : middleware;
    return (req, res, next) => {
      const url = req.url || '';
      // Upload: POST /dynamic-mesh/upload  body: { id, base64 }
      if (req.method === 'POST' && url === '/dynamic-mesh/upload') {
        let body = '';
        req.on('data', (chunk) => { body += chunk; });
        req.on('end', () => {
          try {
            const parsed = JSON.parse(body);
            const id = parsed && parsed.id;
            const base64 = parsed && parsed.base64;
            if (!id || !base64) {
              res.statusCode = 400;
              res.end('id and base64 required');
              return;
            }
            const buf = Buffer.from(base64, 'base64');
            dynamicMeshStore.set(id, buf);
            console.log(`[dynamic-mesh] stored ${id} (${buf.length} bytes)`);
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, id, size: buf.length }));
          } catch (e) {
            res.statusCode = 400;
            res.end('invalid body');
          }
        });
        return;
      }
      // Fetch: GET /dynamic-mesh/<id>.glb
      if (req.method === 'GET') {
        const m = url.match(/^\/dynamic-mesh\/([^/.?]+)\.glb/);
        if (m) {
          const id = m[1];
          const buf = dynamicMeshStore.get(id);
          if (!buf) {
            res.statusCode = 404;
            res.end('not found');
            return;
          }
          res.statusCode = 200;
          res.setHeader('Content-Type', 'model/gltf-binary');
          res.setHeader('Content-Length', String(buf.length));
          res.setHeader('Access-Control-Allow-Origin', '*');
          res.end(buf);
          return;
        }
      }
      return wrapped(req, res, next);
    };
  },
};

module.exports = config;
