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

module.exports = config;
