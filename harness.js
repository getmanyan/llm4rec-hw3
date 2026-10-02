// Loads the app's own data.js and script.js into a Node VM context,
// so the analysis runs exactly the code that is deployed.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');

async function loadApp() {
    const ctx = {
        console, Math, Object, Array, Number, Float32Array, Float64Array, TextDecoder,
        setTimeout, performance,
        window: {},
        document: { getElementById: () => null },
        fetch: async (file) => {
            const buf = fs.readFileSync(path.join(ROOT, file));
            return {
                ok: true,
                text: async () => buf.toString('utf8'),
                arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
            };
        }
    };
    vm.createContext(ctx);
    // Top-level let/const are not exposed on the context, so append accessors
    const expose = `
        globalThis.__app = {
            get movies() { return movies; }, set movies(v) { movies = v; },
            get ratings() { return ratings; }, set ratings(v) { ratings = v; },
            get numUsers() { return numUsers; },
            get numMovies() { return numMovies; },
            get ratingMatrix() { return ratingMatrix; },
            get movieById() { return movieById; },
            get duplicateOf() { return duplicateOf; },
            buildRatingMatrix, cosineSimilarity, getUserBasedRecommendations,
            getItemBasedRecommendations, getSimilarUsers, predictRating, resetCaches,
            getItemSimilarities, getItemVectors, countRated, sparseCosineSimilarity, weightedCosine,
            set: (name, f) => { eval(name + ' = f'); },
            get: (name) => eval(name)
        };`;
    const code = fs.readFileSync(path.join(ROOT, 'data.js'), 'utf8') + '\n' +
                 fs.readFileSync(path.join(ROOT, 'script.js'), 'utf8') + '\n' + expose;
    vm.runInContext(code, ctx);
    await vm.runInContext('loadData()', ctx);
    return ctx.__app;
}

module.exports = { loadApp };
