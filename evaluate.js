// ---------------------------------------------------------------------------
// HW3 offline evaluation. Run from the repo root:  node analysis/evaluate.js
// Uses the app's own data.js + script.js (loaded by harness.js).
// Every number in the report comes from this script. Seeded, so re-runs match.
// ---------------------------------------------------------------------------
const { loadApp } = require('./harness');

// Seeded random generator (mulberry32)
function rng(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
function shuffle(arr, rand) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}
const pct = x => (100 * x).toFixed(1) + '%';
const f2 = x => x.toFixed(2);
const f3 = x => x.toFixed(3);
const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

const LONG_TAIL = 100;   // a movie with fewer than 100 ratings (in the full data) is "long tail"
const LIKED = 4;         // a held-out rating of 4 or 5 counts as a liked movie
const EVAL_USERS = 300;  // users sampled for the strategy comparison

(async () => {
    const app = await loadApp();
    const allRatings = app.ratings.map(r => ({ ...r }));
    const popularity = {};
    for (const m of app.movies) popularity[m.id] = m.numRatings;

    // Original functions, so variants can be switched on and off
    const orig = {
        weightedCosine: app.get('weightedCosine'),
        cosineSimilarity: app.get('cosineSimilarity'),
        sparseCosineSimilarity: app.get('sparseCosineSimilarity')
    };
    function restore() {
        for (const [k, v] of Object.entries(orig)) app.set(k, v);
        app.set('SHRINKAGE', 50); app.set('MIN_SUPPORT', 3);
        app.set('NUM_NEIGHBORS', 20); app.set('ITEM_NEIGHBORS', 20);
        app.resetCaches();
    }
    function useTrain(train) {
        app.ratings = train;
        app.buildRatingMatrix();
        app.resetCaches();
    }

    console.log('=== Data ===');
    console.log(`users ${app.numUsers}, movie ids ${app.numMovies}, movies after merging duplicates ${app.movies.length}, ratings ${allRatings.length}`);
    console.log(`density ${pct(allRatings.length / (app.numUsers * app.movies.length))}`);
    const ratingsPerUser = {};
    for (const r of allRatings) ratingsPerUser[r.userId] = (ratingsPerUser[r.userId] || 0) + 1;
    console.log(`ratings per user: min ${Math.min(...Object.values(ratingsPerUser))}, median ${median(Object.values(ratingsPerUser))}, max ${Math.max(...Object.values(ratingsPerUser))}`);
    const pops = Object.values(popularity);
    console.log(`ratings per movie: median ${median(pops)}, movies with 1 rating ${pops.filter(x => x === 1).length}, long tail (<${LONG_TAIL}) ${pct(pops.filter(x => x < LONG_TAIL).length / pops.length)}`);

    // -----------------------------------------------------------------------
    // 0. Check: sparse and dense cosine give the same item similarities
    // -----------------------------------------------------------------------
    {
        const vecs = app.getItemVectors();
        const dense = id => { const v = new Float32Array(app.numUsers + 1); vecs[id].idx.forEach((u, k) => v[u] = vecs[id].val[k]); return v; };
        const rand = rng(7); let maxDiff = 0;
        for (let t = 0; t < 20000; t++) {
            const a = app.movies[Math.floor(rand() * app.movies.length)].id;
            const b = app.movies[Math.floor(rand() * app.movies.length)].id;
            maxDiff = Math.max(maxDiff, Math.abs(app.cosineSimilarity(dense(a), dense(b)) - app.sparseCosineSimilarity(vecs[a], vecs[b])));
        }
        console.log(`\n[check] dense vs sparse cosine, 20000 random movie pairs: max difference ${maxDiff.toExponential(1)}`);
    }

    // -----------------------------------------------------------------------
    // 1. Why few co-rated items are unreliable (full data)
    // -----------------------------------------------------------------------
    console.log('\n=== 1. Unreliable similarity from few co-rated items (all user pairs) ===');
    {
        const M = app.ratingMatrix; const buckets = {};
        const order = [[1, 2], [3, 5], [6, 10], [11, 25], [26, 50], [51, 1e9]];
        const label = ([a, b]) => b > 1e8 ? `${a}+` : `${a}-${b}`;
        for (const o of order) buckets[label(o)] = { n: 0, sum: 0, high: 0 };
        let pairs = 0, noOverlap = 0;
        for (let u = 1; u <= app.numUsers; u++) for (let v = u + 1; v <= app.numUsers; v++) {
            let dot = 0, na = 0, nb = 0, c = 0; const a = M[u], b = M[v];
            for (let i = 1; i < a.length; i++) { const x = a[i], y = b[i]; if (x && y) { dot += x * y; na += x * x; nb += y * y; c++; } }
            pairs++;
            if (!c) { noOverlap++; continue; }
            const cos = dot / Math.sqrt(na * nb);
            const o = order.find(([lo, hi]) => c >= lo && c <= hi); const bk = buckets[label(o)];
            bk.n++; bk.sum += cos; if (cos >= 0.99) bk.high++;
        }
        console.log(`user pairs ${pairs}, with no co-rated movie ${pct(noOverlap / pairs)}`);
        console.log('co-rated | pairs | mean plain cosine | share with cosine >= 0.99');
        for (const [k, b] of Object.entries(buckets)) console.log(`${k.padEnd(8)} | ${String(b.n).padStart(6)} | ${f3(b.sum / b.n)} | ${pct(b.high / b.n)}`);

        // Nearest neighbours of a few users: plain cosine vs weighted
        const coRated = (a, b) => { let c = 0; for (let i = 1; i < a.length; i++) if (a[i] && b[i]) c++; return c; };
        let plainN = [], weightN = [];
        for (let u = 1; u <= app.numUsers; u++) {
            app.set('weightedCosine', (d, x, y, c) => { const den = Math.sqrt(x) * Math.sqrt(y); return den ? d / den : 0; });
            const p = app.getSimilarUsers(u, 20);
            app.set('weightedCosine', orig.weightedCosine);
            const w = app.getSimilarUsers(u, 20);
            plainN.push(median(p.map(x => coRated(M[u], M[x.userId]))));
            weightN.push(median(w.map(x => coRated(M[u], M[x.userId]))));
            if (u === 1) {
                console.log(`example user 1 (${app.countRated(M[1])} ratings): top neighbour, plain cosine -> user ${p[0].userId}, sim ${f3(p[0].sim)}, co-rated ${coRated(M[1], M[p[0].userId])}; weighted -> user ${w[0].userId}, sim ${f3(w[0].sim)}, co-rated ${coRated(M[1], M[w[0].userId])}`);
            }
        }
        console.log(`median co-rated movies with the 20 nearest neighbours (median over users): plain cosine ${median(plainN)}, weighted ${median(weightN)}`);
    }

    // -----------------------------------------------------------------------
    // Hold-out split: 20% of each user's ratings are hidden
    // -----------------------------------------------------------------------
    const rand = rng(42);
    const byUser = {};
    for (const r of allRatings) (byUser[r.userId] = byUser[r.userId] || []).push(r);
    const train = [], test = {};
    for (const [u, rs] of Object.entries(byUser)) {
        shuffle(rs, rand);
        const k = Math.max(1, Math.round(rs.length * 0.2));
        test[u] = rs.slice(0, k);
        train.push(...rs.slice(k));
    }
    const evalUsers = shuffle(Object.keys(test).map(Number), rng(1)).slice(0, EVAL_USERS);
    const testPairs = [];
    for (const u of evalUsers) for (const r of test[u]) testPairs.push(r);

    // Top-5 metrics for a recommender function
    function topNMetrics(recommend, users = evalUsers) {
        let hits = 0, precision = 0, empty = 0, lt = 0, popSum = 0, n = 0, time = 0;
        const seen = new Set(), lists = {};
        for (const u of users) {
            const t0 = performance.now(); const recs = recommend(u); time += performance.now() - t0;
            lists[u] = recs.map(r => r.id);
            if (recs.length === 0) { empty++; continue; }
            const liked = new Set(test[u].filter(r => r.rating >= LIKED).map(r => r.itemId));
            const h = recs.filter(r => liked.has(r.id)).length;
            if (h > 0) hits++;
            precision += h / 5;
            for (const r of recs) { seen.add(r.id); n++; popSum += popularity[r.id]; if (popularity[r.id] < LONG_TAIL) lt++; }
        }
        return {
            hit: hits / users.length, prec: precision / users.length, empty: empty / users.length,
            coverage: seen.size / app.movies.length, longTail: n ? lt / n : 0, pop: n ? popSum / n : 0,
            ms: time / users.length, lists
        };
    }

    // Rating-prediction error on held-out ratings (both approaches)
    function predictionError() {
        let ub = [0, 0, 0], ib = [0, 0, 0]; // abs error sum, squared error sum, count
        let missUb = 0, missIb = 0;
        const M = app.ratingMatrix;
        for (const u of evalUsers) {
            const sims = [];
            for (let v = 1; v <= app.numUsers; v++) if (v !== u) { const s = app.cosineSimilarity(M[u], M[v]); if (s > 0) sims.push({ v, s }); }
            sims.sort((a, b) => b.s - a.s);
            for (const r of test[u]) {
                const raters = sims.filter(x => M[x.v][r.itemId] !== 0).slice(0, app.get('NUM_NEIGHBORS'));
                let w = 0, ws = 0; for (const x of raters) { w += x.s * M[x.v][r.itemId]; ws += x.s; }
                if (ws > 0) { const e = w / ws - r.rating; ub[0] += Math.abs(e); ub[1] += e * e; ub[2]++; } else missUb++;
                const row = app.getItemSimilarities(r.itemId);
                const nb = [];
                for (let j = 1; j <= app.numMovies; j++) if (M[u][j] !== 0 && row[j] > 0) nb.push({ j, s: row[j] });
                nb.sort((a, b) => b.s - a.s);
                let w2 = 0, ws2 = 0; for (const x of nb.slice(0, app.get('ITEM_NEIGHBORS'))) { w2 += x.s * M[u][x.j]; ws2 += x.s; }
                if (ws2 > 0) { const e = w2 / ws2 - r.rating; ib[0] += Math.abs(e); ib[1] += e * e; ib[2]++; } else missIb++;
            }
        }
        return {
            ubMAE: ub[0] / ub[2], ubRMSE: Math.sqrt(ub[1] / ub[2]), ubMiss: missUb / testPairs.length,
            ibMAE: ib[0] / ib[2], ibRMSE: Math.sqrt(ib[1] / ib[2]), ibMiss: missIb / testPairs.length
        };
    }

    // Baselines
    function mostPopular(u) {
        const M = app.ratingMatrix;
        return [...app.movies].filter(m => M[u][m.id] === 0)
            .sort((a, b) => popularity[b.id] - popularity[a.id]).slice(0, 5).map(m => ({ id: m.id }));
    }
    // Global mean / item mean predictions as an MAE baseline
    function baselineError() {
        const sum = {}, cnt = {}; let g = 0;
        for (const r of train) { sum[r.itemId] = (sum[r.itemId] || 0) + r.rating; cnt[r.itemId] = (cnt[r.itemId] || 0) + 1; g += r.rating; }
        g /= train.length; let eg = 0, ei = 0;
        for (const r of testPairs) { eg += Math.abs(g - r.rating); ei += Math.abs((cnt[r.itemId] ? sum[r.itemId] / cnt[r.itemId] : g) - r.rating); }
        return { global: eg / testPairs.length, item: ei / testPairs.length };
    }

    // Item-Based variant: weighted AVERAGE of the user's ratings over the 20 most similar rated movies
    function itemBasedAverage(u) {
        const M = app.ratingMatrix; const active = M[u];
        const rated = app.movies.filter(m => active[m.id] !== 0).map(m => m.id);
        const rows = rated.map(id => app.getItemSimilarities(id));
        const out = [];
        for (const m of app.movies) {
            if (active[m.id] !== 0) continue;
            const nb = [];
            for (let k = 0; k < rated.length; k++) if (rows[k][m.id] > 0) nb.push({ s: rows[k][m.id], r: active[rated[k]] });
            if (nb.length < 3) continue;
            nb.sort((a, b) => b.s - a.s);
            let w = 0, ws = 0; for (const x of nb.slice(0, 20)) { w += x.s * x.r; ws += x.s; }
            out.push({ id: m.id, score: w / ws, n: nb.length });
        }
        return out.sort((a, b) => b.score - a.score || popularity[b.id] - popularity[a.id]).slice(0, 5);
    }
    // Same as the app's Item-Based score (kept to double-check the app)
    function itemBasedSum(u) {
        const M = app.ratingMatrix; const active = M[u];
        const rated = app.movies.filter(m => active[m.id] !== 0).map(m => m.id);
        const rows = rated.map(id => app.getItemSimilarities(id));
        const out = [];
        for (const m of app.movies) {
            if (active[m.id] !== 0) continue;
            let s = 0; for (let k = 0; k < rated.length; k++) s += rows[k][m.id] * active[rated[k]];
            if (s > 0) out.push({ id: m.id, score: s });
        }
        return out.sort((a, b) => b.score - a.score).slice(0, 5);
    }

    // Mean imputation: missing entries take the vector's own mean (row mean for users,
    // column mean for movies), then a plain cosine over ALL entries.
    function meanImputedDense(a, b) {
        let sa = 0, ca = 0, sb = 0, cb = 0;
        for (let i = 1; i < a.length; i++) { if (a[i]) { sa += a[i]; ca++; } if (b[i]) { sb += b[i]; cb++; } }
        if (!ca || !cb) return 0;
        const ma = sa / ca, mb = sb / cb; let dot = 0, na = 0, nb = 0;
        for (let i = 1; i < a.length; i++) { const x = a[i] || ma, y = b[i] || mb; dot += x * y; na += x * x; nb += y * y; }
        return dot / Math.sqrt(na * nb);
    }
    function meanImputedSparse(a, b) {
        const L = app.numUsers; // length of an item column (user ids 1..numUsers)
        if (!a.idx.length || !b.idx.length) return 0;
        const ma = a.val.reduce((s, x) => s + x, 0) / a.idx.length, mb = b.val.reduce((s, x) => s + x, 0) / b.idx.length;
        let dot = 0, i = 0, j = 0, both = 0, aOnly = 0, bOnly = 0;
        let sumA2 = 0, sumB2 = 0;
        for (const x of a.val) sumA2 += x * x;
        for (const y of b.val) sumB2 += y * y;
        while (i < a.idx.length || j < b.idx.length) {
            if (j >= b.idx.length || (i < a.idx.length && a.idx[i] < b.idx[j])) { dot += a.val[i] * mb; aOnly++; i++; }
            else if (i >= a.idx.length || a.idx[i] > b.idx[j]) { dot += ma * b.val[j]; bOnly++; j++; }
            else { dot += a.val[i] * b.val[j]; both++; i++; j++; }
        }
        const neither = L - both - aOnly - bOnly;
        dot += neither * ma * mb;
        const na = sumA2 + (L - a.idx.length) * ma * ma, nb = sumB2 + (L - b.idx.length) * mb * mb;
        return dot / Math.sqrt(na * nb);
    }

    useTrain(train);
    const base = baselineError();
    console.log(`\n=== Hold-out: 20% of each user's ratings hidden; ${EVAL_USERS} sampled users, ${testPairs.length} hidden ratings ===`);
    console.log(`baseline MAE: global mean ${f3(base.global)}, movie mean ${f3(base.item)}`);

    const strategies = {
        'co-rated only': () => { app.set('weightedCosine', (d, x, y) => { const den = Math.sqrt(x) * Math.sqrt(y); return den ? d / den : 0; }); },
        'mean imputation': () => { app.set('cosineSimilarity', meanImputedDense); app.set('sparseCosineSimilarity', meanImputedSparse); },
        'weighted (gamma=50, app)': () => {},
        'weighted (gamma=10)': () => { app.set('SHRINKAGE', 10); },
        'weighted (gamma=100)': () => { app.set('SHRINKAGE', 100); }
    };

    console.log('\n=== 2. Missing-value strategy (MIN_SUPPORT=3, N=20) ===');
    console.log('strategy | approach | HitRate@5 | Precision@5 | coverage | long-tail | avg #ratings | MAE | RMSE | ms/request');
    const results = {};
    for (const [name, setup] of Object.entries(strategies)) {
        restore(); useTrain(train); setup();
        const t0 = performance.now(); app.getItemSimilarities(app.movies[0].id); // warm one row
        const ub = topNMetrics(u => app.getUserBasedRecommendations(u));
        const tIb0 = performance.now();
        // full item-item table build time (first request cost)
        for (const m of app.movies) app.getItemSimilarities(m.id);
        const buildMs = performance.now() - tIb0;
        const ib = topNMetrics(u => app.getItemBasedRecommendations(u));
        const err = predictionError();
        results[name] = { ub, ib, err };
        console.log(`${name} | user-based | ${pct(ub.hit)} | ${pct(ub.prec)} | ${pct(ub.coverage)} | ${pct(ub.longTail)} | ${Math.round(ub.pop)} | ${f3(err.ubMAE)} | ${f3(err.ubRMSE)} | ${f2(ub.ms)}`);
        console.log(`${name} | item-based | ${pct(ib.hit)} | ${pct(ib.prec)} | ${pct(ib.coverage)} | ${pct(ib.longTail)} | ${Math.round(ib.pop)} | ${f3(err.ibMAE)} | ${f3(err.ibRMSE)} | ${f2(ib.ms)} (+${Math.round(buildMs)} ms one-off item table)`);
    }
    restore(); useTrain(train);
    const pop = topNMetrics(mostPopular);
    console.log(`most popular | - | ${pct(pop.hit)} | ${pct(pop.prec)} | ${pct(pop.coverage)} | ${pct(pop.longTail)} | ${Math.round(pop.pop)} | - | - | -`);

    // -----------------------------------------------------------------------
    // 3. Ranking details: minimum support and the Item-Based score
    // -----------------------------------------------------------------------
    console.log('\n=== 3. Ranking choices (weighted, gamma=50) ===');
    console.log('variant | HitRate@5 | coverage | long-tail | avg #ratings');
    for (const s of [1, 3, 5]) {
        restore(); useTrain(train); app.set('MIN_SUPPORT', s);
        const ub = topNMetrics(u => app.getUserBasedRecommendations(u));
        const ib = topNMetrics(u => app.getItemBasedRecommendations(u));
        console.log(`MIN_SUPPORT=${s} user-based | ${pct(ub.hit)} | ${pct(ub.coverage)} | ${pct(ub.longTail)} | ${Math.round(ub.pop)}`);
        console.log(`MIN_SUPPORT=${s} item-based | ${pct(ib.hit)} | ${pct(ib.coverage)} | ${pct(ib.longTail)} | ${Math.round(ib.pop)}`);
    }
    restore(); useTrain(train);
    const ibSum = topNMetrics(itemBasedSum);
    console.log(`item-based, score = sum(sim x rating) [app] | ${pct(ibSum.hit)} | ${pct(ibSum.coverage)} | ${pct(ibSum.longTail)} | ${Math.round(ibSum.pop)}`);

    restore(); useTrain(train);
    const ibAvg = topNMetrics(itemBasedAverage);
    console.log(`item-based, score = weighted average of the user's ratings (top-20 similar, >=3) | ${pct(ibAvg.hit)} | ${pct(ibAvg.coverage)} | ${pct(ibAvg.longTail)} | ${Math.round(ibAvg.pop)}`);

    // Overlap between the two app lists
    {
        const a = results['weighted (gamma=50, app)'];
        let same = 0, users = 0;
        for (const u of evalUsers) { const x = new Set(a.ub.lists[u]); const y = a.ib.lists[u]; if (!y) continue; same += y.filter(id => x.has(id)).length; users++; }
        console.log(`\nUser-Based vs Item-Based Top-5 (app settings): avg ${f2(same / users)} common movies of 5`);
    }

    // -----------------------------------------------------------------------
    // 4. Efficiency: which side is larger decides the cost
    // -----------------------------------------------------------------------
    console.log('\n=== 4. Efficiency ===');
    {
        restore(); useTrain(allRatings);
        const heavy = Object.entries(ratingsPerUser).sort((a, b) => b[1] - a[1])[0];
        let t = performance.now(); app.getUserBasedRecommendations(Number(heavy[0])); const ubHeavy = performance.now() - t;
        t = performance.now(); app.getItemBasedRecommendations(Number(heavy[0])); const ibHeavyCold = performance.now() - t;
        t = performance.now(); app.getItemBasedRecommendations(Number(heavy[0])); const ibHeavyWarm = performance.now() - t;
        restore(); useTrain(allRatings);
        t = performance.now(); for (const m of app.movies) app.getItemSimilarities(m.id); const full = performance.now() - t;
        const userPairs = app.numUsers * (app.numUsers - 1) / 2, itemPairs = app.movies.length * (app.movies.length - 1) / 2;
        console.log(`users ${app.numUsers} vs movies ${app.movies.length}`);
        console.log(`all similarity pairs: user-user ${userPairs}, item-item ${itemPairs} (${f2(itemPairs / userPairs)}x)`);
        console.log(`per request, User-Based compares the active user with ${app.numUsers - 1} users (each vector has ${app.numMovies} slots)`);
        console.log(`heaviest user ${heavy[0]} (${heavy[1]} ratings): User-Based ${Math.round(ubHeavy)} ms; Item-Based ${Math.round(ibHeavyCold)} ms cold, ${Math.round(ibHeavyWarm)} ms with cached rows`);
        console.log(`full item-item table: ${Math.round(full)} ms, ${(4 * (app.numMovies + 1) ** 2 / 1e6).toFixed(1)} MB as Float32`);
    }

    // -----------------------------------------------------------------------
    // 5. Cold start
    // -----------------------------------------------------------------------
    console.log('\n=== 5. Cold start ===');
    {
        // New users: keep only k ratings of a user, hide the rest
        const heavyUsers = shuffle(Object.keys(byUser).map(Number).filter(u => byUser[u].length >= 60), rng(5)).slice(0, 100);
        console.log(`${heavyUsers.length} users with >= 60 ratings; keep k random ratings, the rest is hidden`);
        console.log('k | user-based: empty list / HitRate@5 / movies shared with most-popular Top-5 | item-based: same | most popular HitRate@5');
        for (const k of [0, 1, 3, 5, 10, 20, 40]) {
            restore();
            const keep = new Set(); const tr = [], te = {};
            for (const r of allRatings) {
                if (heavyUsers.includes(r.userId)) continue; else tr.push(r);
            }
            for (const u of heavyUsers) {
                const rs = shuffle([...byUser[u]], rng(u * 31 + k));
                tr.push(...rs.slice(0, k)); te[u] = rs.slice(k);
            }
            const saved = { ...test }; Object.assign(test, te);
            useTrain(tr);
            const ub = topNMetrics(u => app.getUserBasedRecommendations(u), heavyUsers);
            const ib = topNMetrics(u => app.getItemBasedRecommendations(u), heavyUsers);
            const mp = topNMetrics(mostPopular, heavyUsers);
            Object.assign(test, saved);
            const overlap = (lists) => { let s = 0; for (const u of heavyUsers) { const p = new Set(mp.lists[u]); s += (lists[u] || []).filter(id => p.has(id)).length; } return s / heavyUsers.length; };
            console.log(`${k} | ${pct(ub.empty)} / ${pct(ub.hit)} / ${f2(overlap(ub.lists))} | ${pct(ib.empty)} / ${pct(ib.hit)} / ${f2(overlap(ib.lists))} | ${pct(mp.hit)}`);
        }
        // New items: a movie with no ratings has an empty column
        restore(); useTrain(train);
        const ratedInTrain = new Set(train.map(r => r.itemId));
        const unseen = app.movies.filter(m => !ratedInTrain.has(m.id));
        const vecs = app.getItemVectors();
        const m0 = unseen[0];
        const maxSim = m0 ? Math.max(...app.getItemSimilarities(m0.id)) : NaN;
        let recommended = 0;
        for (const u of evalUsers) {
            const ids = new Set([...app.getUserBasedRecommendations(u), ...app.getItemBasedRecommendations(u)].map(r => r.id));
            for (const m of unseen) if (ids.has(m.id)) recommended++;
        }
        const hiddenOnUnseen = Object.values(test).flat().filter(r => !ratedInTrain.has(r.itemId)).length;
        console.log(`new items: ${unseen.length} movies have no rating in the training part (e.g. "${m0 && m0.title}", empty column of ${vecs[m0.id].idx.length} ratings, max similarity to any movie ${maxSim});`);
        console.log(`  times such a movie appeared in ${EVAL_USERS} x 2 Top-5 lists: ${recommended}; hidden ratings that land on such movies: ${hiddenOnUnseen}`);
        const few = app.movies.filter(m => vecs[m.id].idx.length > 0 && vecs[m.id].idx.length <= 3).length;
        console.log(`  movies with only 1-3 training ratings: ${few} (${pct(few / app.movies.length)} of the catalog)`);
    }

    // -----------------------------------------------------------------------
    // 6. Seeded defect in the starter data.js: genre list (18 names vs 19 flags)
    // -----------------------------------------------------------------------
    console.log('\n=== 6. Starter genre parser vs fixed parser ===');
    const fsx = require('fs'), pathx = require('path');
    const starterNames = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"];
    const fixedNames = ['unknown', ...starterNames];
    const itemLines = fsx.readFileSync(pathx.join(__dirname, '..', 'u.item'), 'latin1').split('\n').filter(l => l.trim());
    const genreVec = {};
    {
        let westernStarter = 0, westernFixed = 0, changed = 0;
        for (const l of itemLines) {
            const f = l.split('|'); const v = f.slice(5, 24).map(Number);
            const before = starterNames.filter((_, i) => v[i] === 1);
            const after = fixedNames.filter((_, i) => v[i] === 1).filter(g => g !== 'unknown');
            if (before.includes('Western')) westernStarter++;
            if (after.includes('Western')) westernFixed++;
            if (before.join() !== after.join()) changed++;
            genreVec[+f[0]] = v;
            if (+f[0] === 1) console.log(`Toy Story (1995): starter -> ${before.join(', ')}; fixed -> ${after.join(', ')}`);
        }
        console.log(`movies whose genre list changes after the fix: ${changed} of ${itemLines.length}`);
        console.log(`movies labelled Western: starter ${westernStarter} (these are War flags), fixed ${westernFixed}`);
        const appToy = app.movieById[1].genres.join(', ');
        console.log(`app's data.js gives Toy Story: ${appToy}`);
    }

    // -----------------------------------------------------------------------
    // 7. Co-rated counts between users + hand-checked similarity
    // -----------------------------------------------------------------------
    console.log('\n=== 7. Co-rated movies between two users, and a hand check ===');
    restore(); useTrain(allRatings);
    {
        const M = app.ratingMatrix; const counts = [];
        for (let u = 1; u <= app.numUsers; u++) for (let v = u + 1; v <= app.numUsers; v++) {
            let k = 0; const a = M[u], b = M[v];
            for (let i = 1; i < a.length; i++) if (a[i] && b[i]) k++;
            counts.push(k);
        }
        console.log(`user pairs ${counts.length}: median co-rated ${median(counts)}, share with <= 2 co-rated ${pct(counts.filter(x => x <= 2).length / counts.length)} (with 0: ${pct(counts.filter(x => x === 0).length / counts.length)})`);
        const common = [];
        for (let i = 1; i < M[1].length; i++) if (M[1][i] && M[88][i]) common.push(`${app.movieById[i].title}: user 1 = ${M[1][i]}, user 88 = ${M[88][i]}`);
        console.log(`user 1 vs user 88 co-rated: ${common.join('; ')}`);
        const plain = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) if (a[i] && b[i]) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.sqrt(x * y); };
        console.log(`code: plain cosine ${plain(M[1], M[88]).toFixed(3)}, app weighted cosine ${app.cosineSimilarity(M[1], M[88]).toFixed(3)} (hand: 1*5/(1*5) = 1.000; x 1/50 = 0.020)`);
        // toy example for the three strategies (4 movies)
        const A = new Float32Array([5, 1, 0, 0]), B = new Float32Array([5, 0, 1, 0]);
        app.set('weightedCosine', (d, x, y) => { const den = Math.sqrt(x) * Math.sqrt(y); return den ? d / den : 0; });
        const co = app.cosineSimilarity(A, B);
        app.set('weightedCosine', orig.weightedCosine);
        const w = app.cosineSimilarity(A, B);
        const mean = meanImputedDense(new Float32Array([0, ...A]), new Float32Array([0, ...B]));
        console.log(`toy A=[5,1,-,-], B=[5,-,1,-]: co-rated only ${co.toFixed(3)} (hand 25/25 = 1), mean imputation ${mean.toFixed(3)} (hand 40/44 = 0.909), weighted ${w.toFixed(3)} (hand 1 x 1/50 = 0.020)`);
    }

    // -----------------------------------------------------------------------
    // 8. Week 2 content-based Top-5 vs Week 3 CF Top-5
    // -----------------------------------------------------------------------
    console.log('\n=== 8. Week 2 (genres) vs Week 3 (ratings) ===');
    {
        const M = app.ratingMatrix;
        const gcos = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return x && y ? d / Math.sqrt(x * y) : 0; };
        const contentTop = (q, exclude) => app.movies.filter(m => !exclude.has(m.id))
            .map(m => ({ id: m.id, s: gcos(q, genreVec[m.id]) }))
            .sort((a, b) => b.s - a.s || popularity[b.id] - popularity[a.id]).slice(0, 5);
        const show = l => l.map(x => `${app.movieById[x.id].title} [${app.movieById[x.id].genres.join('/')}] ${(x.s !== undefined ? x.s : x.score).toFixed(3)}, ${popularity[x.id]} ratings`).join('\n    ');
        console.log('Same movie, Toy Story (1995):');
        console.log('  Week 2 content-based (genre cosine, ties by popularity):\n    ' + show(contentTop(genreVec[1], new Set([1]))));
        const row = app.getItemSimilarities(1);
        console.log('  Week 3 item-item CF (rating cosine, weighted):\n    ' + show(app.movies.filter(m => m.id !== 1).map(m => ({ id: m.id, s: row[m.id] })).sort((a, b) => b.s - a.s).slice(0, 5)));

        let ovU = 0, ovI = 0, pC = 0, pU = 0, pI = 0, gC = 0, gU = 0, gI = 0, n = 0;
        for (let u = 1; u <= app.numUsers; u++) {
            const liked = app.movies.filter(m => M[u][m.id] >= LIKED).map(m => m.id);
            if (!liked.length) continue;
            const prof = new Array(19).fill(0);
            for (const id of liked) genreVec[id].forEach((x, i) => prof[i] += x / liked.length);
            const rated = new Set(app.movies.filter(m => M[u][m.id] !== 0).map(m => m.id));
            const c = contentTop(prof, rated).map(x => x.id);
            const a = app.getUserBasedRecommendations(u).map(x => x.id);
            const b = app.getItemBasedRecommendations(u).map(x => x.id);
            const cs = new Set(c);
            ovU += a.filter(x => cs.has(x)).length; ovI += b.filter(x => cs.has(x)).length;
            const avg = (l, f) => l.length ? l.reduce((s, x) => s + f(x), 0) / l.length : 0;
            pC += avg(c, x => popularity[x]); pU += avg(a, x => popularity[x]); pI += avg(b, x => popularity[x]);
            gC += avg(c, x => gcos(prof, genreVec[x])); gU += avg(a, x => gcos(prof, genreVec[x])); gI += avg(b, x => gcos(prof, genreVec[x]));
            n++;
        }
        console.log(`All ${n} users, Week 2 profile = average genre vector of movies rated >= ${LIKED}:`);
        console.log(`  movies shared with the content Top-5: User-Based ${f2(ovU / n)}, Item-Based ${f2(ovI / n)} of 5`);
        console.log(`  genre cosine to the user's profile: content ${f2(gC / n)}, User-Based ${f2(gU / n)}, Item-Based ${f2(gI / n)}`);
        console.log(`  avg #ratings of recommended movies: content ${Math.round(pC / n)}, User-Based ${Math.round(pU / n)}, Item-Based ${Math.round(pI / n)}`);
    }
})();
