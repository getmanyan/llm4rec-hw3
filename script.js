// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// Missing-value strategy (see week3/readme.md section 6):
//
//   [x] weight similarity by the number of co-rated items
//
// cosineSimilarity looks only at entries rated in BOTH vectors (0 = not rated),
// and then multiplies the cosine by min(n, SHRINKAGE) / SHRINKAGE, where n is the
// number of co-rated entries. Why: on positive 1-5 ratings, a cosine computed
// from 1-2 shared films is close to 1.0 almost by default, so two strangers who
// both gave "Star Wars" a 5 would look like perfect twins. The weight keeps such
// pairs at the bottom until there is enough shared evidence (lecture: "weight by
// common ratings" -> data reliability).
// ---------------------------------------------------------------------------

// Tunable constants (checked with analysis/evaluate.js, see the report)
let SHRINKAGE = 50;            // co-rated items needed for a full-weight similarity
let NUM_NEIGHBORS = 20;        // N most similar users in User-Based CF
let ITEM_NEIGHBORS = 20;       // most similar rated movies used for a predicted rating (Item-Based)
let MIN_SUPPORT = 3;           // User-Based: at least this many of the N neighbours must have rated a movie

// Cache for Item-Based CF: item vectors (matrix columns) and item-item similarities
let itemVectors = null;        // itemVectors[movieId] = sparse { idx: userIds, val: ratings }
let itemSimCache = {};         // movieId -> Float32Array of similarities to every movie

// Initialize the application when the window loads
window.onload = async function() {
    const userBased = document.getElementById('user-based-result');
    const itemBased = document.getElementById('item-based-result');

    try {
        userBased.innerHTML = '<p>Loading movie data...</p>';
        itemBased.innerHTML = '<p>Loading movie data...</p>';

        await loadData();

        populateUserDropdown();
        populateMovieDropdown();

        userBased.innerHTML = '<p>Data loaded. Select a user.</p>';
        itemBased.innerHTML = '<p>Data loaded. Select a user.</p>';
    } catch (error) {
        console.error('Initialization error:', error);
        // The error message is already shown by data.js
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId} (${countRated(ratingMatrix[userId])} ratings)`;
        selectElement.appendChild(option);
    }
}

// Extra (slide version of the task): movie dropdown for "Predict rating"
function populateMovieDropdown() {
    const selectElement = document.getElementById('movie-select');
    if (!selectElement) return;

    const sorted = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    for (const movie of sorted) {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    }
}

// Number of rated (non-zero) entries in a vector
function countRated(vector) {
    let n = 0;
    for (let i = 0; i < vector.length; i++) if (vector[i] !== 0) n++;
    return n;
}

// ---------------------------------------------------------------------------
// Cosine similarity between two rating vectors (weighted approach, see top).
// Only co-rated entries (non-zero in both) are used.
// Returns 0 when the vectors share no rated entries (zero denominator).
// Output: a number in [0, 1] (ratings are positive, so the cosine is never negative).
// Used directly for user-user similarity (dense rows of the rating matrix).
// ---------------------------------------------------------------------------
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0, coRated = 0;

    for (let i = 0; i < a.length; i++) {
        const x = a[i], y = b[i];
        if (x === 0 || y === 0) continue;   // skip anything not rated by both
        dot += x * y;
        normA += x * x;
        normB += y * y;
        coRated++;
    }

    return weightedCosine(dot, normA, normB, coRated);
}

// The same rule for sparse vectors { idx: sorted ids, val: ratings }.
// Item-Based CF compares ~1.4 million movie pairs; a movie column has 944 slots
// but only ~60 ratings on average, so walking only the rated entries is
// several times faster (about 4x for user 1 in my test). The result is identical to cosineSimilarity on the dense columns
// (checked in analysis/evaluate.js).
function sparseCosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0, coRated = 0;
    let i = 0, j = 0;

    while (i < a.idx.length && j < b.idx.length) {
        if (a.idx[i] < b.idx[j]) i++;
        else if (a.idx[i] > b.idx[j]) j++;
        else {                               // co-rated entry
            const x = a.val[i], y = b.val[j];
            dot += x * y;
            normA += x * x;
            normB += y * y;
            coRated++;
            i++; j++;
        }
    }

    return weightedCosine(dot, normA, normB, coRated);
}

// Cosine from co-rated sums, weighted by the amount of shared evidence
function weightedCosine(dot, normA, normB, coRated) {
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    const reliability = Math.min(coRated, SHRINKAGE) / SHRINKAGE;
    return (dot / denominator) * reliability;
}

// ---------------------------------------------------------------------------
// User-Based CF: "users similar to you liked these movies".
// Returns [{ id, title, score, support }] sorted by predicted rating (1-5).
// ---------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = 5) {
    const neighbors = getSimilarUsers(activeUserId, NUM_NEIGHBORS);
    if (neighbors.length === 0) return [];

    const active = ratingMatrix[activeUserId];
    const candidates = [];

    for (const movie of movies) {
        if (active[movie.id] !== 0) continue;              // only movies the user has NOT rated
        const prediction = predictFromNeighbors(neighbors, movie.id);
        if (prediction && prediction.support >= MIN_SUPPORT) {
            candidates.push({ id: movie.id, title: movie.title, score: prediction.score, support: prediction.support });
        }
    }

    return sortCandidates(candidates).slice(0, topK);
}

// Steps 1-2: compare the active user with every other user, keep the N most similar
function getSimilarUsers(activeUserId, n = NUM_NEIGHBORS) {
    const active = ratingMatrix[activeUserId];
    const sims = [];

    for (let u = 1; u <= numUsers; u++) {
        if (u === activeUserId) continue;
        const sim = cosineSimilarity(active, ratingMatrix[u]);
        if (sim > 0) sims.push({ userId: u, sim });
    }

    sims.sort((a, b) => b.sim - a.sim);
    return sims.slice(0, n);
}

// Step 3: similarity-weighted average of the neighbours' ratings for one movie
function predictFromNeighbors(neighbors, movieId) {
    let weighted = 0, weights = 0, support = 0;
    for (const { userId, sim } of neighbors) {
        const r = ratingMatrix[userId][movieId];
        if (r === 0) continue;
        weighted += sim * r;
        weights += sim;
        support++;
    }
    if (weights === 0) return null;
    return { score: weighted / weights, support };
}

// ---------------------------------------------------------------------------
// Item-Based CF: "because you liked X, we recommend Y".
// Score of a candidate = sum over the user's rated movies of sim(rated, candidate) x rating,
// divided by the sum of the user's ratings. The division is the same for every
// candidate of this user, so it does not change the order; it only turns a sum like
// "809.7" into a readable 0-1 "rating-weighted similarity".
// (A weighted average over the 20 most similar rated movies was tested too: it ranks
// obscure movies first; see the report.)
// Returns [{ id, title, score, because }] sorted by score.
// ---------------------------------------------------------------------------
function getItemBasedRecommendations(activeUserId, topK = 5) {
    const active = ratingMatrix[activeUserId];

    // Step 1: item-to-item similarities for every movie the user has rated
    const ratedIds = [];
    for (const movie of movies) if (active[movie.id] !== 0) ratedIds.push(movie.id);
    if (ratedIds.length === 0) return [];
    const simRows = ratedIds.map(id => getItemSimilarities(id));
    let ratingSum = 0;
    for (const id of ratedIds) ratingSum += active[id];

    // Step 2: for each unrated candidate, aggregate similarities weighted by the user's ratings
    const candidates = [];
    for (const movie of movies) {
        const j = movie.id;
        if (active[j] !== 0) continue;

        let score = 0, best = -1, bestContribution = 0;
        for (let k = 0; k < ratedIds.length; k++) {
            const contribution = simRows[k][j] * active[ratedIds[k]];
            score += contribution;
            if (contribution > bestContribution) { bestContribution = contribution; best = k; }
        }
        if (score > 0) {
            candidates.push({ id: j, title: movie.title, score: score / ratingSum, support: 0, because: movieById[ratedIds[best]].title });
        }
    }

    // Step 3: sort and take the top K
    return sortCandidates(candidates).slice(0, topK);
}

// Rating column of every movie as a sparse vector, built once from the rating matrix
function getItemVectors() {
    if (itemVectors) return itemVectors;
    itemVectors = [];
    for (let j = 0; j <= numMovies; j++) itemVectors.push({ idx: [], val: [] });
    for (let u = 1; u <= numUsers; u++) {          // users in increasing order -> idx is sorted
        const row = ratingMatrix[u];
        for (let j = 1; j <= numMovies; j++) {
            if (row[j] !== 0) {
                itemVectors[j].idx.push(u);
                itemVectors[j].val.push(row[j]);
            }
        }
    }
    return itemVectors;
}

// Similarities of one movie to every other movie (cached: each row is computed once)
function getItemSimilarities(movieId) {
    if (itemSimCache[movieId]) return itemSimCache[movieId];

    const vectors = getItemVectors();
    const row = new Float32Array(numMovies + 1);
    for (const other of movies) {
        if (other.id === movieId) continue;
        row[other.id] = sparseCosineSimilarity(vectors[movieId], vectors[other.id]);
    }
    itemSimCache[movieId] = row;
    return row;
}

// Clear caches after the rating matrix changes (used by analysis/evaluate.js)
function resetCaches() {
    itemVectors = null;
    itemSimCache = {};
}

// Ties: higher score, then more supporting neighbours, then more ratings overall
function sortCandidates(candidates) {
    return candidates.sort((a, b) =>
        b.score - a.score ||
        b.support - a.support ||
        movieById[b.id].numRatings - movieById[a.id].numRatings
    );
}

// ---------------------------------------------------------------------------
// Extra (slide version of the task): predict one user's rating for one movie
// with both approaches.
// ---------------------------------------------------------------------------
function predictRating(userId, movieId) {
    const result = { userBased: null, itemBased: null, actual: ratingMatrix[userId][movieId] || null };

    // User-Based: the N most similar users who rated this movie
    const sims = [];
    for (let u = 1; u <= numUsers; u++) {
        if (u === userId || ratingMatrix[u][movieId] === 0) continue;
        const sim = cosineSimilarity(ratingMatrix[userId], ratingMatrix[u]);
        if (sim > 0) sims.push({ userId: u, sim });
    }
    sims.sort((a, b) => b.sim - a.sim);
    const userPrediction = predictFromNeighbors(sims.slice(0, NUM_NEIGHBORS), movieId);
    if (userPrediction) result.userBased = userPrediction;

    // Item-Based: the most similar movies this user has rated
    const target = getItemSimilarities(movieId);
    const active = ratingMatrix[userId];
    const neighbors = [];
    for (const movie of movies) {
        if (movie.id === movieId || active[movie.id] === 0) continue;
        if (target[movie.id] > 0) neighbors.push({ id: movie.id, sim: target[movie.id] });
    }
    neighbors.sort((a, b) => b.sim - a.sim);
    const top = neighbors.slice(0, ITEM_NEIGHBORS);
    let weighted = 0, weights = 0;
    for (const { id, sim } of top) { weighted += sim * active[id]; weights += sim; }
    if (weights > 0) result.itemBased = { score: weighted / weights, support: top.length };

    return result;
}

// Read the selected user and render both recommendation lists
function getRecommendations() {
    const selectElement = document.getElementById('user-select');
    const userId = parseInt(selectElement.value, 10);

    if (isNaN(userId)) {
        renderList('user-based-result', [], 'Please select a user first.');
        renderList('item-based-result', [], 'Please select a user first.');
        return;
    }

    renderList('user-based-result', [], 'Computing...');
    renderList('item-based-result', [], 'Computing...');

    // Let the browser paint the "Computing..." message before the heavy work
    setTimeout(() => {
        const t0 = performance.now();
        const userBased = getUserBasedRecommendations(userId);
        const t1 = performance.now();
        const itemBased = getItemBasedRecommendations(userId);
        const t2 = performance.now();

        renderList('user-based-result', userBased, null,
            'Because you are similar to other users, we recommend:',
            item => `${item.support} of your ${NUM_NEIGHBORS} closest users rated it`,
            t1 - t0);
        renderList('item-based-result', itemBased, null,
            'Because you liked similar movies, we recommend:',
            item => `because you liked ${item.because}`,
            t2 - t1);
    }, 30);
}

// Read the selected user and movie and render both predicted ratings
function getPrediction() {
    const userId = parseInt(document.getElementById('user-select').value, 10);
    const movieId = parseInt(document.getElementById('movie-select').value, 10);
    const box = document.getElementById('prediction-result');

    if (isNaN(userId) || isNaN(movieId)) {
        box.innerHTML = '<p>Select a user and a movie first.</p>';
        return;
    }

    box.innerHTML = '<p>Computing...</p>';
    setTimeout(() => {
        const p = predictRating(userId, movieId);
        const cell = (label, pred, note) => `
            <div class="prediction-cell">
                <span class="prediction-label">${label}</span>
                <span class="prediction-value">${pred ? pred.score.toFixed(1) : '&ndash;'}</span>
                <span class="prediction-note">${pred ? note(pred.support) : 'not enough data'}</span>
            </div>`;
        const actual = p.actual
            ? `<p class="prediction-actual">This user has already rated this movie: ${p.actual}. Both predictions are built from other ratings, so you can compare them with the real one.</p>`
            : '';
        box.innerHTML = `
            <div class="prediction-grid">
                ${cell('User-Based CF', p.userBased, n => `from ${n} similar users`)}
                ${cell('Item-Based CF', p.itemBased, n => `from ${n} similar movies`)}
            </div>${actual}`;
    }, 30);
}

// Render a list of { title, score } into the given element
function renderList(elementId, items, message, intro, explain, ms) {
    const el = document.getElementById(elementId);

    if (message) {
        el.innerHTML = `<p>${message}</p>`;
        return;
    }

    if (!items || items.length === 0) {
        el.innerHTML = '<p>Not enough overlap with other users to recommend anything for this user.</p>';
        return;
    }

    const entries = items
        .map(item => `<li><span class="rec-title">${item.title}</span>
            <span class="rec-score">${Number(item.score).toFixed(2)}</span>
            ${explain ? `<span class="rec-why">${explain(item)}</span>` : ''}</li>`)
        .join('');
    const timing = ms !== undefined ? `<p class="timing">computed in ${Math.round(ms)} ms</p>` : '';
    el.innerHTML = `${intro ? `<p class="intro">${intro}</p>` : ''}<ol>${entries}</ol>${timing}`;
}
