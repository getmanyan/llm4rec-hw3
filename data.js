// ---------------------------------------------------------------------------
// data.js — loading, parsing and the user-item rating matrix (HW3)
//
// Changes vs. the week3 starter (same three parsing issues found in HW2):
//   1. u.item has 19 genre flags (the first one is "unknown"); the starter list
//      had only 18 names, so every genre was shifted by one position.
//   2. u.item is Latin-1 encoded; reading it as UTF-8 garbles 9 titles.
//   3. 18 titles appear twice under two different ids. For CF this matters more
//      than for content-based filtering: the ratings of one film are split
//      between two matrix columns, and the same film can be recommended twice.
//      Duplicates are merged into the id with more ratings (see mergeDuplicates).
// The dataset files themselves are not modified.
// ---------------------------------------------------------------------------

// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // number of movie lines parsed from u.item (= highest movie id, 1682)
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"

// Lookup helpers
let movieById = {};        // id -> { id, title, genres, numRatings }
let duplicateOf = {};      // duplicate id -> canonical id

// Genre names as defined in the u.item file (19 flags, "unknown" first)
const genreNames = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data (Latin-1, see note 2 above)
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBuffer);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        mergeDuplicates();

        // Derive matrix dimensions, then build the rating matrix
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const errorTarget = document.getElementById('user-based-result');
        if (errorTarget) {
            errorTarget.innerHTML = `<p class="error">Error: ${error.message}. Please make sure u.item and u.data are in the correct location.</p>`;
        }
        throw error; // Re-throw so script.js can handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];

        // Extract genres (last 19 fields)
        const genreValues = fields.slice(5, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        movies.push({ id, title, genres, numRatings: 0 });
        numMovies = Math.max(numMovies, id);
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines

        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);

        ratings.push({ userId, itemId, rating, timestamp });
    }
}

// Merge movies that appear twice in u.item under different ids.
// The id with more ratings is kept; ratings of the other id are re-pointed to it.
function mergeDuplicates() {
    const counts = {};
    for (const r of ratings) counts[r.itemId] = (counts[r.itemId] || 0) + 1;

    const byTitle = {};
    for (const m of movies) {
        m.numRatings = counts[m.id] || 0;
        (byTitle[m.title] = byTitle[m.title] || []).push(m);
    }

    duplicateOf = {};
    for (const group of Object.values(byTitle)) {
        if (group.length < 2) continue;
        group.sort((a, b) => b.numRatings - a.numRatings || a.id - b.id);
        for (const dup of group.slice(1)) {
            duplicateOf[dup.id] = group[0].id;
            group[0].numRatings += dup.numRatings;
        }
    }

    movies = movies.filter(m => !(m.id in duplicateOf));
    for (const r of ratings) {
        if (r.itemId in duplicateOf) r.itemId = duplicateOf[r.itemId];
    }

    movieById = {};
    for (const m of movies) movieById[m.id] = m;
}

// ---------------------------------------------------------------------------
// Build the user-item rating matrix.
//
// Shape: (numUsers + 1) x (numMovies + 1), indexed by raw id, so that
//   ratingMatrix[userId][movieId] === rating
// A missing entry is 0. MovieLens ratings are 1-5, so 0 is unambiguous and
// works as the "not rated" mask: no separate boolean matrix is needed.
// Columns of merged duplicate ids simply stay empty.
// If a user rated both copies of a duplicated film, the two ratings are averaged.
// Each row is a Float32Array so similarity loops run on typed arrays.
// ---------------------------------------------------------------------------
function buildRatingMatrix() {
    ratingMatrix = [];
    for (let u = 0; u <= numUsers; u++) {
        ratingMatrix.push(new Float32Array(numMovies + 1));
    }

    for (const r of ratings) {
        const row = ratingMatrix[r.userId];
        if (row[r.itemId] === 0) {
            row[r.itemId] = r.rating;
        } else {
            // Same user rated both copies of a merged duplicate: average them
            row[r.itemId] = (row[r.itemId] + r.rating) / 2;
        }
    }
}
